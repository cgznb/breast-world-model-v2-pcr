"use strict";
const $ = id => document.getElementById(id);
const cases = window.WORLD_CASES;
let selected = 0, current = null, phase = 1, sample = "sample1", loading = 0;
const promises = new Map(), resolvers = new Map(), decodedKeys = new Set();
const methods = [["source", "Observed source"], ["target", "True follow-up"], ["sample", "World V2 draw"], ["mean", "World V2 mean of 4"], ["target_vq", "True follow-up VQ"]];
const canvases = new Map();
window.registerWorldCase = (key, data) => { const resolve=resolvers.get(key); if(resolve){resolvers.delete(key);resolve(data);} };

function image(src) {
  return new Promise((resolve, reject) => { const im = new Image(); im.onload = () => resolve(im); im.onerror = reject; im.src = src; });
}
async function decode(key) {
  if (!promises.has(key)) {
    const work = new Promise((resolve, reject) => {
      resolvers.set(key, resolve);
      const script = document.createElement("script");
      script.src = "cases/" + key + ".js";
      script.onerror = () => reject(new Error("Cannot load " + key));
      script.onload = () => script.remove();
      document.head.append(script);
    }).then(async packed => {
      const result = {volumes: {}};
      for (const [name, item] of Object.entries(packed.volumes)) {
        const im = await image(item.png), canvas = document.createElement("canvas");
        canvas.width = im.width; canvas.height = im.height;
        const ctx = canvas.getContext("2d", {willReadFrequently:true}); ctx.drawImage(im, 0, 0);
        const bytes = ctx.getImageData(0, 0, im.width, im.height).data;
        const values = new Float32Array(3 * 32 * 128 * 128);
        for (let i=0;i<values.length;i++) values[i] = (bytes[4*i]*256 + bytes[4*i+1])*item.scale + item.lower;
        result.volumes[name] = values;
        canvas.width = canvas.height = 0;
      }
      const im = await image(packed.mask), canvas = document.createElement("canvas");
      canvas.width = 128; canvas.height = 4096;
      const ctx = canvas.getContext("2d", {willReadFrequently:true}); ctx.drawImage(im,0,0);
      const bytes = ctx.getImageData(0,0,128,4096).data;
      result.mask = new Uint8Array(32*128*128);
      for (let i=0;i<result.mask.length;i++) result.mask[i] = bytes[4*i] > 0 ? 1 : 0;
      canvas.width = canvas.height = 0;
      decodedKeys.add(key);
      return result;
    });
    promises.set(key, work);
    work.catch(()=>{promises.delete(key);resolvers.delete(key);decodedKeys.delete(key);});
  }
  const result = await promises.get(key);
  for (const cached of [...promises.keys()]) if (cached !== key && cached !== cases[selected].case && decodedKeys.has(cached)) { promises.delete(cached); decodedKeys.delete(cached); }
  return result;
}
function valueAt(values, p, z, i) { return values[(p*32+z)*16384+i]; }
function draw(canvas, name, z) {
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#0c0d0d"; ctx.fillRect(0,0,canvas.width,canvas.height);
  if (!current) return;
  if (z<0 || z>31) { ctx.fillStyle="#9da59f"; ctx.font="12px Arial"; ctx.textAlign="center"; ctx.fillText("Unavailable",64,68); return; }
  const mode = $("channel").value, info = cases[selected], values = current.volumes[name];
  const imageData = new ImageData(128,128), out = imageData.data;
  const [lower,upper] = info.window, limit = info.delta_window;
  for (let i=0;i<16384;i++) {
    let v = valueAt(values,phase,z,i), r,g,b;
    if (mode !== "raw") {
      v = valueAt(values,mode === "early" ? 1 : 2,z,i)-valueAt(values,0,z,i);
      const t = Math.max(-1,Math.min(1,v/limit));
      r = t<0 ? 246+ t*193 : 246-t*66;
      g = 246-Math.abs(t)*198;
      b = t>0 ? 246-t*198 : 246+t*89;
    } else { r=g=b=255*Math.max(0,Math.min(1,(v-lower)/(upper-lower))); }
    const offset=z*16384+i, x=i%128, y=Math.floor(i/128), m=current.mask;
    if ($("mask").checked && m[offset] && (x===0 || x===127 || y===0 || y===127 || !m[offset-1] || !m[offset+1] || !m[offset-128] || !m[offset+128])) { r=56;g=212;b=158; }
    out[4*i]=r;out[4*i+1]=g;out[4*i+2]=b;out[4*i+3]=255;
  }
  const scratch=document.createElement("canvas");scratch.width=scratch.height=128;
  scratch.getContext("2d").putImageData(imageData,0,0);
  const zoom=Number($("zoom").value), size=128/zoom, margin=(128-size)/2;
  ctx.imageSmoothingEnabled=false;ctx.drawImage(scratch,margin,margin,size,size,0,0,canvas.width,canvas.height);
}
function render() {
  const z=Number($("slice").value);
  $("slice-value").textContent=z+" / 31";
  for (const [name,canvas] of canvases) draw(canvas,name==="sample"?sample:name,z);
  for (const element of $("neighbors").children) {
    const offset=Number(element.dataset.offset), position=z+offset;
    draw(element.querySelector("canvas"),$("neighbor-method").value,position);
    element.querySelector("span").textContent=(offset>0?"+":"")+offset+" | "+(position>=0&&position<32?"z="+position:"N/A");
  }
}
async function choose(index) {
  selected=(index+cases.length)%cases.length;
  const ticket=++loading, info=cases[selected];
  current=null;$("case").value=String(selected);$("slice").value=String(info.slice);
  $("case-info").textContent=info.case+" | "+info.transition+" | "+info.days+" days"+(info.clipped?" | crop boundary":"");
  $("status").textContent="Loading volume";
  $("case-png").href="../"+info.case+"/comparison.png";
  $("enhancement-png").href="../"+info.case+"/enhancement.png";
  $("volume-link").href="../"+info.case+"/volumes.npz";
  document.querySelectorAll(".patient-card").forEach((item,i)=>item.classList.toggle("active",i===selected));
  render();
  try { const data=await decode(info.case); if(ticket!==loading)return; current=data; $("status").textContent="32 slices | 2.0 mm axial | 0.7032 mm in-plane";render(); }
  catch(error) { if(ticket===loading)$("status").textContent=String(error); }
}
for (const [name,title] of methods) {
  const figure=document.createElement("figure");figure.className="volume-panel";
  const label=document.createElement("h3");
  if(name==="sample") {
    const select=document.createElement("select"); select.setAttribute("aria-label","Generated sample");
    for(let i=1;i<=4;i++){const option=new Option("World V2 draw "+i,"sample"+i);select.add(option);}
    select.addEventListener("change",()=>{sample=select.value;render();});label.append(select);
  } else label.textContent=title;
  const button=document.createElement("button");button.title="Enlarge "+title;button.setAttribute("aria-label","Enlarge "+title);
  const canvas=document.createElement("canvas");canvas.width=canvas.height=256;button.append(canvas);canvases.set(name,canvas);
  button.addEventListener("click",()=>{$("dialog-title").textContent=title+" | "+cases[selected].case+" | z="+$("slice").value;draw($("dialog-canvas"),name==="sample"?sample:name,Number($("slice").value));$("image-dialog").showModal();});
  figure.append(label,button);$("comparison").append(figure);
}
for(const offset of [...Array.from({length:10},(_,i)=>i-10),...Array.from({length:10},(_,i)=>i+1)]) {
  const item=document.createElement("div");item.className="neighbor";item.dataset.offset=String(offset);
  const canvas=document.createElement("canvas");canvas.width=canvas.height=128;item.append(canvas,document.createElement("span"));$("neighbors").append(item);
}
cases.forEach((info,i)=>{
  $("case").add(new Option(info.case+" | "+info.transition,String(i)));
  const button=document.createElement("button");button.className="patient-card";
  const img=document.createElement("img");img.src="../"+info.case+"/comparison.png";img.alt=info.case+" three-phase MRI comparison";img.loading="lazy";
  const label=document.createElement("span");label.textContent=info.case+" | "+info.transition;
  button.append(img,label);button.addEventListener("click",()=>{choose(i);window.scrollTo({top:0,behavior:"smooth"});});$("patient-grid").append(button);
});
$("case").addEventListener("change",()=>choose(Number($("case").value)));
$("prev").addEventListener("click",()=>choose(selected-1));$("next").addEventListener("click",()=>choose(selected+1));
$("phases").addEventListener("click",event=>{const button=event.target.closest("[data-phase]");if(!button)return;phase=Number(button.dataset.phase);$("channel").value="raw";document.querySelectorAll("[data-phase]").forEach(b=>b.classList.toggle("active",b===button));render();});
for(const id of ["slice","zoom","mask","channel","neighbor-method"])$(id).addEventListener("input",render);
$("anchor").addEventListener("click",()=>{$("slice").value=String(cases[selected].slice);render();});
$("close-dialog").addEventListener("click",()=>$("image-dialog").close());
$("download").addEventListener("click",()=>{if(!current)return;const canvas=document.createElement("canvas");canvas.width=1280;canvas.height=292;const ctx=canvas.getContext("2d");ctx.fillStyle="#fff";ctx.fillRect(0,0,1280,292);ctx.fillStyle="#202324";ctx.font="13px Arial";let i=0;for(const[name,c]of canvases){ctx.fillText(methods[i][1],i*256+6,22);ctx.drawImage(c,i*256,36);i++;}const a=document.createElement("a");a.download=cases[selected].case+"_z"+$("slice").value+".png";a.href=canvas.toDataURL("image/png");a.click();});
lucide.createIcons();choose(0);
