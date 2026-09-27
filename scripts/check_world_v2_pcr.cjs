"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");
const path=require("node:path");
const {chromium}=require("/path/to/research/.codex/euler20_review_browser/node_modules/playwright");
const root=path.resolve(__dirname,"../runs/registered_roi32_20260919/analysis_30_pcr_v1v4_20260919/pcr/evaluation");
async function main(){
 const browser=await chromium.launch({executablePath:"/path/to/research/.codex/euler20_review_browser/browsers/chromium-1243/chrome-linux64/chrome",headless:true,args:["--no-sandbox"]});
 const errors=[],external=[];const output=path.join(root,"browser_checks");fs.mkdirSync(output,{recursive:true});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  page.on("pageerror",error=>errors.push(String(error)));page.on("request",r=>{if(/^https?:/.test(r.url()))external.push(r.url());});
  await page.goto("file://"+path.join(root,"index.html"));
  let combinations=0;
  for(const [model,depths]of[["v1",[1,2,3,4]],["v4",[4]]]){
   await page.click('[data-model="'+model+'"]');
   for(const depth of depths){
    await page.selectOption("#window",String(depth));
    for(const statistic of["mean","ensemble","0","1","2","3","4"]){
     await page.selectOption("#statistic",statistic);
     for(const metric of["auroc","ap","brier","logloss","sensitivity","specificity","accuracy","f1"]){
      await page.selectOption("#metric",metric);
      assert.equal(await page.locator("tbody tr").count(),10);
      const actual=await page.locator("tbody tr").first().locator("td").allTextContents();
      const expected=await page.evaluate(({model,depth,statistic,metric})=>{
       const data=statistic==="mean"?PCR_RESULTS.summary:statistic==="ensemble"?PCR_RESULTS.ensemble:PCR_RESULTS.folds;
       return["42",...["real","copy_T0","direct_mc4","rollout_mc4","previous_real_mc4"].map(source=>{
        const row=data.find(r=>r.model===model&&r.depth===depth&&r.seed===42&&r.source===source&&(statistic==="mean"||statistic==="ensemble"||r.outer===Number(statistic)));
        return statistic==="mean"?row[metric+"_mean"].toFixed(4)+" +/- "+row[metric+"_fold_sd"].toFixed(4):row[metric].toFixed(4);
       })];
      },{model,depth,statistic,metric});
      assert.deepEqual(actual,expected);combinations++;
     }
    }
   }
  }
  assert.equal(await page.locator("#window option:disabled").count(),3);
  await page.click('[data-model="v1"]');await page.selectOption("#window","4");await page.selectOption("#statistic","mean");await page.selectOption("#metric","auroc");
  for(const width of[1440,1920,390,360]){
   await page.setViewportSize({width,height:1000});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
   assert(await page.locator(".results-plot").evaluateAll(images=>images.every(im=>im.complete&&im.naturalWidth>0)));
   await page.screenshot({path:path.join(output,"viewport_"+width+".png"),fullPage:false});
  }
  const links=await page.locator("a[href]").evaluateAll(elements=>elements.map(a=>a.getAttribute("href")));
  for(const link of links)assert(fs.existsSync(path.resolve(root,link)),"Missing file "+link);
  assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
  const result={passed:true,metric_view_combinations:combinations,seeds_per_view:10,unavailable_v4_windows_disabled:true,
   viewports:[1440,1920,390,360],plots_loaded:true,links_checked:links.length,external_requests:0,browser_errors:errors};
  fs.writeFileSync(path.join(output,"verification.json"),JSON.stringify(result,null,2)+"\n");console.log(JSON.stringify(result));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
