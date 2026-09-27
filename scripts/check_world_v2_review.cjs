"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {chromium} = require("/path/to/research/.codex/euler20_review_browser/node_modules/playwright");
const root = path.resolve(__dirname, "../runs/registered_roi32_20260919/analysis_30_pcr_v1v4_20260919/review/web");

async function ready(page) {
  await page.waitForFunction(() => document.querySelector("#status").textContent.startsWith("32 slices"), {timeout:60000});
}
async function pixels(page, index=3) {
  return page.locator(".volume-panel canvas").nth(index).evaluate(canvas => {
    const data=canvas.getContext("2d").getImageData(0,0,canvas.width,canvas.height).data;
    const values=[];let low=255,high=0;
    for(let i=0;i<data.length;i+=4){low=Math.min(low,data[i]);high=Math.max(high,data[i]);if(i%124===0)values.push(data[i]);}
    return {low,high,values};
  });
}
async function main() {
  const output=path.join(root,"browser_checks");fs.mkdirSync(output,{recursive:true});
  const browser=await chromium.launch({executablePath:"/path/to/research/.codex/euler20_review_browser/browsers/chromium-1243/chrome-linux64/chrome",headless:true,args:["--no-sandbox"]});
  const errors=[],external=[];
  try {
    const context=await browser.newContext({viewport:{width:1440,height:1050},acceptDownloads:true});
    const page=await context.newPage();
    page.on("pageerror",error=>errors.push(String(error)));
    page.on("request",request=>{if(/^https?:/.test(request.url()))external.push(request.url());});
    await page.goto("file://"+path.join(root,"index.html"));await ready(page);
    assert.equal(await page.locator(".patient-card").count(),30);
    assert.equal(await page.locator(".neighbor").count(),20);
    const original=await pixels(page);assert(original.high-original.low>80);
    await page.screenshot({path:path.join(output,"desktop.png"),fullPage:false});
    for(let i=0;i<30;i++){
      await page.selectOption("#case",String(i));await ready(page);
      const value=await pixels(page);assert(value.high-value.low>40,"Blank case "+(i+1));
    }
    await page.selectOption("#case","0");await ready(page);
    for(const index of [1,2,0,3,0])await page.selectOption("#case",String(index));
    await ready(page);await page.waitForFunction(()=>resolvers.size===0);
    assert.deepEqual((await pixels(page)).values,original.values);
    await page.selectOption('[aria-label="Generated sample"]',"sample2");const sample2=await pixels(page,2);
    await page.selectOption('[aria-label="Generated sample"]',"sample1");assert.notDeepEqual((await pixels(page,2)).values,sample2.values);
    await page.click('[data-phase="0"]');assert.notDeepEqual((await pixels(page)).values,original.values);
    await page.click('[data-phase="2"]');assert((await pixels(page)).high-(await pixels(page)).low>40);
    await page.click('[data-phase="1"]');await page.selectOption("#channel","early");assert.notDeepEqual((await pixels(page)).values,original.values);
    await page.selectOption("#channel","late");await page.selectOption("#channel","raw");
    await page.uncheck("#mask");assert.notDeepEqual((await pixels(page)).values,original.values);
    await page.check("#mask");
    await page.locator("#slice").fill("0");assert.equal(await page.locator(".neighbor span").filter({hasText:"N/A"}).count(),10);
    await page.locator("#slice").fill("31");assert.equal(await page.locator(".neighbor span").filter({hasText:"N/A"}).count(),10);
    await page.click("#anchor");assert.deepEqual((await pixels(page)).values,original.values);
    await page.locator("#zoom").fill("2");assert.notDeepEqual((await pixels(page)).values,original.values);await page.locator("#zoom").fill("1");
    for(const mode of ["mean","sample3","sample4","source","target"])await page.selectOption("#neighbor-method",mode);
    await page.locator(".volume-panel button").nth(3).click();assert(await page.locator("#image-dialog").isVisible());await page.click("#close-dialog");
    const downloadPromise=page.waitForEvent("download");await page.click("#download");const download=await downloadPromise;await download.saveAs(path.join(output,"comparison_download.png"));
    for(const width of [1920,390,360]){
      await page.setViewportSize({width,height:900});
      const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);assert(!overflow,"Viewport overflow "+width);
      await page.screenshot({path:path.join(output,"viewport_"+width+".png"),fullPage:false});
      const overlaps=await page.locator(".toolbar button,.toolbar select,.volume-panel h3").evaluateAll(items=>items.filter(el=>el.scrollWidth>el.clientWidth+1).map(el=>el.outerHTML));
      assert.deepEqual(overlaps,[]);
    }
    assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
    const result={passed:true,patients:30,canvases_nonblank:true,draw_phase_slice_mask_zoom_switches:true,neighbors:20,
      unavailable_edge_slices_explicit:true,rapid_case_switches:true,download_verified:true,viewports:[1440,1920,390,360],external_requests:0,browser_errors:errors};
    fs.writeFileSync(path.join(output,"verification.json"),JSON.stringify(result,null,2)+"\n");console.log(JSON.stringify(result));
    await context.close();
  } finally { await browser.close(); }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
