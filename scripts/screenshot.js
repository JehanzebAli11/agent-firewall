// Screenshot a list of HTML files with the preinstalled Playwright (node).
// Usage: [SHOT_W=1100 SHOT_H=640 SHOT_DSF=1] node screenshot.js out_dir a.html b.html ...
const { chromium } = require('playwright');
(async () => {
  const [outDir, ...files] = process.argv.slice(2);
  const b = await chromium.launch();
  const pg = await b.newPage({ viewport: { width: +(process.env.SHOT_W || 1100), height: +(process.env.SHOT_H || 640) }, deviceScaleFactor: +(process.env.SHOT_DSF || 1) });
  for (const f of files) {
    await pg.goto('file://' + f);
    await pg.screenshot({ path: outDir + '/' + require('path').basename(f).replace('.html', '.png') });
  }
  await b.close();
})();
