// Screenshot a list of HTML files with the preinstalled Playwright (node). Usage: node screenshot.js out_dir a.html b.html ...
const { chromium } = require('playwright');
(async () => {
  const [outDir, ...files] = process.argv.slice(2);
  const b = await chromium.launch();
  const pg = await b.newPage({ viewport: { width: 1100, height: 640 }, deviceScaleFactor: 1 });
  for (const f of files) {
    await pg.goto('file://' + f);
    await pg.screenshot({ path: outDir + '/' + require('path').basename(f).replace('.html', '.png') });
  }
  await b.close();
})();
