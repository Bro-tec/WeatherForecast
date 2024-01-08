const puppeteer = require('puppeteer');
const shell = require('shelljs');
let url1 = "";
let url = [ "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/subdaily/air_temperature/historical/",
            "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/subdaily/cloudiness/historical/",
            "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/subdaily/extreme_wind/historical/",
            "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/subdaily/moisture/historical/",
            "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/subdaily/pressure/historical/",
            "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/subdaily/wind/historical/"
        ];

(async () => {

    const browser = await puppeteer.launch({
        'Accept-Charset': 'utf-8',
        'Content-Type': 'text/html; charset=utf-8',
        //ignoreDefaultArgs: true,
        headless: false,

        args: [
            '--no-sandbox',
            '--ignoreHTTPSErrors',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-infobars',
            '--window-position=25,25',
            `--window-size=${1620},${1080}`,
            // '--ignore-certifcate-errors',
            // '--ignore-certifcate-errors-spki-list',
            '--user-agent="Chrome/97.0.4692.45 Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Safari/537.36"',
            //'--headless',
            // '--enable-background-networking',
            // '--enable-features=NetworkService,NetworkServiceInProcess',
            // '--disable-background-timer-throttling',
            // '--disable-backgrounding-occluded-windows',
            // '--disable-breakpad',
            // '--enable-client-side-phishing-detection',
            // '--disable-hang-monitor',
            // '--disable-ipc-flooding-protection',
            // '--disable-popup-blocking',
            // '--enable-scrollbars',
            // '--remote-debugging-port=0'
        ]
    });

    const pid1 = browser.process().pid;
    //await console.log(browser.browserContexts());
    let page = await browser.newPage();

    await console.log("start");

    // await page._client.send('Page.setDownloadBehavior', {behavior: 'allow', downloadPath: './'});
    for (let u = 0; u < url.length; u++) {
        await page.goto(url[u], { waitUntil: 'load' });

        try {
            let i = 2;
            while (true) {
            // for (let x = 0; x < 6; x++) {
                await page.evaluate((i) => {
                    let link = document.getElementsByTagName('a');
                    if (link[i].innerText.endsWith(".zip")){
                        return link[i].click();
                    }
                },i);
                // await page.click('a')[i];
                await page.waitForTimeout(100);
                i += 1;
                await console.log("zip: "+ i);
            }
        } catch (error) {
            await console.log("Did work and work is done");
        }
        await page.waitForTimeout(10000);
    }

    await console.log("end");
    await page.waitForTimeout(5000);

    await page.close();
    await page.waitForTimeout(200);
    await shell.exec('TASKKILL /PID ' + pid1 + ' /F');
    //await shell.exec('TASKKILL /?');
    await process.exit();
})();