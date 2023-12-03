const puppeteer = require('puppeteer');
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
            '--window-position=0,0',
            `--window-size=${1620},${1080}`,
            '--ignore-certifcate-errors',
            '--ignore-certifcate-errors-spki-list',
            '--user-agent="Chrome/97.0.4692.45 Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Safari/537.36"',
            //'--headless',
            '--enable-background-networking',
            '--enable-features=NetworkService,NetworkServiceInProcess',
            '--disable-background-timer-throttling',
            '--disable-backgrounding-occluded-windows',
            '--disable-breakpad',
            '--enable-client-side-phishing-detection',
            '--disable-hang-monitor',
            '--disable-ipc-flooding-protection',
            '--disable-popup-blocking',
            '--enable-scrollbars',
            '--remote-debugging-port=0'
        ]
    });

    const pid1 = browser.process().pid;
    //await console.log(browser.browserContexts());
    const page = await browser.newPage();

    for (let i = 0; i < url.length; i++) {
        await page.goto(url[0], { waitUntil: 'load' });

        try {
            let i = 2;
            while (true) {
                await page.evaluate((i) => {
                    let link = document.getElementsByTagName('a');
                        return link[i].click();
                },i);
                // await page.click('a')[i];
                await page.waitForTimeout(200);
                i += 1;
            }
        } catch (error) {
            await console.log("Did work and work is done");
        }
    }

    await page.waitForTimeout(30000);

    await page.close();
    await page.waitForTimeout(200);
    let shell = require('shelljs');
    await shell.exec('TASKKILL /PID ' + pid1 + ' /F');
    //await shell.exec('TASKKILL /?');
    await process.exit();
})();