import requests
import aiohttp
import asyncio
import random
from datetime import datetime

# Response = requests.get("https://api.brightsky.dev/weather?lat=52&lon=7.6&date=2020-04-21")

def getSourcesByStationIDDate(i, date):
    # print(f"https://api.brightsky.dev/weather?dwd_station_id={i}&date=2020-04-21")
    Response = requests.get(f"https://api.brightsky.dev/weather?dwd_station_id={i}&date={date}")
    if str(Response.status_code)[0] == "2": # raises exception when not a 2xx response
        WeatherData = Response.json()
        if "sources" in WeatherData.keys():
            if len(WeatherData["sources"]) != 0:
                first_record = datetime.strptime(WeatherData["sources"][0]["first_record"][0:14], '%Y-%m-%dT%H:')
                last_record = datetime.strptime(WeatherData["sources"][0]["last_record"][0:14], '%Y-%m-%dT%H:')
                return [WeatherData["sources"][0]["dwd_station_id"],WeatherData["sources"][0]["station_name"],WeatherData["sources"][0]["height"],WeatherData["sources"][0]["lat"],WeatherData["sources"][0]["lon"],first_record,last_record]
            else:
                return "error","error"
        else:
            return "error","error"
    else:
        return "error","error"

async def getWeatherByStationIDDate(i, date, wait=True, delay=2):
    if wait:
        await asyncio.sleep(random.uniform(0, delay))
    try:
        # print(f"https://api.brightsky.dev/weather?dwd_station_id={i}&date=2020-04-21")
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.brightsky.dev/weather?dwd_station_id={i}&date={date}") as response:
                await asyncio.sleep(0.2)
                # print(i, ":", response.status)
                if str(response.status) == "429":
                    # print(i, ":", "redoing")
                    # await asyncio.sleep(random.uniform(2, 4))
                    return await getWeatherByStationIDDate(i, date, delay=15)
                elif str(response.status)[0] == "2": # raises exception when not a 2xx response
                    WeatherData = await response.json()
                    if "weather" in WeatherData.keys():
                        if len(WeatherData["weather"]) != 0:
                            # first_record = datetime.strptime(WeatherData["weather"][0]["first_record"][0:14], '%Y-%m-%dT%H:')
                            # last_record = datetime.strptime(WeatherData["weather"][0]["last_record"][0:14], '%Y-%m-%dT%H:')
                            return WeatherData["weather"]
                        else:
                            return "error","error"
                    else:
                        return "error","error"
                else:
                    return "error","error"
    except Exception:
        return "error","error"


if __name__ == "__main__":
    from tqdm import tqdm
    import os.path
    import pandas as pd

    if os.path.isfile("stations.csv"):
        print("file already exists")
    else:
        # print(stations("01766"))
        stations_list = []
        for i in tqdm(range(0,100000)):
            stri = str(i)
            for ch in range(0,5-len(str(i))):
                stri = "0" + stri
            data = getSourcesByStationIDDate(stri, "2020-04-21")
            if data[0] != "error":
                stations_list.append(data)
        # stations_df = pd.DataFrame(columns=["ID","Name","height","lat","lon","start","end"])
        stations_df = pd.DataFrame(stations_list, columns=["ID","Name","height","lat","lon","start","end"])
        stations_df.to_csv("stations.csv", sep=',', index=False, encoding='utf-8')
        

