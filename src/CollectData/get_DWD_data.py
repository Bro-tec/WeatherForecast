# import requests
import aiohttp
import asyncio
import random
import time
from datetime import datetime as dt
from datetime import timedelta as td

from datetime import datetime


async def getSourcesByStationIDDate(i, date, wait=True, delay=2):
    if wait:
        await asyncio.sleep(random.uniform(0, delay))
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://api.brightsky.dev/weather?dwd_station_id={i}&date={date}"
            ) as response:
                if str(response.status) == "429":
                    return await getSourcesByStationIDDate(i, date, delay=10)
                elif (
                    str(response.status)[0] == "2"
                ):  # raises exception when not a 2xx response
                    WeatherData = await response.json()
                    if "sources" in WeatherData.keys():
                        if len(WeatherData["sources"]) != 0:
                            first_record = datetime.strptime(
                                WeatherData["sources"][0]["first_record"][0:14],
                                "%Y-%m-%dT%H:",
                            )
                            last_record = datetime.strptime(
                                WeatherData["sources"][0]["last_record"][0:14],
                                "%Y-%m-%dT%H:",
                            )
                            return [
                                WeatherData["sources"][0]["dwd_station_id"],
                                WeatherData["sources"][0]["station_name"],
                                WeatherData["sources"][0]["height"],
                                WeatherData["sources"][0]["lat"],
                                WeatherData["sources"][0]["lon"],
                                first_record,
                                last_record,
                            ]
                        else:
                            return "error", "error"
                    else:
                        return "error", "error"
                else:
                    return "error", str(response.status)
    except Exception:
        return "error", "error4"


async def getWeatherByStationIDDate(i, date, wait=True, delay=2):
    if wait:
        await asyncio.sleep(random.uniform(0, delay))
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://api.brightsky.dev/weather?dwd_station_id={i}&date={date}"
            ) as response:
                await asyncio.sleep(0.2)
                # if code was too fast it gets error 429 so it just retries witha little delay
                if str(response.status) == "429":
                    return await getWeatherByStationIDDate(i, date, delay=15)
                elif (
                    str(response.status)[0] == "2"
                ):  # raises exception when not a 2xx response
                    WeatherData = await response.json()
                    if "weather" in WeatherData.keys():
                        if len(WeatherData["weather"]) != 0:
                            return WeatherData["weather"]
                        else:
                            return ["error", "error1"]
                    else:
                        return ["error", "error2"]
                else:
                    return ["error", str(response.status)]
    except Exception:
        print("error gets handled here")
        return ["error", "error4"]


# https://stackoverflow.com/questions/553303/how-to-generate-a-random-date-between-two-other-dates
def str_time_prop(start, time_format, prop):
    today = dt.now() - td(days=3)
    stime = time.mktime(time.strptime(start, time_format))
    etime = time.mktime(
        time.strptime(f"{today.year}-{today.month}-{today.day}", time_format)
    )

    ptime = stime + prop * (etime - stime)

    return time.strftime(time_format, time.localtime(ptime))
