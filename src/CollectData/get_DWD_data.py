# import requests
import aiohttp
import asyncio
import random
import time
from datetime import datetime as dt
from datetime import timedelta as td

# import json
from datetime import datetime


# def add_time(rand_dates, date, time_format):
#     rand_dates.append(
#         time.strftime(
#             "%Y-%m-%d",
#             time.localtime(time.mktime(time.strptime(date, time_format))),
#         )
#     )


# def str_time_prop(start, time_format, prop):
#     today = dt.now() - td(days=3)
#     stime = time.mktime(time.strptime(start, time_format))
#     etime = time.mktime(
#         time.strptime(f"{today.year}-{today.month}-{today.day}", time_format)
#     )

#     ptime = stime + prop * (etime - stime)

#     return time.strftime(time_format, time.localtime(ptime))


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


# if __name__ == "__main__":
#     from tqdm import tqdm
#     import os.path
#     import pandas as pd
#     import time
#     from datetime import datetime as dt
#     from datetime import timedelta as td
#     import get_DWD_data as gdwd
#     import get_learning_data as gld
#     from random import random as random

#     if os.path.isfile("stations.csv"):
#         print("file already exists")
#         print("cleaning stations.csv")
#         stations = gld.load_stations_csv()
#         stations["label_vals"] = False
#         stations["label_icon"] = False
#         stations["label_condition"] = False
#         rand_dates = [
#             str_time_prop("2018-01-01", "%Y-%m-%d", random()) for i in range(15)
#         ]
#         rand_dates.append(
#             time.strftime(
#                 "%Y-%m-%d",
#                 time.localtime(time.mktime(time.strptime("2024-09-12", "%Y-%m-%d"))),
#             )
#         )
#         for si, st in tqdm(enumerate(stations.iterrows()), total=stations.shape[0]):
#             # if str(st[1]["ID"]) == "00006":
#             #     continue
#             # print(si, ", ", st[1]["ID"], ", ", rd)
#             datas = gld.getWeatherData(st[1]["ID"], rand_dates)
#             for data in datas:
#                 # print(data[0])
#                 if data[0] != "error":
#                     for di in range(len(data)):
#                         # print(stations.loc[si, "label_icon"])
#                         if data[di]["icon"] != None:
#                             stations.loc[si, "label_icon"] = True
#                         if data[di]["condition"] != None:
#                             stations.loc[si, "label_condition"] = True
#                         if (
#                             data[di]["precipitation"] != None
#                             or data[di]["pressure_msl"] != None
#                             or data[di]["sunshine"] != None
#                             or data[di]["temperature"] != None
#                             or data[di]["wind_direction"] != None
#                             or data[di]["wind_speed"] != None
#                             or data[di]["cloud_cover"] != None
#                             or data[di]["dew_point"] != None
#                             or data[di]["relative_humidity"] != None
#                             or data[di]["visibility"] != None
#                             or data[di]["wind_gust_direction"] != None
#                             or data[di]["wind_gust_speed"] != None
#                             or data[di]["condition"] != None
#                             or data[di]["precipitation_probability"] != None
#                             or data[di]["precipitation_probability_6h"] != None
#                             or data[di]["solar"] != None
#                         ):
#                             stations.loc[si, "label_vals"] = True
#                     # print(stations.iloc[si, :])
#         stations.to_csv("stations.csv", sep=",", index=False, encoding="utf-8")

# else:
#     print("file doesn't exists")
#     print("creating stations.csv")
#     # print(stations("01766"))
#     stations_list = []
#     for i in tqdm(range(0, 99999)):
#         stri = str(i)
#         for ch in range(0, 5 - len(str(i))):
#             stri = "0" + stri
#         rand_dates = [
#             str_time_prop("2016-01-01", "%Y-%m-%d", random()) for i in range(2)
#         ]
#         rand_dates.append(
#             time.strftime(
#                 "%Y-%m-%d",
#                 time.localtime(time.mktime(time.strptime("2024-09-12", "%Y-%m-%d"))),
#             )
#         )
#         for rd in rand_dates:
#             data = gdwd.getSourcesByStationIDDate(stri, rd)
#             if data[0] != "error":
#                 stations_list.append(data)
#                 break
#     # stations_df = pd.DataFrame(columns=["ID","Name","height","lat","lon","start","end"])
#     stations_df = pd.DataFrame(
#         stations_list,
#         columns=["ID", "Name", "height", "lat", "lon", "start", "end"],
#     )
#     stations_df.to_csv("stations.csv", sep=",", index=False, encoding="utf-8")
#     if os.path.isfile("stations.csv"):
#         print("file already exists")
#         print("cleaning stations.csv")
#         stations = gld.load_stations_csv()
#         stations["label_vals"] = False
#         stations["label_icon"] = False
#         stations["label_condition"] = False
#         rand_dates = [
#             str_time_prop("2016-01-01", "%Y-%m-%d", random()) for i in range(25)
#         ]
#         rand_dates.append(
#             time.strftime(
#                 "%Y-%m-%d",
#                 time.localtime(time.mktime(time.strptime("2024-09-12", "%Y-%m-%d"))),
#             )
#         )
#         for si, st in tqdm(enumerate(stations.iterrows()), total=stations.shape[0]):
#             # print(si, ", ", st[1]["ID"], ", ", rd)
#             datas = gld.getWeatherData(st[1]["ID"], rand_dates)
#             for data in datas:
#                 # print(data[0])
#                 if data[0] != "error":
#                     for di in range(len(data)):
#                         if data[di]["icon"] != None:
#                             stations.loc[si, "label_icon"] = True
#                         if data[di]["condition"] != None:
#                             stations.loc[si, "label_condition"] = True
#                         if (
#                             data[di]["precipitation"] != None
#                             or data[di]["pressure_msl"] != None
#                             or data[di]["sunshine"] != None
#                             or data[di]["temperature"] != None
#                             or data[di]["wind_direction"] != None
#                             or data[di]["wind_speed"] != None
#                             or data[di]["cloud_cover"] != None
#                             or data[di]["dew_point"] != None
#                             or data[di]["relative_humidity"] != None
#                             or data[di]["visibility"] != None
#                             or data[di]["wind_gust_direction"] != None
#                             or data[di]["wind_gust_speed"] != None
#                             or data[di]["condition"] != None
#                             or data[di]["precipitation_probability"] != None
#                             or data[di]["precipitation_probability_6h"] != None
#                             or data[di]["solar"] != None
#                         ):
#                             stations.loc["label_vals", si] = True
#                     # print(stations.iloc[si, :])
#         stations.to_csv("stations.csv", sep=",", index=False, encoding="utf-8")
