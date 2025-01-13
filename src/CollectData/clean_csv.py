import time
from datetime import datetime as dt
from datetime import timedelta as td
import get_DWD_data as gdwd
import get_learning_data as gld
from random import random as random

from tqdm import tqdm
import os.path
import pandas as pd


# https://stackoverflow.com/questions/553303/how-to-generate-a-random-date-between-two-other-dates
def str_time_prop(start, time_format, prop):
    today = dt.now() - td(days=3)
    stime = time.mktime(time.strptime(start, time_format))
    etime = time.mktime(
        time.strptime(f"{today.year}-{today.month}-{today.day}", time_format)
    )

    ptime = stime + prop * (etime - stime)

    return time.strftime(time_format, time.localtime(ptime))


if os.path.isfile("stations.csv"):
    print("file already exists")
    print("cleaning stations.csv")
    stations = gld.load_stations_csv()
    stations["label_vals"] = False
    stations["label_icon"] = False
    stations["label_condition"] = False
    rand_dates = [str_time_prop("2024-01-01", "%Y-%m-%d", random()) for i in range(15)]
    rand_dates.append(
        time.strftime(
            "%Y-%m-%d",
            time.localtime(time.mktime(time.strptime("2024-09-12", "%Y-%m-%d"))),
        )
    )
    for si, st in tqdm(enumerate(stations.iterrows()), total=stations.shape[0]):
        # if str(st[1]["ID"]) == "00006":
        #     continue
        # print(si, ", ", st[1]["ID"], ", ", rd)
        datas = gld.getWeatherData(st[1]["ID"], rand_dates)
        for data in datas:
            # print(data[0])
            if data[0] != "error":
                for di in range(len(data)):
                    # print(stations.loc[si, "label_icon"])
                    if data[di]["icon"] != None:
                        stations.loc[si, "label_icon"] = True
                    if data[di]["condition"] != None:
                        stations.loc[si, "label_condition"] = True
                    if (
                        data[di]["precipitation"] != None
                        or data[di]["pressure_msl"] != None
                        or data[di]["sunshine"] != None
                        or data[di]["temperature"] != None
                        or data[di]["wind_direction"] != None
                        or data[di]["wind_speed"] != None
                        or data[di]["cloud_cover"] != None
                        or data[di]["dew_point"] != None
                        or data[di]["relative_humidity"] != None
                        or data[di]["visibility"] != None
                        or data[di]["wind_gust_direction"] != None
                        or data[di]["wind_gust_speed"] != None
                        or data[di]["condition"] != None
                        or data[di]["precipitation_probability"] != None
                        or data[di]["precipitation_probability_6h"] != None
                        or data[di]["solar"] != None
                    ):
                        stations.loc[si, "label_vals"] = True
                # print(stations.iloc[si, :])
    stations.to_csv("stations.csv", sep=",", index=False, encoding="utf-8")


else:
    print("file doesn't exists")
    print("creating stations.csv")
    # print(stations("01766"))
    stations_list = []
    rand_dates = [str_time_prop("2024-01-01", "%Y-%m-%d", random()) for i in range(2)]
    rand_dates.append(
        time.strftime(
            "%Y-%m-%d",
            time.localtime(time.mktime(time.strptime("2025-01-04", "%Y-%m-%d"))),
        )
    )
    stations_lists = gld.getSourceData(rand_dates)
    for sl in stations_lists:
        # print(sl[0])
        if sl[0] != "error":
            stations_list.append(sl)
    # for i in tqdm(range(0, 99999)):
    #     stri = str(i)
    #     for ch in range(0, 5 - len(str(i))):
    #         stri = "0" + stri

    #     for rd in rand_dates:
    #         data = gdwd.getSourcesByStationIDDate(stri, rd)
    #         if data[0] != "error":
    #             stations_list.append(data)
    #             break
    # stations_df = pd.DataFrame(columns=["ID","Name","height","lat","lon","start","end"])
    stations_df = pd.DataFrame(
        stations_list,
        columns=["ID", "Name", "height", "lat", "lon", "start", "end"],
    )
    stations_df.to_csv("stations.csv", sep=",", index=False, encoding="utf-8")
    if os.path.isfile("stations.csv"):
        print("file already exists")
        print("cleaning stations.csv")
        stations = gld.load_stations_csv()
        stations["label_vals"] = False
        stations["label_icon"] = False
        stations["label_condition"] = False
        rand_dates = [
            str_time_prop("2024-01-01", "%Y-%m-%d", random()) for i in range(2)
        ]
        rand_dates.append(
            time.strftime(
                "%Y-%m-%d",
                time.localtime(time.mktime(time.strptime("2025-01-3", "%Y-%m-%d"))),
            )
        )
        for si, st in tqdm(enumerate(stations.iterrows()), total=stations.shape[0]):
            # print(si, ", ", st[1]["ID"], ", ", rd)
            datas = gld.getWeatherData(st[1]["ID"], rand_dates)
            for data in datas:
                # print(data[0])
                if data[0] != "error":
                    for di in range(len(data)):
                        if data[di]["icon"] != None:
                            stations.loc[si, "label_icon"] = True
                        if data[di]["condition"] != None:
                            stations.loc[si, "label_condition"] = True
                        if (
                            data[di]["precipitation"] != None
                            or data[di]["pressure_msl"] != None
                            or data[di]["sunshine"] != None
                            or data[di]["temperature"] != None
                            or data[di]["wind_direction"] != None
                            or data[di]["wind_speed"] != None
                            or data[di]["cloud_cover"] != None
                            or data[di]["dew_point"] != None
                            or data[di]["relative_humidity"] != None
                            or data[di]["visibility"] != None
                            or data[di]["wind_gust_direction"] != None
                            or data[di]["wind_gust_speed"] != None
                            or data[di]["condition"] != None
                            or data[di]["precipitation_probability"] != None
                            or data[di]["precipitation_probability_6h"] != None
                            or data[di]["solar"] != None
                        ):
                            stations.loc[si, "label_vals"] = True
                    # print(stations.iloc[si, :])
        stations.to_csv("stations.csv", sep=",", index=False, encoding="utf-8")
