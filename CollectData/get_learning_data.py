from CollectData import get_DWD_data as dwd  # otherwise lstm cant import this
import asyncio
from datetime import datetime as dt
from datetime import timedelta as td
import pandas as pd
import numpy as np  # euclidean: 34:03 each epoche
from progress.bar import Bar
import math
import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# list of label values
icons = [
    None,
    "clear-day",
    "clear-night",
    "partly-cloudy-day",
    "partly-cloudy-night",
    "cloudy",
    "fog",
    "wind",
    "rain",
    "sleet",
    "snow",
    "hail",
    "thunderstorm",
    "dry",
    "moist",
    "wet",
    "rime",
    "ice",
    "glaze",
    "not dry",
    "reserved",
]
# setted global bar to show status without using tqdm becaus of async
hourly_bar = Bar("Processing", max=20)
daily_bar = Bar("Processing", max=20)


# loading data from excel file
def load_stations_csv():
    return pd.read_csv(
        "stations.csv", dtype={"ID": object}
    )  # otherwise lstm cant load this


# calculating euclidean distance using pandas
async def euclidean(chosen, unchosen):
    cols = ["lon", "lat", "height"]
    unchosen["dist"] = np.linalg.norm(
        chosen[cols].values - unchosen[cols].values, axis=1
    )
    return unchosen


# choosing closest city to long and lat. Can get manipulated to get bugger distance to cities
async def chooseByNearest(chosen, unchosen, addLon=0, addLat=0):
    chosen.iloc[0, 4] += addLon
    chosen.iloc[0, 3] += addLat
    dists = await euclidean(chosen.head(1), unchosen)
    dists.sort_values(by=["dist"], ascending=True, inplace=True)
    return dists["ID"]


# listing cities starting with closest cities to point in single to multiple column counts
async def getCities(cityP, cityID, distance):
    chosen = cityP[cityP["ID"] == cityID]
    unchosen = cityP.drop(cityP[cityP["ID"] == cityID].index)
    if len(chosen) < 1:
        return "Error"
    if len(unchosen) < 1:
        return "Error"
    if distance.lower() == "near":
        return pd.DataFrame(await chooseByNearest(chosen, unchosen))
    elif distance.lower() == "close" or distance.lower() == "far":
        if distance.lower() == "close":
            return pd.concat(
                [
                    await chooseByNearest(chosen, unchosen),
                    await chooseByNearest(chosen, unchosen, addLat=0.5),
                    await chooseByNearest(chosen, unchosen, addLon=0.5),
                    await chooseByNearest(chosen, unchosen, addLat=-0.5),
                    await chooseByNearest(chosen, unchosen, addLon=-0.5),
                ],
                axis=1,
            )
        return pd.concat(
            [
                await chooseByNearest(chosen, unchosen),
                await chooseByNearest(chosen, unchosen, addLat=0.5),
                await chooseByNearest(chosen, unchosen, addLon=0.5),
                await chooseByNearest(chosen, unchosen, addLat=-0.5),
                await chooseByNearest(chosen, unchosen, addLon=-0.5),
                await chooseByNearest(chosen, unchosen, addLat=2.5),
                await chooseByNearest(chosen, unchosen, addLon=2.5),
                await chooseByNearest(chosen, unchosen, addLat=-2.5),
                await chooseByNearest(chosen, unchosen, addLon=-2.5),
            ],
            axis=1,
        )
    return pd.DataFrame(columns=["error"])


# filter special for hourly named dataloader
async def filter_dataHourly(data):
    data.drop(["source_id"], axis=1, inplace=True)
    data["icon"] = [float(icons.index(data["icon"][i])) for i in range(len(data))]
    data["condition"] = [
        float(icons.index(data["condition"][i])) for i in range(len(data))
    ]
    if "fallback_source_ids" in data:
        data.drop(["fallback_source_ids"], axis=1, inplace=True)
    data.drop(["timestamp"], axis=1, inplace=True)
    return data


# filter special for daily named dataloader, uses filter_dataHourly because the needs overlap
async def filter_dataComplete(data):
    data = await filter_dataHourly(data)
    data.fillna(0, inplace=True)
    return data


# joining found data for hourly or returning error
async def joinDataHourly(cityID, cities, lc, date):
    cities.sort_index(inplace=True)
    data = await dwd.getWeatherByStationIDDate(cityID, date)
    if len(data) < 24:
        return pd.DataFrame(columns=["error"])
    data = await filter_dataHourly(pd.DataFrame(data))
    for j in range(cities.shape[1]):
        i = 0
        lcount = lc
        while i < lcount:
            newdata = await dwd.getWeatherByStationIDDate(
                cities.iloc[i, j], date, wait=False
            )
            i += 1
            if len(newdata) < 24:
                if lcount >= lc * 7:
                    print("error occured")
                    return pd.DataFrame(columns=["error"])
                lcount += 1
                continue
            newdata = pd.DataFrame(newdata)
            newdata = await filter_dataHourly(newdata)
            data = pd.concat([data, pd.DataFrame(newdata)], ignore_index=True, axis=1)
    data.fillna(0, inplace=True)
    return data.round(3)


# joining found data for daily or returning error
async def joinDataComplete(cityID, cities, lc, date):
    data = await dwd.getWeatherByStationIDDate(cityID, date, delay=5)
    if len(data) <= 24:
        xer1 = dt.now()
        return pd.DataFrame(columns=["error"])
    data = pd.DataFrame(data)
    data = data[:24]
    for j in range(cities.shape[1]):
        i = 0
        lcount = lc
        while i - lc < lcount:
            newdata = await dwd.getWeatherByStationIDDate(
                cities.iloc[i, j], date, wait=False
            )
            i += 1
            if len(newdata) <= 24:
                if i - lc >= 12:
                    return pd.DataFrame(columns=["error"])
                lcount += 1
                continue
            newdata = newdata[:24]
            data = pd.concat([data, pd.DataFrame(newdata)], ignore_index=True)
    return await filter_dataComplete(data)


# returns label for daily
async def labelMinMax(cityID, date):
    data = await dwd.getWeatherByStationIDDate(cityID, date)
    if data[0] == "error":
        return pd.DataFrame(columns=["error"])
    label_Data = pd.DataFrame(data)
    label_Data = await filter_dataComplete(label_Data)
    conditions = label_Data["condition"].value_counts().index[0]
    icons = label_Data["icon"].value_counts().index[0]
    return pd.DataFrame(
        {
            "temperature_min": [min(label_Data["temperature"])],
            "temperature_max": max(label_Data["temperature"]),
            "wind_speed_min": min(label_Data["wind_speed"]),
            "wind_speed_max": max(label_Data["wind_speed"]),
            "condition": conditions,
            "icon": icons,
        }
    )


# returns label for hourly
async def label24(cityID, date):
    data = await dwd.getWeatherByStationIDDate(cityID, date)
    if data[0] == "error":
        return pd.DataFrame(columns=["error"])
    label_Data = pd.DataFrame(data)
    label_Data = await filter_dataComplete(label_Data)
    return label_Data[
        [
            "temperature",
            "wind_direction",
            "wind_speed",
            "visibility",
            "condition",
            "icon",
        ]
    ]


# gathered async function to get hourly inputs and labels
async def get_DataHourlyAsync(row, cityP, date, get_mode="normal"):
    cityID = row.ID
    cities = await getCities(cityP, cityID, "near")
    train_Data = await joinDataHourly(cityID, cities, 4, date.date())
    if "error" in train_Data or len(train_Data) < 25:
        hourly_bar.next()
        return pd.DataFrame(columns=["error"])
    train_Data["t"] = [i for i in range(len(train_Data))]
    if get_mode != "normal":
        return train_Data[:-1].to_numpy()
    if len(train_Data) > 25:
        train_Data = train_Data[:25]
    date = date + td(days=1)
    label_Data24 = await label24(cityID, date)
    if "error" in label_Data24 or len(label_Data24) < 24:
        hourly_bar.next()
        return pd.DataFrame(columns=["error"])
    if len(label_Data24) > 24:
        label_Data24 = label_Data24[:24]
    label_Data = train_Data[[3, 4, 5, 9, 12, 16]]
    hourly_bar.next()
    return (
        train_Data[:-1].to_numpy(),
        label_Data.iloc[1:].to_numpy(),
        label_Data24.to_numpy(),
    )


# gathered async function to get daily inputs and labels
async def get_DataDailyAsync(row, cityP, date, get_mode="normal"):
    cityID = row.ID
    cities = await getCities(cityP, cityID, "far")
    train_Data = await joinDataComplete(cityID, cities, 1, date.date())
    if "error" in train_Data:
        daily_bar.next()
        return pd.DataFrame(columns=["error"])
    if get_mode != "normal":
        return train_Data[:-1].to_numpy()
    date = date + td(days=1)
    label_Data_Daily1 = await labelMinMax(cityID, date)
    date = date + td(days=1)
    label_Data_Daily2 = await labelMinMax(cityID, date)
    date = date + td(days=1)
    label_Data_Daily3 = await labelMinMax(cityID, date)
    date = date + td(days=1)
    label_Data_Daily4 = await labelMinMax(cityID, date)
    date = date + td(days=1)
    label_Data_Daily5 = await labelMinMax(cityID, date)
    date = date + td(days=1)
    label_Data_Daily6 = await labelMinMax(cityID, date)
    date = date + td(days=1)
    label_Data_Daily7 = await labelMinMax(cityID, date)
    daily_bar.next()
    return [
        train_Data,
        label_Data_Daily1,
        label_Data_Daily2,
        label_Data_Daily3,
        label_Data_Daily4,
        label_Data_Daily5,
        label_Data_Daily6,
        label_Data_Daily7,
    ]


# main async function to get hourly inputs and labels, to use parallelized dataretrival which makes the code faster
async def DataHourlyAsync(
    cityloop, cityP, date, get_mode="normal", r_mode=False, minTime=0, duration=0
):
    global hourly_bar
    hourly_bar = Bar("Processing", max=len(cityloop))
    lists = []
    if r_mode:
        lists = await asyncio.gather(
            *[
                get_DataHourlyAsync(
                    row,
                    cityP,
                    minTime + td(days=random.randint(0, duration - 3)),
                    get_mode=get_mode,
                )
                for row in cityloop.itertuples(index=False)
            ]
        )
    else:
        lists = await asyncio.gather(
            *[
                get_DataHourlyAsync(row, cityP, date, get_mode=get_mode)
                for row in cityloop.itertuples(index=False)
            ]
        )
    if get_mode == "normal":
        train_np = np.zeros(shape=(1, 1))
        label_np = np.zeros(shape=(1, 1))
        label24_np = np.zeros(shape=(1, 1))
        i = 0
        for l in lists:
            if len(l) > 0:
                if len(l[0]) > 0 and len(l[1]) > 0 and len(l[2]):
                    if i == 0:
                        train_np = np.array([l[0]])
                        label_np = np.array([l[1]])
                        label24_np = np.array([l[2]])
                        i += 1
                    else:
                        train_np = np.concatenate([train_np, np.array([l[0]])], axis=1)
                        label_np = np.concatenate([label_np, np.array([l[1]])], axis=1)
                        label24_np = np.concatenate(
                            [label24_np, np.array([l[2]])], axis=1
                        )
        if train_np.shape[1] == 1:
            return "error", "error", "error"
        return (
            train_np.reshape(-1, train_np.shape[2]),
            label_np.reshape(-1, label_np.shape[2]),
            label24_np.reshape(-1, label24_np.shape[2]),
        )
    else:
        train_np = np.zeros(shape=(1, 1))
        i = 0
        for l in lists:
            if len(l) > 0:
                if i == 0:
                    train_np = np.array([l[0]])
                    i += 1
                else:
                    train_np = np.concatenate([train_np, np.array([l[0]])], axis=1)
        if train_np.shape[1] == 1:
            return "error", "error", "error"
        return train_np


# main async function to get daily inputs and labels, to use parallelized dataretrival which makes the code faster
async def DataDailyAsync(
    cityloop, cityP, date, get_mode="normal", r_mode=False, minTime=0, duration=0
):
    global daily_bar
    daily_bar = Bar("Processing", max=len(cityloop))
    lists = []
    if r_mode:
        lists = await asyncio.gather(
            *[
                get_DataDailyAsync(
                    row,
                    cityP,
                    minTime + td(days=random.randint(0, duration - 3)),
                    get_mode=get_mode,
                )
                for row in cityloop.itertuples(index=False)
            ]
        )
    else:
        lists = await asyncio.gather(
            *[
                get_DataDailyAsync(row, cityP, date, get_mode=get_mode)
                for row in cityloop.itertuples(index=False)
            ]
        )
    if get_mode == "normal":
        train_np = np.zeros(shape=(1, 456, 17))
        label_np1 = np.zeros(shape=(1, 6))
        label_np2 = np.zeros(shape=(1, 6))
        label_np3 = np.zeros(shape=(1, 6))
        label_np4 = np.zeros(shape=(1, 6))
        label_np5 = np.zeros(shape=(1, 6))
        label_np6 = np.zeros(shape=(1, 6))
        label_np7 = np.zeros(shape=(1, 6))
        i = 0
        for l in lists:
            if len(l) > 0:
                if (
                    len(l[0]) > 0
                    and len(l[1]) > 0
                    and len(l[2]) > 0
                    and len(l[3]) > 0
                    and len(l[4]) > 0
                    and len(l[5]) > 0
                    and len(l[6]) > 0
                    and len(l[7]) > 0
                ):
                    if i == 0:
                        train_np = np.array([l[0]])
                        label_np1 = np.array([l[1]])
                        label_np2 = np.array([l[2]])
                        label_np3 = np.array([l[3]])
                        label_np4 = np.array([l[4]])
                        label_np5 = np.array([l[5]])
                        label_np6 = np.array([l[6]])
                        label_np7 = np.array([l[7]])
                        i += 1
                    else:
                        train_np = np.concatenate([train_np, np.array([l[0]])], axis=0)
                        label_np1 = np.concatenate(
                            [label_np1, np.array([l[1]])], axis=0
                        )
                        label_np2 = np.concatenate(
                            [label_np2, np.array([l[2]])], axis=0
                        )
                        label_np3 = np.concatenate(
                            [label_np3, np.array([l[3]])], axis=0
                        )
                        label_np4 = np.concatenate(
                            [label_np4, np.array([l[4]])], axis=0
                        )
                        label_np5 = np.concatenate(
                            [label_np5, np.array([l[5]])], axis=0
                        )
                        label_np6 = np.concatenate(
                            [label_np6, np.array([l[6]])], axis=0
                        )
                        label_np7 = np.concatenate(
                            [label_np7, np.array([l[7]])], axis=0
                        )
        if train_np.shape[1] == 1:
            return "error", "error", "error"
        return (
            train_np.reshape(train_np.shape[0], -1),
            label_np1,
            label_np2,
            label_np3,
            label_np4,
            label_np5,
            label_np6,
            label_np7,
        )
    else:
        train_np = np.zeros(shape=(1, 456, 17))
        i = 0
        for l in lists:
            if len(l) > 0:
                if i == 0:
                    train_np = np.array([l[0]])
                    i += 1
                else:
                    train_np = np.concatenate([train_np, np.array([l[0]])], axis=0)
        if train_np.shape[1] == 1:
            return "error", "error", "error"
        return train_np.reshape(train_np.shape[0], -1)


# main function for daily dataretrival to switch between async and non async
def gen_trainDataDaily_Async(skip_days=0, redos=1):
    cityP = load_stations_csv()
    minTime = dt.strptime(min(cityP["start"]), "%Y-%m-%d %H:%M:%S")
    minTime = minTime + td(days=skip_days)
    actualTime = dt.now()
    duration = actualTime - minTime
    print(duration, "=", actualTime, "-", minTime)
    for d in range(duration.days - 1):
        date = minTime + td(days=d)
        s = math.floor(len(cityP) / redos)
        for i in range(redos):
            if i == redos - 1:
                # x,y1,y2,y3,y4,y5,y6,y7 = asyncio.run(DataDailyAsync(cityP[s*i:], cityP, date, r_mode=True, minTime=minTime, duration=duration.days))
                x, y1, y2, y3, y4, y5, y6, y7 = asyncio.run(
                    DataDailyAsync(cityP[s * i :], cityP, date)
                )
            else:
                # x,y1,y2,y3,y4,y5,y6,y7 = asyncio.run(DataDailyAsync(cityP[s*i:s*(i+1)], cityP, date, r_mode=True, minTime=minTime, duration=duration.days))
                x, y1, y2, y3, y4, y5, y6, y7 = asyncio.run(
                    DataDailyAsync(cityP[s * i : s * (i + 1)], cityP, date)
                )
            if x == "error":
                print("was error so code continued")
                continue
            yield x, y1, y2, y3, y4, y5, y6, y7, (d + skip_days)


# main function for hourly dataretrival to switch between async and non async
def gen_trainDataHourly_Async(skip_days=0, redos=1):
    cityP = load_stations_csv()
    minTime = dt.strptime(min(cityP["start"]), "%Y-%m-%d %H:%M:%S")
    minTime = minTime + td(days=skip_days)
    actualTime = dt.now()
    duration = actualTime - minTime
    print(duration, "=", actualTime, "-", minTime)
    for d in range(duration.days - 3):
        date = minTime + td(days=d)
        s = math.floor(len(cityP) / redos)
        for i in range(redos):
            if i == redos - 1:
                x, y, z = asyncio.run(DataHourlyAsync(cityP[s * i :], cityP, date))
                # x,y,z = asyncio.run(DataHourlyAsync(cityP[s*i:], cityP, date, r_mode=True, minTime=minTime, duration=duration.days))
            else:
                x, y, z = asyncio.run(
                    DataHourlyAsync(cityP[s * i : s * (i + 1)], cityP, date)
                )
                # x,y,z = asyncio.run(DataHourlyAsync(cityP[s*i:s*(i+1)], cityP, date, r_mode=True, minTime=minTime, duration=duration.days))
            if x == "error":
                print("was error so code continued")
                continue
            yield x, y, z, (d + skip_days)


# hourly data retreval for predictions
def get_predictDataHourly(date, back=0, city="", id=""):
    cityP = load_stations_csv()
    cities = load_stations_csv()
    if city != "":
        cityP = cityP[cityP["Name"] == city]
    elif id != "":
        cityP = cityP[cityP["ID"] == id]
    else:
        return

    if date <= dt.now() - td(days=3):
        return asyncio.run(DataHourlyAsync(cityP, cities, date))
    else:
        return asyncio.run(DataHourlyAsync(cityP, cities, date, get_mode="input"))


# daily data retreval for predictions
def get_predictDataDaily(date, city="", id=""):
    cityP = load_stations_csv()
    cities = load_stations_csv()
    if city != "":
        cityP = cityP[cityP["Name"] == city]
    elif id != "":
        cityP = cityP[cityP["ID"] == id]
    else:
        return

    if date <= dt.now() - td(days=3):
        return asyncio.run(DataDailyAsync(cityP, cities, date))
    else:
        return asyncio.run(DataDailyAsync(cityP, cities, date, get_mode="input"))
