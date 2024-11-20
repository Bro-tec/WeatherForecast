import CollectData.get_DWD_data as dwd  # otherwise lstm cant import this
import asyncio
from datetime import datetime as dt
from datetime import timedelta as td
import pandas as pd
import numpy as np  # euclidean: 34:03 each epoche
from progress.bar import Bar
import math
import random
from tqdm import tqdm
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

continous_data = {}

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
async def filter_dataHourly(data, feature_labels):
    data.drop(["source_id"], axis=1, inplace=True)
    for j in range(len(icons)):
        data["icon" + str(j)] = [
            1 if icons.index(data["icon"][i]) == j else 0 for i in range(len(data))
        ]
        data["condition" + str(j)] = [
            1 if icons.index(data["condition"][i]) == j else 0 for i in range(len(data))
        ]
    # data["icon"] = [float(icons.index(data["icon"][i])) for i in range(len(data))]
    # data["condition"] = [
    #     float(icons.index(data["condition"][i])) for i in range(len(data))
    # ]
    if "fallback_source_ids" in data:
        data.drop(["fallback_source_ids"], axis=1, inplace=True)
    # data["timestamp"] = [int(str(data.iloc[i, 0])[11:13]) for i in range(data.shape[0])]
    # data.drop(["timestamp"], axis=1, inplace=True)
    data = data[feature_labels]
    return data


# joining found data for hourly or returning error
async def joinDataHourly(cityID, cities, lc, date, feature_labels):
    data = await dwd.getWeatherByStationIDDate(cityID, date)
    if len(data) < 24:
        return pd.DataFrame(columns=["error"])
    data = await filter_dataHourly(pd.DataFrame(data), feature_labels)
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
            newdata = await filter_dataHourly(newdata, feature_labels)
            data = pd.concat([data, pd.DataFrame(newdata)], ignore_index=True, axis=1)
    data.fillna(0, inplace=True)
    return data.round(3)


# gathered async function to get hourly inputs and labels
async def get_DataHourlyAsync(
    row,
    cityP,
    date,
    distance="near",
    feature_labels=[],
    month=False,
    hours=False,
    position=False,
):
    cities = await getCities(cityP, row.ID, distance)
    train_Data = await joinDataHourly(row.ID, cities, 4, date.date(), feature_labels)
    if "error" in train_Data or len(train_Data) < 25:
        hourly_bar.next()
        return pd.DataFrame(columns=["error"]), row.ID
    if hours:
        train_Data["hours"] = [i for i in range(len(train_Data))]
    if month:
        train_Data["month"] = [int(date.month) for i in range(len(train_Data))]
    if position:
        train_Data["lon"] = [row.lon for i in range(len(train_Data))]
        train_Data["lat"] = [row.lat for i in range(len(train_Data))]
        train_Data["height"] = [row.height for i in range(len(train_Data))]
    if len(train_Data) > 24:
        train_Data = train_Data[:24]
    hourly_bar.next()
    return (train_Data.to_numpy(), row.ID)


# main async function to get hourly inputs and labels, to use parallelized dataretrival which makes the code faster
async def DataHourlyAsync(
    cityloop,
    cityP,
    date,
    r_mode=False,
    minTime=0,
    duration=0,
    continous_hour_range=24,
    label_hour_range=12,
    feature_labels=[
        "temperature",
        "wind_direction",
        "wind_speed",
        "visibility",
        "icon0",
        "icon1",
        "icon2",
        "icon3",
        "icon4",
        "icon5",
        "icon6",
        "icon7",
        "icon8",
        "icon9",
        "icon10",
        "icon11",
        "icon12",
        "icon13",
        "icon14",
        "icon15",
        "icon16",
        "icon17",
        "icon18",
        "icon19",
        "icon20",
        "condition0",
        "condition1",
        "condition2",
        "condition3",
        "condition4",
        "condition5",
        "condition6",
        "condition7",
        "condition8",
        "condition9",
        "condition10",
        "condition11",
        "condition12",
        "condition13",
        "condition14",
        "condition15",
        "condition16",
        "condition17",
        "condition18",
        "condition19",
        "condition20",
    ],
    month=False,
    hours=False,
    position=False,
):
    global continous_data
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
                    feature_labels=feature_labels,
                    month=month,
                    hours=hours,
                    position=position,
                )
                for row in cityloop.itertuples(index=False)
            ]
        )
    else:
        lists = await asyncio.gather(
            *[
                get_DataHourlyAsync(
                    row,
                    cityP,
                    date,
                    feature_labels=feature_labels,
                    month=month,
                    hours=hours,
                    position=position,
                )
                for row in cityloop.itertuples(index=False)
            ]
        )

    for l in tqdm(lists, total=len(lists)):
        # print("l[0]: ", l[0].shape)
        if len(l) > 0:
            if len(l[0]) > 0:  # and len(l[2]):
                # print(str(l[1]) in continous_data)
                if not str(l[1]) in continous_data:
                    continous_data[str(l[1])] = np.array(l[0])
                else:
                    continous_data[str(l[1])] = np.concatenate(
                        [continous_data[str(l[1])], np.array(l[0])], axis=0
                    )
                    if continous_data[str(l[1])].shape[0] > continous_hour_range:
                        continous_data[str(l[1])] = continous_data[str(l[1])][
                            continous_data[str(l[1])].shape[0] - continous_hour_range :
                        ]
    train_list = []
    label_list = []

    # Iterate through the keys of the continuous data dictionary
    for k in tqdm(continous_data.keys(), total=len(continous_data.keys())):
        if continous_data[k].shape[0] == continous_hour_range:
            for j in range(continous_hour_range - label_hour_range - 1):
                train_list.append(continous_data[k][j : j + label_hour_range])
                label_list.append(
                    continous_data[k][j + label_hour_range + 1][
                        [i for i in range(len(feature_labels))]
                    ]
                )

    # Convert lists to numpy arrays in a single step
    train_np = np.array(train_list)
    label_np = np.array(label_list)
    return train_np, label_np


# main function for hourly dataretrival to switch between async and non async
def gen_trainDataHourly_Async(
    skip_days=0, redos=1, seq=12, max_batch=24, month=False, hours=False, position=False
):
    cityP = load_stations_csv()
    cityP = cityP  # [:10]  # for testing only
    minTime = dt.strptime(min(cityP["start"]), "%Y-%m-%d %H:%M:%S")
    minTime = minTime + td(days=skip_days)
    actualTime = dt.now()
    duration = actualTime - minTime
    print(duration, "=", actualTime, "-", minTime)
    for d in range(duration.days - 3):
        date = minTime + td(days=d)
        s = math.floor(len(cityP) / redos)
        for i in range(redos):
            x, y = [], []
            if i == redos - 1:
                x, y = asyncio.run(
                    DataHourlyAsync(
                        cityP[s * i :],
                        cityP,
                        date,
                        continous_hour_range=max_batch,
                        label_hour_range=seq,
                        month=month,
                        hours=hours,
                        position=position,
                    )
                )
                # x,y,z = asyncio.run(DataHourlyAsync(cityP[s*i:], cityP, date, r_mode=True, minTime=minTime, duration=duration.days))
            else:
                x, y = asyncio.run(
                    DataHourlyAsync(
                        cityP[s * i : s * (i + 1)],
                        cityP,
                        date,
                        continous_hour_range=max_batch,
                        month=month,
                        hours=hours,
                        position=position,
                    )
                )
                # x,y,z = asyncio.run(DataHourlyAsync(cityP[s*i:s*(i+1)], cityP, date, r_mode=True, minTime=minTime, duration=duration.days))
            # print(x.shape)
            # print(y.shape)
            # print(z.shape)
            # if x == "error":
            #     print("was error so code continued")
            #     continue
            yield x, y, (d + skip_days), (i + 1)


# hourly data retreval for predictions
def get_predictDataHourly(date, city="", id=""):
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
        return asyncio.run(DataHourlyAsync(cityP, cities, date))


async def getWeatherByStationIDDate(stid, dates):
    list = await asyncio.gather(
        *[dwd.getWeatherByStationIDDate(stid, date) for date in dates]
    )
    return list


def getWeatherData(stid, dates):
    return asyncio.run(getWeatherByStationIDDate(stid, dates))
