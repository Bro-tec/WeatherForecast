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
import torch
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
status_bar = Bar("Processing", max=20)


# loading data from excel file
def load_stations_csv():
    df = pd.read_csv(
        "stations.csv", dtype={"ID": object}
    )  # otherwise lstm cant load this
    return df[df["label_vals"] == True]


def load_stations_by_IDs(ids):
    df = pd.read_csv(
        "stations.csv", dtype={"ID": object}
    )  # otherwise lstm cant load this
    return df[df["ID"].isin(ids)]


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
async def getCities(cityP, cityID, distance="near"):
    chosen = cityP[cityP["ID"] == cityID]
    unchosen = cityP.drop(cityP[cityP["ID"] == cityID].index)
    if len(chosen) < 1:
        return "Error"
    if len(unchosen) < 1:
        return "Error"
    if distance.lower() == "near":
        return pd.DataFrame(await chooseByNearest(chosen, unchosen))
    # the code in elif wont get used anymore
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
    if "source_id" in data:
        data.drop(["source_id"], axis=1, inplace=True)
    else:
        print("Error:\n", data)
        return data
    for j in range(len(icons)):
        data["icon" + str(j)] = [
            1 if icons.index(data["icon"][i]) == j else 0 for i in range(len(data))
        ]
        data["condition" + str(j)] = [
            1 if icons.index(data["condition"][i]) == j else 0 for i in range(len(data))
        ]
    # print("wind_direction: ", data["wind_direction"].to_list())
    data["wind_direction"] -= 22
    # print("wind_direction changed: ", data["wind_direction"].to_list())
    data["wind_direction"] = [
        390 if math.isnan(data["wind_direction"][i]) else data["wind_direction"][i]
        for i in range(len(data))
    ]
    # print("wind_direction new val: ", data["wind_direction"].to_list())
    data["wind_direction"] = [
        math.floor(data["wind_direction"][i] / 45)
        if data["wind_direction"][i] >= 0
        else math.floor(data["wind_direction"][i] * -1 / 45)
        for i in range(len(data))
    ]
    for j in range(9):
        data["wind_direction" + str(j)] = [
            1 if data["wind_direction"][i] == j else 0 for i in range(len(data))
        ]

    data["wind_gust_direction"] -= 22
    data["wind_gust_direction"] = [
        390
        if math.isnan(data["wind_gust_direction"][i])
        else data["wind_gust_direction"][i]
        for i in range(len(data))
    ]
    data["wind_gust_direction"] = [
        math.floor(data["wind_gust_direction"][i] / 45)
        if data["wind_gust_direction"][i] >= 0
        else math.floor(data["wind_gust_direction"][i] * -1 / 45)
        for i in range(len(data))
    ]
    for j in range(9):
        data["wind_gust_direction" + str(j)] = [
            1 if data["wind_gust_direction"][i] == j else 0 for i in range(len(data))
        ]
    if "fallback_source_ids" in data:
        data.drop(["fallback_source_ids"], axis=1, inplace=True)
    return data[feature_labels]


# joining found data for hourly or returning error
async def joinDataHourly(cityID, cities, lc, date, feature_labels, ignore_len=False):
    newCities = []
    data = await dwd.getWeatherByStationIDDate(cityID, date)
    if len(data) < 24 and not ignore_len:
        return pd.DataFrame(columns=["error"]), []
    if data[0] == "error":
        return pd.DataFrame(columns=["error"]), []
    data = await filter_dataHourly(pd.DataFrame(data), feature_labels)
    # print("filter_dataHourly: ", data)
    for j in range(cities.shape[1]):
        i = 0
        lcount = lc
        while i < lcount:
            newdata = await dwd.getWeatherByStationIDDate(
                cities.iloc[i, j], date, wait=False
            )
            i += 1
            if len(newdata) < len(data):
                if lcount >= lc * 7:
                    print("error occured")
                    return pd.DataFrame(columns=["error"]), []
                lcount += 1
                continue
            newdata = pd.DataFrame(newdata)
            newCities.append(cities.iloc[i, j])
            newdata = await filter_dataHourly(newdata, feature_labels)
            data = pd.concat([data, pd.DataFrame(newdata)], ignore_index=True, axis=1)
    data.fillna(0, inplace=True)
    return (data.round(3), newCities)


# gathered async function to get hourly inputs and labels
async def get_DataHourlyAsync(
    row,
    cityP,
    date,
    distance="near",
    feature_labels=[],
    next_city_amount=4,
    month=False,
    hours=False,
    position=False,
    ignore_len=False,
):
    cities = await getCities(cityP, row.ID, distance)
    train_Data, newCities = await joinDataHourly(
        row.ID,
        cities,
        next_city_amount,
        date.date(),
        feature_labels,
        # ignore_len=ignore_len,
    )
    if "error" in train_Data or len(train_Data) < 25:
        hourly_bar.next()
        print("bad error occured: ", row.ID)
        # print(row)
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
    # print("get_DataHourlyAsync: ", train_Data.to_numpy().shape, row.ID, newCities)
    return (train_Data.to_numpy(), row.ID, newCities)


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
    feature_labels=[],
    next_city_amount=4,
    month=False,
    hours=False,
    position=False,
    ignore_len=False,
    label=True,
):
    global continous_data
    # print("unchanged continous_data: ", len(continous_data.keys()))
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
                    next_city_amount=next_city_amount,
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
                    next_city_amount=next_city_amount,
                    month=month,
                    hours=hours,
                    position=position,
                    # ignore_len=ignore_len,
                )
                for row in cityloop.itertuples(index=False)
            ]
        )

    # print(continous_hour_range, label_hour_range)
    # id_list = cityloop.ID.to_list()
    id_list = []
    # print("idList", id_list)
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
                id_list += l[2]

    id_list = list(set(id_list))
    ids = []
    train_list = []
    label_list = []

    # print("continous_data: ", len(continous_data.keys()))

    print(
        "continous_hour_range: ",
        continous_hour_range,
        "label_hour_range: ",
        label_hour_range,
    )
    # Iterate through the keys of the continuous data dictionary
    for k in tqdm(continous_data.keys(), total=len(continous_data.keys())):
        if continous_data[k].shape[0] >= continous_hour_range:
            ids.append(k)
            extra = 1
            if label:
                extra = -1
            print("j loop: ", continous_data[k].shape[0] - label_hour_range + extra)
            print("continous_data[k]:", continous_data[k].shape[0])
            for j in range(continous_data[k].shape[0] - label_hour_range + extra):
                train_list.append(continous_data[k][j : j + label_hour_range])
                if label:
                    # print(continous_data[k].shape, j + label_hour_range)
                    label_list.append(
                        continous_data[k][j + label_hour_range][
                            [i for i in range(len(feature_labels))]
                        ]
                    )

    # Convert lists to numpy arrays in a single step
    train_np = np.array(train_list)
    label_np = np.array(label_list)
    # print("trs_List: ", len(train_list), len(label_list))
    print("trs: ", train_np.shape, label_np.shape)
    return id_list, train_np, label_np, ids


# main function for hourly dataretrival to switch between async and non async
def gen_trainDataHourly_Async(
    skip_days=0,
    redos=1,
    seq=12,
    max_batch=24,
    next_city_amount=4,
    feature_labels=[
        "temperature",
        "wind_speed",
        "visibility",
        "wind_direction0",
        "wind_direction1",
        "wind_direction2",
        "wind_direction3",
        "wind_direction4",
        "wind_direction5",
        "wind_direction6",
        "wind_direction7",
        "wind_direction8",
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
                _, x, y, _ = asyncio.run(
                    DataHourlyAsync(
                        cityP[s * i :],
                        cityP,
                        date,
                        continous_hour_range=max_batch,
                        label_hour_range=seq,
                        next_city_amount=next_city_amount,
                        month=month,
                        hours=hours,
                        position=position,
                        feature_labels=feature_labels,
                    )
                )
                # x,y,z = asyncio.run(DataHourlyAsync(cityP[s*i:], cityP, date, r_mode=True, minTime=minTime, duration=duration.days))
            else:
                _, x, y, _ = asyncio.run(
                    DataHourlyAsync(
                        cityP[s * i : s * (i + 1)],
                        cityP,
                        date,
                        continous_hour_range=max_batch,
                        next_city_amount=next_city_amount,
                        month=month,
                        hours=hours,
                        position=position,
                        feature_labels=feature_labels,
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
def get_predictDataHourly(
    date,
    city=[],
    id=[],
    seq=12,
    forecast=24,
    next_city_amount=4,
    feature_labels=[],
    month=True,
    hours=True,
    position=True,
):
    cityP = load_stations_csv()
    cities = load_stations_csv()
    if len(city) > 0:
        cityP = cityP[cityP["Name"].isin(city)]
        id = cityP["ID"].to_list()
    elif len(id) > 0:
        cityP = cityP[cityP["ID"].isin(id)]
        id = cityP["ID"].to_list()
    else:
        id = cityP["ID"].to_list()
        print("continuing with all cities")

    if date <= dt.now() - td(days=1):
        return asyncio.run(DataHourlyAsync(cityP, cities, date))
    else:
        x, y, z = id, [], []
        x2 = []
        for i in range(forecast):
            print("Iterations: ", forecast, "/", (i + 1))
            print(list(set(x) - set(x2)))
            for day in range(math.floor(seq / 24)):
                asyncio.run(
                    DataHourlyAsync(
                        cities[cities["ID"].isin(list(set(x) - set(x2)))],
                        cities,
                        date - td(days=math.floor(seq / 24) - day + 1),
                        continous_hour_range=seq + 1,
                        label_hour_range=seq,
                        feature_labels=feature_labels,
                        month=month,
                        hours=hours,
                        position=position,
                        next_city_amount=next_city_amount,
                        ignore_len=True,
                        label=False,
                    )
                )
            x3, y, z, x1 = asyncio.run(
                DataHourlyAsync(
                    cities[cities["ID"].isin(list(set(x) - set(x2)))],
                    cities,
                    date - td(days=1),
                    continous_hour_range=seq + 1,
                    label_hour_range=seq,
                    feature_labels=feature_labels,
                    month=month,
                    hours=hours,
                    position=position,
                    next_city_amount=next_city_amount,
                    ignore_len=True,
                    label=False,
                )
            )
            x2 = x
            x = list(set(x + x3))
            # print("x2: ", x2)
            # print("y: ", y.shape)
            # print("z: ", z.shape)
        return x1, y, z


async def getWeatherByStationIDDate(stid, dates):
    list = await asyncio.gather(
        *[dwd.getWeatherByStationIDDate(stid, date) for date in dates]
    )
    return list


def get_indices(lst, targets):
    indices = []
    for index, element in enumerate(lst):
        if element in targets:
            indices.append(index)
    return indices


def continue_prediction(
    model,
    output_list,
    id_list,
    time_list,
    vals,
    h,
    date,
    device,
    prediction,
    next_city_amount=4,
    show_all=True,
    ids=[],
):
    global continous_data
    extended_id_list = id_list.copy()
    for hr in range(1, h + 1):
        for i, id in enumerate(id_list):
            if hr >= 24:
                time_list.append(time_list[i] + (hr % 24))
            else:
                time_list.append(time_list[i] + hr)
            month = int(math.floor(int(date.month) / 24))

            close = asyncio.run(getCities(load_stations_by_IDs(id_list), id))
            close_list = close["ID"].to_list()
            # print(id, close["ID"], "\n", close_list[0], close_list[1])
            # print("vals:", len(vals), ", ", len(id_list), ", ", len(extended_id_list))
            other_seq = output_list[
                (hr - 1) * len(id_list) + id_list.index(close_list[0])
            ]
            for nca in range(1, next_city_amount):
                other_seq = torch.cat(
                    [
                        output_list[
                            (hr - 1) * len(id_list) + id_list.index(close_list[i])
                        ],
                    ],
                    dim=0,
                )
            new_seq = torch.cat(
                [
                    output_list[(hr - 1) * len(id_list) + i],
                    other_seq,
                    torch.Tensor([time_list[(hr - 1) * len(id_list) + i] + 1]).to(
                        device
                    ),
                    torch.Tensor([month]).to(device),
                    torch.Tensor(vals[i]).to(device),
                ],
                dim=0,
            )
            # print(continous_data[id].shape, np.array([new_seq.tolist()]).shape)
            print(new_seq.shape)
            continous_data[id] = np.concatenate(
                [continous_data[id], np.array([new_seq.tolist()])], axis=0
            )
            continous_data[id] = continous_data[id][1:]
            # print(new_seq.shape)
            # print(new_seq[-5], new_seq[-4], new_seq[-3], new_seq[-2], new_seq[-1])
            output = prediction(model, continous_data[id], [], device)
            extended_id_list.append(id)
            output_list = torch.cat([output_list, output.detach().clone()], dim=0)
    if len(ids) > 0 and not show_all:
        gi = get_indices(extended_id_list, ids)
        print(extended_id_list, ids)
        print("indices: ", gi)
        output_list = output_list[gi]
        extended_id_list = [extended_id_list[i] for i in gi]
        time_list = [time_list[i] for i in gi]
    return output_list, extended_id_list, time_list


def getWeatherData(stid, dates):
    return asyncio.run(getWeatherByStationIDDate(stid, dates))


async def get_logic(i, rand_dates):
    stri = str(i)
    for ch in range(0, 5 - len(str(i))):
        stri = "0" + stri
    for rd in rand_dates:
        data = await dwd.getSourcesByStationIDDate(stri, rd, delay=30)
        status_bar.next()
        if data[0] != "error":
            return data
    return ["error", "error"]
    # status_bar.next()


async def getSourceByStationIDDate(rd):
    global status_bar
    rang = 100000
    status_bar = Bar("Processing", max=rang * len(rd))
    list = await asyncio.gather(*[get_logic(i, rd) for i in range(0, rang)])
    return list


def getSourceData(rd):
    return asyncio.run(getSourceByStationIDDate(rd))


def create_feature(
    precipitation,
    precipitation_probability,
    precipitation_probability_6h,
    pressure_msl,
    temperature,
    sunshine,
    wind_direction,
    wind_speed,
    cloud_cover,
    dew_point,
    wind_gust_direction,
    wind_gust_speed,
    condition,
    relative_humidity,
    visibility,
    solar,
    icon,
    months,
    hours,
    pos,
):
    print(
        precipitation,
        precipitation_probability,
        precipitation_probability_6h,
        pressure_msl,
        temperature,
        sunshine,
        wind_direction,
        wind_speed,
        cloud_cover,
        dew_point,
        wind_gust_direction,
        wind_gust_speed,
        condition,
        relative_humidity,
        visibility,
        solar,
        icon,
        months,
        hours,
        pos,
    )
    features = []
    indx = [
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
    ]
    if precipitation or precipitation == "true":
        features.append("precipitation")
        indx[0] = features.index("precipitation")
    if precipitation_probability or precipitation_probability == "true":
        features.append("precipitation_probability")
        indx[1] = features.index("precipitation_probability")
    if precipitation_probability_6h or precipitation_probability_6h == "true":
        features.append("precipitation_probability_6h")
        indx[2] = features.index("precipitation_probability_6h")
    if pressure_msl or pressure_msl == "true":
        features.append("pressure_msl")
        indx[3] = features.index("pressure_msl")
    if temperature or temperature == "true":
        features.append("temperature")
        indx[4] = features.index("temperature")
    if sunshine or sunshine == "true":
        features.append("sunshine")
        indx[5] = features.index("sunshine")
    if wind_speed or wind_speed == "true":
        features.append("wind_speed")
        indx[6] = features.index("wind_speed")
    if cloud_cover or cloud_cover == "true":
        features.append("cloud_cover")
        indx[7] = features.index("cloud_cover")
    if dew_point or dew_point == "true":
        features.append("dew_point")
        indx[8] = features.index("dew_point")
    if wind_gust_speed or wind_gust_speed == "true":
        features.append("wind_gust_speed")
        indx[9] = features.index("wind_gust_speed")
    if relative_humidity or relative_humidity == "true":
        features.append("relative_humidity")
        indx[10] = features.index("relative_humidity")
    if visibility or visibility == "true":
        features.append("visibility")
        indx[11] = features.index("visibility")
    if solar or solar == "true":
        features.append("solar")
        indx[12] = features.index("solar")
    if wind_direction or wind_direction == "true":
        for i in range(9):
            features.append(f"wind_direction{i}")
        indx[13] = features.index("wind_direction0")
    if wind_gust_direction or wind_gust_direction == "true":
        for i in range(9):
            features.append(f"wind_gust_direction{i}")
        indx[14] = features.index("wind_gust_direction0")
    if icon or icon == "true":
        for i in range(21):
            features.append(f"icon{i}")
        indx[15] = features.index("icon0")
    if condition or condition == "true":
        for i in range(21):
            features.append(f"condition{i}")
        indx[16] = features.index("condition0")
    if months == "true":
        months = True
    elif months == "false":
        months = False
    if hours == "true":
        hours = True
    elif hours == "false":
        hours = False
    if pos == "true":
        pos = True
    elif pos == "false":
        pos = False
    return features, indx, months, hours, pos
