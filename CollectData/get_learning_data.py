from CollectData import get_DWD_data as dwd # otherwise lstm cant import this
import asyncio
from datetime import datetime as dt
from datetime import timedelta as td
import pandas as pd
import numpy as np # euclidean: 34:03 each epoche
from progress.bar import Bar
import math
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

icons = ["clear-day", "clear-night", "partly-cloudy-day", "partly-cloudy-night", "cloudy", "fog", "wind", "rain", "sleet", "snow", "hail", "thunderstorm", "dry", "moist", "wet", "rime", "ice", "glaze", "not dry", "reserved", None]
hourly_bar = Bar('Processing', max=20)
daily_bar = 0

def load_stations_csv():
    return pd.read_csv('stations.csv',dtype={'ID': object}) # otherwise lstm cant load this

async def euclidean(chosen, unchosen):
    cols=['lon','lat','height']
    unchosen["dist"] = np.linalg.norm(chosen[cols].values - unchosen[cols].values, axis=1)
    return unchosen

async def chooseByNearest(chosen, unchosen, addLon=0, addLat=0):
    # chosen["lon"] += addLon
    # chosen["lat"] += addLat
    chosen.iloc[0,4] += addLon
    chosen.iloc[0,3] += addLat
    dists = await euclidean(chosen.head(1), unchosen)
    dists.sort_values(by=['dist'], ascending=True, inplace=True)
    return dists["ID"]

async def getCities(cityP,cityID,distance):
    chosen = cityP[cityP['ID']==cityID]
    unchosen = cityP.drop(cityP[cityP['ID']==cityID].index)
    if len(chosen) < 1:
        return "Error"
    if len(unchosen) < 1:
        return "Error"
    if distance.lower() == "near":
        return pd.DataFrame(await chooseByNearest(chosen, unchosen))
    elif distance.lower() == "close" or distance.lower() == "far":
        if distance.lower() == "close":
            return pd.concat([  await chooseByNearest(chosen, unchosen), await chooseByNearest(chosen, unchosen, addLat=0.5), 
                                await chooseByNearest(chosen, unchosen, addLon=0.5), await chooseByNearest(chosen, unchosen, addLat=-0.5), 
                                await chooseByNearest(chosen, unchosen, addLon=-0.5)], axis=1)
        return pd.concat([  await chooseByNearest(chosen, unchosen), await chooseByNearest(chosen, unchosen, addLat=0.5), 
                            await chooseByNearest(chosen, unchosen, addLon=0.5), await chooseByNearest(chosen, unchosen, addLat=-0.5), 
                            await chooseByNearest(chosen, unchosen, addLon=-0.5), await chooseByNearest(chosen, unchosen, addLat=2.5), 
                            await chooseByNearest(chosen, unchosen, addLon=2.5), await chooseByNearest(chosen, unchosen, addLat=-2.5), 
                            await chooseByNearest(chosen, unchosen, addLon=-2.5)], axis=1)
    return pd.DataFrame(columns=["error"])

async def filter_dataHourly(data):
    data.drop(["source_id"], axis=1, inplace=True)
    data["icon"] = data["icon"].isin(icons).astype(float)
    data["condition"] = data["condition"].isin(icons).astype(float)
    if "fallback_source_ids" in data:
        data.drop(["fallback_source_ids"], axis=1, inplace=True)
    data.drop(["timestamp"], axis=1, inplace=True)
    # data.fillna(0,inplace=True)
    return data

async def filter_dataComplete(data):
    data = await filter_dataHourly(data)
    data.fillna(0,inplace=True)
    return data

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
            newdata = await dwd.getWeatherByStationIDDate(cities.iloc[i,j], date, wait=False)
            i += 1
            if len(newdata) < 24:
                if lcount >= lc*5:
                    print("error occured")
                    return pd.DataFrame(columns=["error"])
                lcount +=1
                continue
            newdata = pd.DataFrame(newdata)
            
            newdata = await filter_dataHourly(newdata)
            data = pd.concat([data, pd.DataFrame(newdata)], ignore_index=True, axis=1)
    data.fillna(0,inplace=True)
    return data.round(3)

async def joinDataComplete(cityID, cities, lc, date):
    x = dt.now()
    data = await dwd.getWeatherByStationIDDate(cityID, date)
    if len(data) <= 24:
        xer1 = dt.now()
        print("Dauer bis zum 1. error:", xer1-x)
        return pd.DataFrame(columns=["error"])
    x0 = dt.now()
    print("Dauer bis Datei 1 gefunden:", x0-x)
    data = pd.DataFrame(data)
    data = data[:24]
    for j in range(cities.shape[1]):
        x1 = dt.now()
        i = 0
        lcount = lc
        while i-lc < lcount:
            newdata = await dwd.getWeatherByStationIDDate(cities.iloc[i,j], date, wait=False)
            i += 1
            if len(newdata) <= 24:
                if i-lc >= 12:
                    xer2 = dt.now()
                    print("Dauer bis zum 2. error:", xer2-x1)
                    return pd.DataFrame(columns=["error"])
                lcount +=1
                continue
            x2 = dt.now()
            print("Dauer bis Datei", (j+2), "gefunden:", x2-x1)
            newdata = newdata[:24]
            # data.append(filter_dataComplete(pd.DataFrame(newdata)), inplace=True)
            data = pd.concat([data, pd.DataFrame(newdata)], ignore_index=True)
            x3 = dt.now()
            print("Dauer bis Datei", (j+2), "concatted ist:", x2-x1)
    return await filter_dataComplete(data)

async def labelMinMax(cityID, date):
    data = await dwd.getWeatherByStationIDDate(cityID, date)
    if data[0] == "error":
        return pd.DataFrame(columns=["error"])
    label_Data = pd.DataFrame(data)
    label_Data = await filter_dataComplete(label_Data)
    conditions = label_Data["condition"].value_counts().index[0]
    icons = label_Data["icon"].value_counts().index[0]
    return pd.DataFrame({   "temperature_min": [min(label_Data["temperature"])], "temperature_max": max(label_Data["temperature"]),
                            "wind_speed_min": min(label_Data["wind_speed"]), "wind_speed_max": max(label_Data["wind_speed"]),
                            "condition": conditions, "icon": icons})

async def label24(cityID, date):
    data = await dwd.getWeatherByStationIDDate(cityID, date)
    if data[0] == "error":
        return pd.DataFrame(columns=["error"])
    label_Data = pd.DataFrame(data)
    label_Data = await filter_dataComplete(label_Data)
    return label_Data[["temperature","wind_direction","wind_speed","visibility","condition","icon"]]

async def get_DataHourlyAsync(row, cityP, date):
    cityID = row.ID
    cities = await getCities(cityP,cityID,"near")
    train_Data = await joinDataHourly(cityID, cities, 4, date.date())
    if "error" in train_Data or len(train_Data) < 25:
        hourly_bar.next()
        return pd.DataFrame(columns=["error"])
    if len(train_Data) > 25:
        train_Data = train_Data[:25]
    date = date + td(days=1)
    label_Data24 = await label24(cityID, date)
    if "error" in label_Data24 or len(label_Data24) < 24:
        hourly_bar.next()
        return pd.DataFrame(columns=["error"])
    if len(label_Data24) > 24:
        label_Data24 = label_Data24[:24]
    label_Data = train_Data[[3,4,5,9,12,16]]
    hourly_bar.next()
    return train_Data[:-1].to_numpy(), label_Data.iloc[1:].to_numpy(), label_Data24.to_numpy()

async def get_DataDailyAsync(row, cityP, date):
    cityID = row.ID
    cities = await getCities(cityP,cityID, "far")
    x = dt.now()
    train_Data = await joinDataComplete(cityID, cities, 1, date.date())
    print("Dauer von joinDataComplete:",dt.now()-x)
    if "error" in train_Data:
        hourly_bar.next()
        return pd.DataFrame(columns=["error"]) 
    # train_Data = await joinDataMinMax(cityID, cities, 4, date.date())
    date = date + td(days=1)
    label_Data_Daily = await labelMinMax(cityID, date)
    if "error" in label_Data_Daily or len(label_Data_Daily) != 1:
        hourly_bar.next()
        return pd.DataFrame(columns=["error"])
    hourly_bar.next()
    return [train_Data, label_Data_Daily]

async def DataHourlyAsync(cityloop, cityP, date):
    global hourly_bar 
    hourly_bar = Bar('Processing', max=len(cityloop))
    lists = await asyncio.gather(
        *[get_DataHourlyAsync(row, cityP, date) for row in cityloop.itertuples(index=False)]
        )
    train_np = np.zeros(shape=(1, 1))
    label_np = np.zeros(shape=(1, 1))
    label24_np = np.zeros(shape=(1, 1))
    i=0
    for l in lists:
        if len(l) > 0:
            if i == 0:
                train_np = np.array([l[0]])
                label_np = np.array([l[1]])
                label24_np = np.array([l[2]])
                i+=1
            else:
                train_np = np.concatenate([train_np,np.array([l[0]])], axis=1)
                label_np = np.concatenate([label_np,np.array([l[1]])], axis=1)
                label24_np = np.concatenate([label24_np,np.array([l[2]])], axis=1)
    # print("train", train_np.shape)
    # print("label", label_np.shape)
    # print("label24", label24_np.shape)
    if train_np.shape[1] == 1:
        return "error", "error", "error" 
    return train_np.reshape(-1, train_np.shape[2]), label_np.reshape(-1, label_np.shape[2]), label24_np.reshape(-1, label24_np.shape[2])

async def DataDailyAsync(cityloop, cityP, date):
    global daily_bar 
    daily_bar = Bar('Processing', max=len(cityP))
    lists = await asyncio.gather(
        *[get_DataDailyAsync(row, cityP, date) for row in cityloop.itertuples(index=False)]
        )
    train_np = np.zeros(shape=(1, 240, 17))
    label_np = np.zeros(shape=(1, 6))
    for l in lists:
        if len(l) > 0:
            train_np = np.concatenate([train_np,np.array([l[0]])], axis=1)
            label_np = np.concatenate([label_np,np.array([l[1]])], axis=1)
    if train_np.shape[1] == 1:
        return "error", "error", "error" 
    return train_np, label_np

def gen_trainDataDaily_Async():
    cityP = load_stations_csv()
    minTime = dt.strptime(min(cityP["start"]), '%Y-%m-%d %H:%M:%S')
    actualTime = dt.now()
    duration = actualTime - minTime
    print(duration, "=", actualTime, "-", minTime)
    for d in range(duration.days-1):
        date = minTime + td(days=d)
        # x,y = asyncio.run(DataDailyAsync(cityP, date))
        redos = 15
        # print(len(cityP),"/", redos)
        s = math.floor(len(cityP)/redos)
        # print("=",s)
        for i in range(redos):
            # print(i)
            if i == redos-1:
                x,y = asyncio.run(DataDailyAsync(cityP[s*i:], cityP, date))
            else:
                x,y = asyncio.run(DataDailyAsync(cityP[s*i:s*(i+1)], cityP, date))
            if x == "error":
                print("was error so code continued")
                continue
            print("train", x.shape)
            print("label", y.shape)
        yield x,y,d

def gen_trainDataHourly_Async():
    cityP = load_stations_csv()
    minTime = dt.strptime(min(cityP["start"]), '%Y-%m-%d %H:%M:%S')
    actualTime = dt.now()
    duration = actualTime - minTime
    print(duration, "=", actualTime, "-", minTime)
    for d in range(duration.days-1):
        date = minTime + td(days=d)
        redos = 15
        # print(len(cityP),"/", redos)
        s = math.floor(len(cityP)/redos)
        # print("=",s)
        for i in range(redos):
            # print(i)
            if i == redos-1:
                x,y,z = asyncio.run(DataHourlyAsync(cityP[s*i:], cityP, date))
            else:
                x,y,z = asyncio.run(DataHourlyAsync(cityP[s*i:s*(i+1)], cityP, date))
            if x == "error":
                print("was error so code continued")
                continue
            print("train", x.shape)
            print("label", y.shape)
            print("label24", z.shape)
            
            # print(x)
            # print(y)
            # print(z)
            yield x,y,z,d

def predictDataHourly(date=dt.now()-td(days=2), back=0, city="", id=""):
    cityP = load_stations_csv()
    if city != "":
        cityP = cityP[cityP['Name']==city]
    elif id != "":
        cityP = cityP[cityP['ID']==id]

    date = date - td(days=back)

    return asyncio.run(DataHourlyAsync(cityP, cityP, date))

def predictDataDaily(date=dt.now()-td(days=2), back=0, city="", id=""):
    cityP = load_stations_csv()
    if city != "":
        cityP = cityP[cityP['Name']==city]
    elif id != "":
        cityP = cityP[cityP['ID']==id]

    date = date - td(days=back)

    return asyncio.run(DataDailyAsync(cityP, date))

# if __name__ == "__main__":
#     trainData = gen_trainDataHourly()
#     for i in range(1):
#         train, label = next(trainData)
#         print(train)
#         print(label)
        


