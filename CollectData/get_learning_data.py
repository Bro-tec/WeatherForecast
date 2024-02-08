from CollectData import get_DWD_data as dwd # otherwise lstm cant import this
import asyncio
from datetime import datetime as dt
from datetime import timedelta as td
import pandas as pd
import numpy as np # euclidean: 34:03 each epoche
from tqdm import tqdm

icons = ["clear-day", "clear-night", "partly-cloudy-day", "partly-cloudy-night", "cloudy", "fog", "wind", "rain", "sleet", "snow", "hail", "thunderstorm", "dry", "moist", "wet", "rime", "ice", "glaze", "not dry", "reserved", None]


def load_stations_csv():
    return pd.read_csv('stations.csv',dtype={'ID': object}) # otherwise lstm cant load this

def euclidean(chosen, unchosen):
    cols=['lon','lat','height']
    unchosen["dist"] = np.linalg.norm(chosen[cols].values - unchosen[cols].values, axis=1)
    return unchosen

def chooseByNearest(chosen, unchosen, addLon=0, addLat=0):
    chosen["lon"] += addLon
    chosen["lat"] += addLat
    dists = euclidean(chosen.head(1), unchosen)
    dists.sort_values(by=['dist'], ascending=True, inplace=True)
    return dists["ID"]

def getCities(cityP,cityName,distance):
    chosen = cityP[cityP['Name']==cityName]
    unchosen = cityP.drop(cityP[cityP['Name']==cityName].index)
    if len(chosen) < 1:
        return "Error"
    if len(unchosen) < 1:
        return "Error"
    if distance.lower() == "near":
        return pd.DataFrame(chooseByNearest(chosen, unchosen))
    elif distance.lower() == "close" or distance.lower() == "far":
        if distance.lower() == "close":
            return pd.concat([  chooseByNearest(chosen, unchosen), chooseByNearest(chosen, unchosen, addLat=0.5), 
                                chooseByNearest(chosen, unchosen, addLon=0.5), chooseByNearest(chosen, unchosen, addLat=-0.5), 
                                chooseByNearest(chosen, unchosen, addLon=-0.5)], axis=1)
        return pd.concat([  chooseByNearest(chosen, unchosen), chooseByNearest(chosen, unchosen, addLat=0.5), 
                            chooseByNearest(chosen, unchosen, addLon=0.5), chooseByNearest(chosen, unchosen, addLat=-0.5), 
                            chooseByNearest(chosen, unchosen, addLon=-0.5), chooseByNearest(chosen, unchosen, addLat=2.5), 
                            chooseByNearest(chosen, unchosen, addLon=2.5), chooseByNearest(chosen, unchosen, addLat=-2.5), 
                            chooseByNearest(chosen, unchosen, addLon=-2.5)], axis=1)
    return pd.DataFrame(columns=["error"])

def filter_dataHourly(data):
    data.drop(["source_id"], axis=1, inplace=True)
    data["icon"] = data["icon"].isin(icons).astype(float)
    data["condition"] = data["condition"].isin(icons).astype(float)
    if "fallback_source_ids" in data:
        data.drop(["fallback_source_ids"], axis=1, inplace=True)
    data.drop(["timestamp"], axis=1, inplace=True)
    # data.fillna(0,inplace=True)
    return data

def filter_dataComplete(data):
    data = filter_dataHourly(data)
    data.fillna(0,inplace=True)
    return data

def joinDataHourly(cityID, cities, lc, date):
    data = dwd.getWeatherByStationIDDate(cityID, date)
    if len(data) < 24:
        return pd.DataFrame(columns=["error"])
    data =filter_dataHourly(pd.DataFrame(data))
    for j in range(cities.shape[1]):
        i = 0
        lcount = lc
        while i < lcount:
            newdata = dwd.getWeatherByStationIDDate(cities.iloc[i,j], date)
            i += 1
            if len(newdata) < 24:
                if i-lc >= 3:
                    return pd.DataFrame(columns=["error"])
                lcount +=1
                continue
            newdata = pd.DataFrame(newdata)
            newdata = filter_dataHourly(newdata)
            data = pd.concat([data, pd.DataFrame(newdata)], ignore_index=True, axis=1)
    data.fillna(0,inplace=True)
    return data

def joinDataComplete(cityID, cities, lc, date):
    x = dt.now()
    data = dwd.getWeatherByStationIDDate(cityID, date)
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
            newdata = dwd.getWeatherByStationIDDate(cities.iloc[i,j], date)
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
    return filter_dataComplete(data)

def labelMinMax(cityID, date):
    data = dwd.getWeatherByStationIDDate(cityID, date)
    if data[0] == "error":
        return pd.DataFrame(columns=["error"])
    label_Data = pd.DataFrame(data)
    label_Data = filter_dataComplete(label_Data)
    conditions = label_Data["condition"].value_counts().index[0]
    icons = label_Data["icon"].value_counts().index[0]
    return pd.DataFrame({   "temperature_min": [min(label_Data["temperature"])], "temperature_max": max(label_Data["temperature"]),
                            "wind_speed_min": min(label_Data["wind_speed"]), "wind_speed_max": max(label_Data["wind_speed"]),
                            "condition": conditions, "icon": icons})

def label24(cityID, date):
    data = dwd.getWeatherByStationIDDate(cityID, date)
    if data[0] == "error":
        return pd.DataFrame(columns=["error"])
    label_Data = pd.DataFrame(data)
    label_Data = filter_dataComplete(label_Data)
    return label_Data[["temperature","wind_direction","wind_speed","visibility","condition","icon"]]

async def get_DataHourlyAsync(row, cityP, date):
    cityName = row.Name
    cityID = row.ID
    cities = getCities(cityP,cityName,"near")
    # x = dt.now()
    train_Data = joinDataHourly(cityID, cities, 4, date.date())
    if "error" in train_Data or len(train_Data) < 25:
        # print("joinDataHourly error dauer:", dt.now()-x)
        return pd.DataFrame(columns=["error"])
    # print("joinDataHourly dauer:", dt.now()-x)
    if len(train_Data) > 25:
        train_Data = train_Data[:25]
    date = date + td(days=1)
    label_Data24 = label24(cityID, date)
    if "error" in label_Data24 or len(label_Data24) < 24:
        # print("joinDataHourly unnötiger error dauer:", dt.now()-x)
        return pd.DataFrame(columns=["error"])
    if len(label_Data24) > 24:
        label_Data24 = label_Data24[:24]
    # label_Data = train_Data[["precipitation","pressure_msl","sunshine","temperature","wind_direction","wind_speed","cloud_cover","dew_point","relative_humidity","visibility","wind_gust_direction","wind_gust_speed","condition","precipitation_probability","precipitation_probability_6h","solar","icon"]]
    # label_Data = train_Data[["temperature","wind_direction","wind_speed","visibility","condition","icon"]]
    label_Data = train_Data[[3,4,5,9,12,16]]
    return [train_Data[:-1], label_Data.iloc[1:], label_Data24]

async def get_DataDailyAsync(row, cityP, date):
    cityName = row.Name
    cityID = row.ID
    cities = getCities(cityP,cityName, "far")
    x = dt.now()
    train_Data = joinDataComplete(cityID, cities, 1, date.date())
    print("Dauer von joinDataComplete:",dt.now()-x)
    if "error" in train_Data:
        return pd.DataFrame(columns=["error"]) 
    # train_Data = joinDataMinMax(cityID, cities, 4, date.date())
    date = date + td(days=1)
    label_Data_Daily = labelMinMax(cityID, date)
    if "error" in label_Data_Daily or len(label_Data_Daily) != 1:
        return pd.DataFrame(columns=["error"])
    return [train_Data, label_Data_Daily]

async def DataHourlyAsync(cityP, date):
    lists = await asyncio.gather(
        *[get_DataHourlyAsync(row, cityP, date) for row in tqdm(cityP.itertuples(index=False),total=1562)]
        )
    print(lists)

async def DataDailyAsync(cityP, date):
    lists = await asyncio.gather(
        *[get_DataHourlyAsync(row, cityP, date) for row in tqdm(cityP.itertuples(index=False),total=1562)]
        )
    print(lists)


def DataHourly(cityP, date):
    train_np = np.zeros(shape=(1, 1))
    label_np = np.zeros(shape=(1, 1))
    label24_np = np.zeros(shape=(1, 1))
    i=0
    for row in tqdm(cityP.itertuples(index=False),total=1562):
        cityName = row.Name
        cityID = row.ID
        cities = getCities(cityP,cityName,"near")
        # x = dt.now()
        train_Data = joinDataHourly(cityID, cities, 4, date.date())
        if "error" in train_Data or len(train_Data) < 25:
            # print("joinDataHourly error dauer:", dt.now()-x)
            continue
        # print("joinDataHourly dauer:", dt.now()-x)
        if len(train_Data) > 25:
            train_Data = train_Data[:25]
        date = date + td(days=1)
        label_Data24 = label24(cityID, date)
        if "error" in label_Data24 or len(label_Data24) < 24:
            # print("joinDataHourly unnötiger error dauer:", dt.now()-x)
            continue
        if len(label_Data24) > 24:
            label_Data24 = label_Data24[:24]
        # label_Data = train_Data[["precipitation","pressure_msl","sunshine","temperature","wind_direction","wind_speed","cloud_cover","dew_point","relative_humidity","visibility","wind_gust_direction","wind_gust_speed","condition","precipitation_probability","precipitation_probability_6h","solar","icon"]]
        # label_Data = train_Data[["temperature","wind_direction","wind_speed","visibility","condition","icon"]]
        label_Data = train_Data[[3,4,5,9,12,16]]
        if i == 0:
            train_np = np.array([train_Data[:-1].to_numpy()])
            label_np = np.array([label_Data.iloc[1:].to_numpy()])
            label24_np = np.array([label_Data24.to_numpy()])
        else:
            train_np = np.concatenate([train_np,np.array([train_Data[:-1].to_numpy()])])
            label_np = np.concatenate([label_np,np.array([label_Data.iloc[1:].to_numpy()])])
            label24_np = np.concatenate([label24_np,np.array([label_Data24.to_numpy()])])
        i+=1
        if i == 5:
            break
        
    return train_np, label_np, label24_np

def DataDaily(cityP, date):
    train_np = np.zeros(shape=(1, 240, 17))
    label_np = np.zeros(shape=(1, 6))
    
    for row in tqdm(cityP.itertuples(index=False),total=1562):
        cityName = row.Name
        cityID = row.ID
        cities = getCities(cityP,cityName, "far")
        x = dt.now()
        train_Data = joinDataComplete(cityID, cities, 1, date.date())
        print("Dauer von joinDataComplete:",dt.now()-x)
        if "error" in train_Data:
            continue
        # train_Data = joinDataMinMax(cityID, cities, 4, date.date())
        date = date + td(days=1)
        label_Data_Daily = labelMinMax(cityID, date)
        if "error" in label_Data_Daily or len(label_Data_Daily) != 1:
            continue
        if train_np.shape[0] == 5:
            break
        train_np = np.concatenate([train_np,[train_Data.to_numpy()]], axis=0)
        label_np = np.concatenate([label_np,label_Data_Daily.to_numpy()], axis=0)
    return train_np, label_np

def gen_trainDataDaily():
    cityP = load_stations_csv()
    minTime = dt.strptime(min(cityP["start"]), '%Y-%m-%d %H:%M:%S')
    actualTime = dt.now()
    duration = actualTime - minTime
    print(duration, "=", actualTime, "-", minTime)
    for d in range(duration.days-1):
        date = minTime + td(days=d)
        print(d)
        x,y = DataDaily(cityP, date)
        yield x,y,d

def gen_trainDataHourly():
    cityP = load_stations_csv()
    minTime = dt.strptime(min(cityP["start"]), '%Y-%m-%d %H:%M:%S')
    actualTime = dt.now()
    duration = actualTime - minTime
    print(duration, "=", actualTime, "-", minTime)
    for d in range(duration.days-1):
        date = minTime + td(days=d)
        print(d)
        x,y,z = DataHourly(cityP, date)
        yield x,y,z,d

def predictDataHourly(date=dt.now()-td(days=2), back=0, city="", id=""):
    cityP = load_stations_csv()
    if city != "":
        cityP = cityP[cityP['Name']==city]
    elif id != "":
        cityP = cityP[cityP['ID']==id]

    date = date - td(days=back)

    return DataHourly(cityP, date)

def predictDataDaily(date=dt.now()-td(days=2), back=0, city="", id=""):
    cityP = load_stations_csv()
    if city != "":
        cityP = cityP[cityP['Name']==city]
    elif id != "":
        cityP = cityP[cityP['ID']==id]

    date = date - td(days=back)

    return DataDaily(cityP, date)

# if __name__ == "__main__":
#     trainData = gen_trainDataHourly()
#     for i in range(1):
#         train, label = next(trainData)
#         print(train)
#         print(label)
        


