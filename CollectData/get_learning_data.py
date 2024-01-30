from CollectData import get_DWD_data as dwd # otherwise lstm cant import this
from datetime import datetime as dt
from datetime import timedelta as td
import pandas as pd
from dask import dataframe as ddf
import numpy as np
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
        return chooseByNearest(chosen, unchosen)
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
    # data.fillna(0,inplace=True)
    return data

def filter_dataComplete(data):
    data = filter_dataHourly(data)
    data.drop(["timestamp"], axis=1, inplace=True)
    data.fillna(0,inplace=True)
    return data

def filter_data2(data):
    label_Data = train_Data[["precipitation","pressure_msl","sunshine","temperature","wind_direction","wind_speed","cloud_cover","dew_point","relative_humidity","visibility","wind_gust_direction","wind_gust_speed","condition","precipitation_probability","precipitation_probability_6h","solar","icon"]]
    data.drop(["source_id"], axis=1, inplace=True)
    data.drop(["timestamp"], axis=1, inplace=True)
    data["icon"] = data["icon"].isin(icons).astype(float)
    data["condition"] = data["condition"].isin(icons).astype(float)
    if "fallback_source_ids" in data:
        data.drop(["fallback_source_ids"], axis=1, inplace=True)
    return data

def joinDataHourly(cityID, cities, lc, date):
    data = dwd.getWeatherByStationIDDate(cityID, date)
    if data[0]== "error":
        return pd.DataFrame(columns=["error"])
    data =filter_dataHourly(pd.DataFrame(data))
    # mdata = ddf.from_pandas(data, npartitions=5)
    for j in range(cities.shape[1]):
        i = 0
        # m = dt.now()
        # while i < lc:
        #     newdata = dwd.getWeatherByStationIDDate(cities.iloc[i,j], date)
        #     i += 1
        #     if newdata[0] == "error":
        #         lc +=1
        #         continue
        #     newdata = ddf.from_pandas(filter_data(pd.DataFrame(newdata)), npartitions=5)
        #     mdata = ddf.merge(data, newdata, on="timestamp", how="outer", suffixes=("", "_" + str(i+4-lc))).compute()
        # m1 = dt.now()
        # newdata = []
        # i = 0
        # lc=4
        lcount = lc
        while i < lcount:
            newdata = dwd.getWeatherByStationIDDate(cities.iloc[i,j], date)
            i += 1
            if newdata[0] == "error":
                lcount +=1
                continue
            newdata = pd.DataFrame(newdata)
            newdata = filter_dataHourly(newdata)
            data = pd.merge(data, newdata, on="timestamp", how="outer", suffixes=("", "_" + str(i+4-lc)))

    
    # m2 = dt.now()

    # print("dask:", m1-m)
    # print("pandas:", m2-m1)

    # if data.equals(mdata):
    #     print("Daten sind gleich")
    # else:
    #     print("Fehler Daten sind nicht gleich")

    data.drop(["timestamp"], axis=1, inplace=True)
    data.fillna(0,inplace=True)
    return data


def joinDataMinMax(cityID, cities, lc, date):
    data = dwd.getWeatherByStationIDDate(cityID, date)
    if data[0]== "error":
        return pd.DataFrame(columns=["error"])
    data = filter_data2(pd.DataFrame(data))
    i = 0
    while i < lc:
        newdata = dwd.getWeatherByStationIDDate(cities.iloc[i,0], date)
        i += 1
        if newdata[0] == "error":
            lc +=1
            continue
        newdata = pd.DataFrame(newdata)
        newdata = filter_data(newdata)
        data = pd.merge(data, newdata, on="timestamp", how="outer", suffixes=("", "_" + str(i+4-lc)))

    data.drop(["timestamp"], axis=1, inplace=True)
    data.fillna(0,inplace=True)
    return data

def joinDataComplete(cityID, cities, lc, date):
    data = dwd.getWeatherByStationIDDate(cityID, date)
    if len(data) <= 24:
        return pd.DataFrame(columns=["error"])
    data = pd.DataFrame(data)
    data = data[:24]
    for j in range(cities.shape[1]):
        i = 0
        lcount = lc
        while i < lcount:
            newdata = dwd.getWeatherByStationIDDate(cities.iloc[i,j], date)
            i += 1
            if len(newdata) <= 24:
                lcount +=1
                continue
            newdata = newdata[:24]
            # data.append(filter_dataComplete(pd.DataFrame(newdata)), inplace=True)
            data = pd.concat([data, pd.DataFrame(newdata)], ignore_index=True)
    return filter_dataComplete(data)

def labelMinMax(cityID, date):
    data = dwd.getWeatherByStationIDDate(cityID, date)
    if data[0] == "error":
        return pd.DataFrame(columns=["error"])
    label_Data = pd.DataFrame(data)
    label_Data = filter_dataComplete(label_Data)
    conditions = label_Data["condition"].value_counts().head(1).iloc[0]
    icons = label_Data["condition"].value_counts().head(1).iloc[0]
    print(conditions)
    print(icons)
    return pd.DataFrame({   "temperature_min": [min(label_Data["temperature"])], "temperature_max": max(label_Data["temperature"]),
                            "wind_speed_min": min(label_Data["wind_speed"]), "wind_speed_max": max(label_Data["wind_speed"]),
                            "condition": conditions, "icon": icons})

def label24(cityID, date):
    data = dwd.getWeatherByStationIDDate(cityID, date)
    if data[0] == "error":
        return pd.DataFrame(columns=["error"])
    label_Data = pd.DataFrame(data)
    label_Data = filter_dataHourly(label_Data)
    return label_Data[["temperature","wind_direction","wind_speed","visibility","condition","icon"]]

def gen_trainDataDaily():
    cityP = load_stations_csv()
    minTime = dt.strptime(min(cityP["start"]), '%Y-%m-%d %H:%M:%S')
    actualTime = dt.now()
    duration = actualTime - minTime
    print(duration, "=", actualTime, "-", minTime)
    for d in range(duration.days-1):
        date = minTime + td(days=d)

        train_np = np.zeros(shape=(1, 1))
        label_np = np.zeros(shape=(1, 1))
        label24_np = np.zeros(shape=(1, 1))
        i=0
        for row in tqdm(cityP.itertuples(index=False),total=1562):
            cityName = row.Name
            cityID = row.ID
            cities = getCities(cityP,cityName, "far")
            train_Data = joinDataComplete(cityID, cities, 1, date.date())
            # train_Data = joinDataMinMax(cityID, cities, 4, date.date())
            if "error" in train_Data:
                continue
            date = minTime + td(days=d+1)
            label_Data24 = labelMinMax(cityID, date)
            if "error" in label_Data24 or len(label_Data24) < 24:
                continue
            x3 = dt.now()
            if len(label_Data24) > 24:
                label_Data24 = label_Data24[:24]
            # label_Data = train_Data[["precipitation","pressure_msl","sunshine","temperature","wind_direction","wind_speed","cloud_cover","dew_point","relative_humidity","visibility","wind_gust_direction","wind_gust_speed","condition","precipitation_probability","precipitation_probability_6h","solar","icon"]]
            label_Data = train_Data[["temperature","wind_direction","wind_speed","visibility","condition","icon"]]
            label_Data24 = label_Data24[["temperature","wind_direction","wind_speed","visibility","condition","icon"]]
            if i == 0:
                train_np = train_Data[:-1].to_numpy()
                label_np = label_Data.iloc[1:].to_numpy()
                label24_np = label_Data24.to_numpy()
            else:
                train_np += train_Data[:-1].to_numpy()
                label_np += label_Data.iloc[1:].to_numpy()
                label24_np += label_Data24.to_numpy()
            i+1
        yield train_np, label_np, label24_np

def gen_trainDataHourly():
    cityP = load_stations_csv()
    minTime = dt.strptime(min(cityP["start"]), '%Y-%m-%d %H:%M:%S')
    actualTime = dt.now()
    duration = actualTime - minTime
    print(duration, "=", actualTime, "-", minTime)
    for d in range(duration.days-1):
        date = minTime + td(days=d)

        train_np = np.zeros(shape=(1, 1))
        label_np = np.zeros(shape=(1, 1))
        label24_np = np.zeros(shape=(1, 1))
        i=0
        for row in tqdm(cityP.itertuples(index=False),total=1562):
            cityName = row.Name
            cityID = row.ID
            cities = getCities(cityP,cityName,"near")
            train_Data = joinDataHourly(cityID, cities, 4, date.date())
            if "error" in train_Data or len(train_Data) < 25:
                continue
            if len(train_Data) > 25:
                train_Data = train_Data[:25]
            date = minTime + td(days=d+1)
            label_Data24 = label24(cityID, date)
            if "error" in label_Data24 or len(label_Data24) < 24:
                continue
            x3 = dt.now()
            if len(label_Data24) > 24:
                label_Data24 = label_Data24[:24]
            # label_Data = train_Data[["precipitation","pressure_msl","sunshine","temperature","wind_direction","wind_speed","cloud_cover","dew_point","relative_humidity","visibility","wind_gust_direction","wind_gust_speed","condition","precipitation_probability","precipitation_probability_6h","solar","icon"]]
            label_Data = train_Data[["temperature","wind_direction","wind_speed","visibility","condition","icon"]]
            if i == 0:
                train_np = train_Data[:-1].to_numpy()
                label_np = label_Data.iloc[1:].to_numpy()
                label24_np = label_Data24.to_numpy()
            else:
                train_np += train_Data[:-1].to_numpy()
                label_np += label_Data.iloc[1:].to_numpy()
                label24_np += label_Data24.to_numpy()
            i+1
        yield train_np, label_np, label24_np


# if __name__ == "__main__":
#     trainData = gen_trainDataHourly()
#     for i in range(1):
#         train, label = next(trainData)
#         print(train)
#         print(label)
        


