from CollectData import get_DWD_data as dwd # otherwise lstm cant import this
from datetime import datetime as dt
from datetime import timedelta as td
import datetime
import pandas as pd
import numpy as np

icons = ["clear-day", "clear-night", "partly-cloudy-day", "partly-cloudy-night", "cloudy", "fog", "wind", "rain", "sleet", "snow", "hail", "thunderstorm", "dry", "moist", "wet", "rime", "ice", "glaze", "not dry", "reserved", None]


def load_stations_csv():
    return pd.read_csv('stations.csv',dtype={'ID': object}) # otherwise lstm cant load this

def euclidean(chosen, unchosen):
    cols=['lon','lat','height']
    unchosen["dist"] = np.linalg.norm(chosen[cols].values - unchosen[cols].values, axis=1)
    return unchosen

def chooseByNearest(chosen, unchosen):
    dists = euclidean(chosen.head(1), unchosen)
    dists.sort_values(by=['dist'], ascending=True, inplace=True)
    return dists

def getCities(cityP,cityName):
    chosen = cityP[cityP['Name']==cityName]
    unchosen = cityP.drop(cityP[cityP['Name']==cityName].index)
    if len(chosen) < 1:
        return "Error"
    if len(unchosen) < 1:
        return "Error"
    return chooseByNearest(chosen, unchosen)

def filter_data(data):
    data.drop(["source_id"], axis=1, inplace=True)
    # data["icon"] = [icons.index(_) for _ in data["icon"]]
    # data["condition"] = [icons.index(_) for _ in data["condition"]]
    data["icon"] = data["icon"].isin(icons).astype(float)
    data["condition"] = data["condition"].isin(icons).astype(float)
    if "fallback_source_ids" in data:
        data.drop(["fallback_source_ids"], axis=1, inplace=True)
    data.fillna(0,inplace=True)
    return data


def joinData(cityID, cities, lc, date):
    data = dwd.getWeatherByStationIDDate(cityID, date)
    
    if data[0]== "error":
        return pd.DataFrame(columns=["error"])
    data =pd.DataFrame(data)
    data = filter_data(data)
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

    data = data.drop(["timestamp"], axis=1)
    return data


def label(cityID, date):
    data = dwd.getWeatherByStationIDDate(cityID, date)
    if data[0] == "error":
        return pd.DataFrame(columns=["error"])
    label_Data = pd.DataFrame(data)
    label_Data = filter_data(label_Data)
    return label_Data

def gen_trainDataDaily():
    cityP = load_stations_csv()
    minTime = dt.strptime(min(cityP["start"]), '%Y-%m-%d %H:%M:%S')
    actualTime = dt.now()
    duration = actualTime - minTime
    print(duration, "=", actualTime, "-", minTime)
    for d in range(duration.days-1):
        date = minTime + td(days=d)
        for i in range(len(cityP)):
            cityName = cityP.iloc[i]["Name"]
            cityID = cityP.iloc[i]["ID"]
            cities = getCities(cityP,cityName)
            train_Data = joinData(cityID, cities, 4, date.date())
            if "error" in train_Data:
                continue
            date = minTime + td(days=d+1)
            # print(cityID, date.date())
            label_Data = label(cityID, date.date())
            label_Data = label_Data.drop(["timestamp"], axis=1)
            if "error" in label_Data:
                continue
            yield train_Data.values.to(), label_Data.values.tolist()

def gen_trainDataHourly():
    cityP = load_stations_csv()
    minTime = dt.strptime(min(cityP["start"]), '%Y-%m-%d %H:%M:%S')
    actualTime = dt.now()
    duration = actualTime - minTime
    print(duration, "=", actualTime, "-", minTime)
    for d in range(duration.days-1):
        date = minTime + td(days=d)
        for row in cityP.itertuples(index=False):
            cityName = row.Name
            cityID = row.ID
            cities = getCities(cityP,cityName)
            train_Data = joinData(cityID, cities, 4, date.date())
            if "error" in train_Data or len(train_Data) > 25:
                continue
            date = minTime + td(days=d+1)
            label_Data = train_Data[["precipitation","pressure_msl","sunshine","temperature","wind_direction","wind_speed","cloud_cover","dew_point","relative_humidity","visibility","wind_gust_direction","wind_gust_speed","condition","precipitation_probability","precipitation_probability_6h","solar","icon"]]
            # label_Data = label_Data.drop(["timestamp"], axis=1)
            if "error" in label_Data:
                continue

            yield train_Data[:-1].to_numpy(),label_Data.iloc[1:].to_numpy()


# if __name__ == "__main__":
#     trainData = gen_trainDataHourly()
#     for i in range(1):
#         train, label = next(trainData)
#         print(train)
#         print(label)
        


