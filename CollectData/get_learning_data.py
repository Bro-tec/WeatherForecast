from CollectData import get_DWD_data as dwd # otherwise lstm cant import this
from datetime import datetime as dt
from datetime import timedelta as td
import datetime
import pandas as pd
import numpy as np




def load_stations_csv():
    df = pd.read_csv('stations.csv',dtype={'ID': object}) # otherwise lstm cant load this
    return df

def euclidean(chosen, unchosen):
    cols=['lon','lat','height']
    unchosen["dist"] = np.linalg.norm(chosen[cols].values - unchosen[cols].values, axis=1)
    return unchosen

def chooseByNearest(chosen, unchosen):
    dists = euclidean(chosen.head(1), unchosen)
    dists = dists.sort_values(by=['dist'], ascending=True)
    return dists

def getCities(cityP,cityName):
    chosen = cityP[cityP['Name']==cityName]
    unchosen = cityP.drop(cityP[cityP['Name']==cityName].index)
    if len(chosen) < 1:
        return "Error"
    if len(unchosen) < 1:
        return "Error"
    return chooseByNearest(chosen, unchosen)


def joinData(cityID, cities, lc, date):
    # print(cityID, cities, date)
    data = dwd.getWeatherByStationIDDate(cityID, date)
    
    if data[0]== "error":
        return pd.DataFrame(columns=["error"])
    # print(json.dumps(data, indent = 4, sort_keys = True))
    data =pd.DataFrame(data)
    data = data.drop(["source_id"], axis=1)
    if "fallback_source_ids" in data:
        data = data.drop(["fallback_source_ids"], axis=1)
    i = 0
    while i < lc:
        newdata = dwd.getWeatherByStationIDDate(cities.iloc[i,0], date)
        i += 1
        if newdata[0] == "error":
            lc +=1
            continue
        newdata = pd.DataFrame(newdata)
        newdata = newdata.drop(["source_id"], axis=1)
        if "fallback_source_ids" in newdata:
            newdata = newdata.drop(["fallback_source_ids"], axis=1)
        data = pd.merge(data, newdata, on="timestamp", how="outer", suffixes=("", "_" + str(i+4-lc)))

        #data = data.join(newdata, lsuffix='', rsuffix='{i}', how='outer').reset_index()
    # print("joinData",data.columns)
    # data = data.drop(["source_id", "source_id_0", "source_id_1", "source_id_2", "source_id_3", "fallback_source_ids_3", "fallback_source_ids_2", "fallback_source_ids_1", "fallback_source_ids_0"], axis=1)
    data = data.fillna(0)
    return data

def label(cityID, date):
    data = dwd.getWeatherByStationIDDate(cityID, date)
    if data[0] == "error":
        return pd.DataFrame(columns=["error"])
    label_Data = pd.DataFrame(data)
    # print("lableData",label_Data.columns)
    label_Data = label_Data.drop(["source_id"], axis=1)
    label_Data = label_Data.fillna(0)
    return label_Data

def gen_trainDataHourly():
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
            
            if "error" in label_Data:
                continue

            yield train_Data, label_Data

def label_nextday(df):
    df = df.drop(df.index[df["MESS_DATUM_ZEIT"].values == max(df["MESS_DATUM_ZEIT"])], axis=0)
    print(df)
    label = pd.DataFrame()
    for d in range(len(df)):
        ind = df.index[df["MESS_DATUM_ZEIT"].values == df["MESS_DATUM_ZEIT"][d]+ datetime.timedelta(days=1)]
        label = pd.concat([df.iloc[ind]])
    print(label)

# if __name__ == "__main__":
#     trainData = gen_trainDataHourly()
#     for i in range(1):
#         train, label = next(trainData)
#         print(train)
#         print(label)
        


