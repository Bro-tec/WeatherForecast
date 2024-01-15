import db_con as db
from datetime import datetime as dt
import datetime
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

sqlcities = "./Database/Cities.db"
sqltemperature = "./Database/Temperature.db"
sqlcloudiness = "./Database/Cloudiness.db"
sqlextreme_wind = "./Database/Extreme_wind.db"
sqlmoisture = "./Database/Moisture.db"
sqlpressure = "./Database/Pressure.db"
sqlwind = "./Database/Wind.db"

def ungroup(data):
    grouped = data.groupby('MESS_DATUM_ZEIT')
    newdf = pd.DataFrame()
    for group_name, group_data in grouped:
        usedf = group_data.drop(['MESS_DATUM_ZEIT'], axis=1)
        dir = {}
        for c in usedf.columns:
            dir[c+"_min"] = [min(usedf[c])]
            dir[c+"_max"] = [max(usedf[c])]
        mdz = pd.concat([pd.DataFrame(dir),pd.DataFrame({'MESS_DATUM_ZEIT':[group_name]})], axis=1)
        newdf = pd.concat([newdf, mdz])
    return newdf

def convert_date(data,vdata):
    if len(vdata) > 0:
        vdata["MESS_DATUM_ZEIT"] = [dt.utcfromtimestamp(d).date() for d in vdata["MESS_DATUM"]]
        #data = data.rename({"MESS_DATUM": "dat"})
        vdata = vdata.drop(["Cityname","MESS_DATUM"], axis=1) 
        vdata = ungroup(vdata)
        data = data.join(vdata, lsuffix='', rsuffix='_right', how='outer').reset_index()
    else:
        vdata = vdata.drop(["MESS_DATUM","Cityname"], axis=1) 
        newdata = pd.DataFrame()
        for c in vdata.columns:
            newdata[c+"_min"] = np.nan
            newdata[c+"_max"] = np.nan
        data = pd.concat([data,newdata])
    #data["MESS_DATUM_ZEIT"] = [(pd.to_timedelta(d, unit='s') + pd.to_datetime('1960-1-1')) for d in data["MESS_DATUM"]]
    return data

def convert_date2(data, vdata):
    if len(vdata) > 0:
        vdata["MESS_DATUM_ZEIT"] = [dt.utcfromtimestamp(d).date() for d in vdata["dat"]]
        vdata = vdata.drop(["dat","Cityname"], axis=1)
        vdata = ungroup(vdata)
        data = data.join(vdata, lsuffix='', rsuffix='_right', how='outer').reset_index()
    else:
        vdata = vdata.drop(["dat","Cityname"], axis=1) 
        newdata = pd.DataFrame()
        for c in vdata.columns:
            newdata[c+"_min"] = np.nan
            newdata[c+"_max"] = np.nan
        data = pd.concat([data,newdata])
    #data["MESS_DATUM_ZEIT"] = [(pd.to_timedelta(d, unit='s') + pd.to_datetime('1960-1-1')) for d in data["MESS_DATUM"]]
    return data

def euclidean(chosen, unchosen):
    cols=['longitude','latitude','altitude']
    unchosen["dist"] = np.linalg.norm(chosen[cols].values - unchosen[cols].values, axis=1)
    return unchosen

def chooseByFourNearest(chosen, unchosen):
    dists = euclidean(chosen.head(1), unchosen)
    dists = dists.sort_values(by=['dist'], ascending=True)
    return dists.head(4)

def getCities(cityP,cityName):
    chosen = cityP[cityP['Cityname']==cityName]
    unchosen = cityP.drop(cityP[cityP['Cityname']==cityName].index)
    if len(chosen) < 1:
        return "Error"
    if len(unchosen) < 1:
        return "Error"
    return chooseByFourNearest(chosen, unchosen)

def getWeatherByCity(cityName):
    temp = db.ReadyByCity(sqltemperature, "temperature", cityName)
    cloud = db.ReadyByCity(sqlcloudiness, "cloudiness", cityName)
    extrWind = db.ReadyByCity(sqlextreme_wind, "extreme_wind", cityName)
    moist = db.ReadyByCity(sqlmoisture, "moisture", cityName)
    press = db.ReadyByCity(sqlpressure, "pressure", cityName)
    wind = db.ReadyByCity(sqlwind, "wind", cityName)

    data = pd.DataFrame()
    data = convert_date2(data,temp)
    data = convert_date2(data,cloud)
    data = convert_date2(data,extrWind)
    data = convert_date2(data,moist)
    data = convert_date2(data,press)
    data = convert_date(data,wind)
    
    #data = pd.concat([temp,cloud,extrWind,moist,press,wind])
    return data
    # return temp["temp"],temp["RF"],cloud["N"],cloud["CD"],extrWind["strenght"],moist["VP_TER"], moist["E_TF_TER"], moist["TF_TER"], moist["RF_TER"],press["Pressure"],wind

def joinData(cityName, cities):
    data = getWeatherByCity(cityName)
    for i in range(len(cities)):
        newdata = getWeatherByCity(cities.iloc[i,0])
        data = pd.merge(data, newdata, on="MESS_DATUM_ZEIT", how="outer", suffixes=("", "_" + str(i)))

        #data = data.join(newdata, lsuffix='', rsuffix='{i}', how='outer').reset_index()
    data = data.fillna(0)
    return data


def gen_trainData():
    cityP = db.Read(sqlcities, "cities")
    for i in range(len(cityP)):
        cityName = cityP.iloc[i]["Cityname"]
        cities = getCities(cityP,cityName)
        train_Data = joinData(cityName, cities)
        yield train_Data

def label_nextday(df):
    df = df.drop(df.index[df["MESS_DATUM_ZEIT"].values == max(df["MESS_DATUM_ZEIT"])], axis=0)
    print(df)
    label = pd.DataFrame()
    for d in range(len(df)):
        ind = df.index[df["MESS_DATUM_ZEIT"].values == df["MESS_DATUM_ZEIT"][d]+ datetime.timedelta(days=1)]
        label = pd.concat([df.iloc[ind]])
    print(label)

if __name__ == "__main__":
    trainData = gen_trainData()
    for i in range(1):
        train = next(trainData)
        label_nextday(train)
        


