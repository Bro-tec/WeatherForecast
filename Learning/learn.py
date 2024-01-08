import db_con as db
from datetime import datetime as dt
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

def convert_date(data):
    data["MESS_DATUM_ZEIT"] = [dt.utcfromtimestamp(d) for d in data["MESS_DATUM"]]
    return data

def euclidean(chosen, unchosen):
    cols=['longitude','latitude','altitude']
    unchosen["dist"] = np.linalg.norm(chosen[cols].values - unchosen[cols].values, axis=1)
    return unchosen

def chooseByFourNearest(chosen, unchosen):
    print(unchosen)
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

        #print(temp)
        temp = convert_date(temp)
        cloud = convert_date(cloud)
        extrWind = convert_date(extrWind)
        moist = convert_date(moist)
        press = convert_date(press)
        wind = convert_date(wind)

def getWeatherByCities(cityName):
    temp = db.ReadyByCity(sqltemperature, "temperature", cityName)
    cloud = db.ReadyByCity(sqlcloudiness, "cloudiness", cityName)
    extrWind = db.ReadyByCity(sqlextreme_wind, "extreme_wind", cityName)
    moist = db.ReadyByCity(sqlmoisture, "moisture", cityName)
    press = db.ReadyByCity(sqlpressure, "pressure", cityName)
    wind = db.ReadyByCity(sqlwind, "wind", cityName)

    #print(temp)
    temp = convert_date(temp)
    cloud = convert_date(cloud)
    extrWind = convert_date(extrWind)
    moist = convert_date(moist)
    press = convert_date(press)
    wind = convert_date(wind)

def gen_trainData():
    cityP = db.Read(sqlcities, "cities")
    for i in range(len(cityP)):
        cityName = cityP.iloc[i]["Cityname"]
        cities = getCities(cityP,cityName)
        getWeatherByCities(cityName,cities)
        yield cities


if __name__ == "__main__":
    trainData = gen_trainData()
    for i in range(1):
        cities = next(trainData)


