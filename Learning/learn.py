import database_con as db
import datetime
import pandas as pd


sqlcities = "../Database/Cities.db"
sqltemperature = "../Database/Temperature.db"
sqlcloudiness = "../Database/Cloudiness.db"
sqlextreme_wind = "../Database/Extreme_wind.db"
sqlmoisture = "../Database/Moisture.db"
sqlpressure = "../Database/Pressure.db"
sqlwind = "../Database/Wind.db"


cityP = db.Read(sqlcities, "cities")

for i in range(len(cityP)):
    cityName = cityP.iloc[0]["Cityname"]
    temp = db.ReadyByCity(sqltemperature, "temperature", cityName)
    cloud = db.ReadyByCity(sqlcloudiness, "cloudiness", cityName)
    extrWind = db.ReadyByCity(sqlextreme_wind, "extreme_wind", cityName)
    moist = db.ReadyByCity(sqlmoisture, "moisture", cityName)
    press = db.ReadyByCity(sqlpressure, "pressure", cityName)
    wind = db.ReadyByCity(sqlwind, "wind", cityName)

    print(temp)
    # print(temp[0]["MESS_DATUM"])

    #date = pd.to_datetime(int(temp[:]["MESS_DATUM"]), utc=True, unit='ms')
    #print(temp['MESS_DATUM']) = 
    # for j in range(len(temp)):
    #     #print(pd.to_datetime(temp.iloc[j]['MESS_DATUM']))
    #     date = datetime.datetime.fromtimestamp(temp.iloc[j]["MESS_DATUM"] / 1e3)
    #     print(date)


