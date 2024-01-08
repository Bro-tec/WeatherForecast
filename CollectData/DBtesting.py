import db_con as db
import os
import zipfile as zpf
import pandas as pd


path = "C:/Users/Shazil Khan/Downloads"
sqlcities = "./Database/Cities.db"
sqltemperature = "./Database/Temperature.db"
sqlcloudiness = "./Database/Cloudiness.db"
sqlextreme_wind = "./Database/Extreme_wind.db"
sqlmoisture = "./Database/Moisture.db"
sqlpressure = "./Database/Pressure.db"
sqlwind = "./Database/Wind.db"

print(db.Read(sqlcities, "cities"))
print(db.Read(sqltemperature, "temperature"))
print(db.Read(sqlcloudiness, "cloudiness"))
print(db.Read(sqlextreme_wind, "extreme_wind"))
print(db.Read(sqlmoisture, "moisture"))
print(db.Read(sqlpressure, "pressure"))
wind = db.Read(sqlwind, "wind")
print(wind)