import database_con as db
import zipfile
import glob
import os
import pandas as pd

path = "C:/Users/Shazil Khan/Downloads/cities_list.xlsx"
sqltemp = "./Databases/Temperature.db"
sqlCities = "./Databases/Cities.db"

db.CreateTable(sqlCities, "CREATE TABLE IF NOT EXISTS cities(Cityname Varchar(20), long double, lat double);")
db.CreateTable(sqltemp, "CREATE TABLE IF NOT EXISTS temp(Cityname Varchar(20), long double, lat double);")

df = pd.read_excel(path)

for index, row in df.iterrows():
    print(row['name'], row['lon'], row['lat'])