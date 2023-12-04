import database_con as db
import os
import zipfile as zpf
import pandas as pd


path = "C:/Users/khan0/Downloads"
sqlCities = "./Database/Cities.db"
sqltemperature = "./Database/Temperature.db"
sqlcloudiness = "./Database/Cloudiness.db"
sqlextreme_wind = "./Database/Extreme_wind.db"
sqlmoisture = "./Database/Moisture.db"
sqlpressure = "./Database/Pressure.db"
sqlwind = "./Database/Wind.db"

#os.chdir("./CollectData")

#f = open(sqlCities, "a")
#f.close()
db.CreateTable(sqlCities, "CREATE TABLE IF NOT EXISTS cities(Cityname Varchar(20) primary key, longitude double, latitude double, altitude double);")
db.CreateTable(sqltemperature, "CREATE TABLE IF NOT EXISTS temperature(Cityname Varchar(20), temp double, RF int, dat datetime);")
db.CreateTable(sqlcloudiness, "CREATE TABLE IF NOT EXISTS cloudiness(Cityname Varchar(20), N int, CD int, dat datetime);")
db.CreateTable(sqlextreme_wind, "CREATE TABLE IF NOT EXISTS extreme_wind(Cityname Varchar(20), strenght double, dat datetime);")
db.CreateTable(sqlmoisture, "CREATE TABLE IF NOT EXISTS moisture(Cityname Varchar(20), VP_TER double, E_TF_TER int, TF_TER double, RF_TER double, dat datetime);")
db.CreateTable(sqlpressure, "CREATE TABLE IF NOT EXISTS pressure(Cityname Varchar(20), Pressure double, dat datetime);")
db.CreateTable(sqlwind, "CREATE TABLE IF NOT EXISTS wind(Cityname Varchar(20), FK int, DK int, dat datetime);")

files = [f for f in os.listdir(path) if f.startswith("terminwerte") and f.endswith('.zip')]

cty = ""
for file_name in files:
    file_path = "/".join([path,file_name])
    with zpf.ZipFile(file_path, 'r') as zf:
        for ff in zf.namelist():
            if ff.startswith("Metadaten_Geraete_Luftdruck_Stationshoehe") and ff.endswith('.txt'): # optional filtering by filetype
                with zf.open(ff) as f:
                    df = pd.read_csv(f, header=0, sep=";", encoding="utf-8", encoding_errors='ignore',engine='python')
                    print(df.columns)
                    wdf = df[["Stationsname", "Geo. Laenge [Grad]", "Geo. Breite [Grad]", "Stationshoehe [m]"]]
                    wdf = wdf.dropna().head(1)
                    wdf = wdf.rename(columns={"Stationsname": "Cityname", "Geo. Laenge [Grad]": "longitude", "Geo. Breite [Grad]": "latitude", "Stationshoehe [m]": "altitude"})
                    cty = str(wdf["Cityname"].iloc[0])
                    db.write(sqlCities, "cities", wdf)
            if ff.startswith("produkt") and ff.endswith('.txt'):
                with zf.open(ff) as f:
                    df = pd.read_csv(f, header=0, sep=";", encoding="utf-8", encoding_errors='ignore',engine='python')
                    print(df)
                    if file_name.startswith("terminwerte_TU"):
                        wdf = df[["TT_TER", "RF_TER", "MESS_DATUM"]]
                        wdf["Cityname"] =cty
                        wdf = wdf.rename({"TT_TER": "temp", "RF_TER": "RF", "MESS_DATUM": "Dat"})
                        print(wdf)
                        db.write(sqltemperature, "temperature", wdf)
                    if file_name.startswith("terminwerte_N"):
                        wdf = df[["N_TER", "CD_TER", "MESS_DATUM"]]
                        wdf["Cityname"] =cty
                        wdf = wdf.rename({"N_TER": "N", "CD_TER": "CD", "MESS_DATUM": "Dat"})
                        print(wdf)
                        db.write(sqlcloudiness, "cloudiness", wdf)
                    if file_name.startswith("terminwerte_FX3") or file_name.startswith("terminwerte_FX6"):
                        wdf = df[["FX_911_3", "MESS_DATUM"]]
                        wdf["Cityname"] =cty
                        wdf = wdf.rename({"FX_911_3": "stregth", "MESS_DATUM": "Dat"})
                        print(wdf)
                        db.write(sqlextreme_wind, "extreme_wind", wdf)
                    if file_name.startswith("terminwerte_TF"):
                        wdf = df[["VP_TER", "E_TF_TER", "TF_TER", "RF_TER", "MESS_DATUM"]]
                        wdf["Cityname"] =cty
                        wdf = wdf.rename({"VP_TER": "VP", "E_TF_TER": "E_TF", "TF_TER": "TF", "RF_TER": "RF", "MESS_DATUM": "Dat"})
                        print(wdf)
                        db.write(sqlmoisture, "moisture", wdf)
                    if file_name.startswith("terminwerte_PP"):
                        wdf = df[["PP_TER", "MESS_DATUM"]]
                        wdf["Cityname"] =cty
                        wdf = wdf.rename({"PP_TER": "Pressure", "MESS_DATUM": "Dat"})
                        print(wdf)
                        db.write(sqlpressure, "pressure", wdf)
                    if file_name.startswith("terminwerte_FK"):
                        wdf = df[["DK_TER", "FK_TER", "MESS_DATUM"]]
                        wdf["Cityname"] =cty
                        wdf = wdf.rename({"DK_TER": "DK","FK_TER": "FK", "MESS_DATUM": "Dat"})
                        print(wdf)
                        db.write(sqlwind, "wind", wdf)
