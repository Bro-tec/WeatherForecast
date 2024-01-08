import db_con as db
import os
import zipfile as zpf
import pandas as pd
import warnings


warnings.filterwarnings('ignore')


path = "C:Users/khan0/Downloads"
sqlCities = "./Database/Cities.db"
sqltemperature = "./Database/Temperature.db"
sqlcloudiness = "./Database/Cloudiness.db"
sqlextreme_wind = "./Database/Extreme_wind.db"
sqlmoisture = "./Database/Moisture.db"
sqlpressure = "./Database/Pressure.db"
sqlwind = "./Database/Wind.db"


db.CreateTable(sqlCities, "CREATE TABLE IF NOT EXISTS cities(Cityname Varchar(20) primary key, longitude double, latitude double, altitude double);")
db.CreateTable(sqltemperature, "CREATE TABLE IF NOT EXISTS temperature(dat datetime, Cityname Varchar(20), temp double, RF int, CONSTRAINT pk_Temp PRIMARY KEY (dat,Cityname));")
db.CreateTable(sqlcloudiness, "CREATE TABLE IF NOT EXISTS cloudiness(dat datetime, Cityname Varchar(20), N int, CD int, CONSTRAINT pk_Cloud PRIMARY KEY (dat,Cityname));")
db.CreateTable(sqlextreme_wind, "CREATE TABLE IF NOT EXISTS extreme_wind(dat datetime, Cityname Varchar(20), strenght double, CONSTRAINT pk_ExWind PRIMARY KEY (dat,Cityname));")
db.CreateTable(sqlmoisture, "CREATE TABLE IF NOT EXISTS moisture(dat datetime, Cityname Varchar(20), VP_TER double, E_TF_TER int, TF_TER double, RF_TER double, CONSTRAINT pk_Moist PRIMARY KEY (dat,Cityname));")
db.CreateTable(sqlpressure, "CREATE TABLE IF NOT EXISTS pressure(dat datetime, Cityname Varchar(20), Pressure double, CONSTRAINT pk_Press PRIMARY KEY (dat,Cityname));")
db.CreateTable(sqlwind, "CREATE TABLE IF NOT EXISTS wind(dat datetime, Cityname Varchar(20), FK int, DK int, CONSTRAINT pk_Wind PRIMARY KEY (dat,Cityname));")

files = [f for f in os.listdir(path) if f.startswith("terminwerte") and f.endswith('.zip')]

cty = ""
for file_name in files:
    file_path = "/".join([path,file_name]) 
    print(file_path)
    fl = file_path.split("_")
    # print(fl[2])
    # if fl[2] == "00044":
    #     continue
    # else:
    with zpf.ZipFile(file_path, 'r') as zf:
        cty = "KA"
        for ff in zf.namelist():
            if ff.startswith("Metadaten_Geographie") and ff.endswith('.txt'): # optional filtering by filetype
                with zf.open(ff) as f:
                    
                    df = pd.read_csv(f, header=0, sep=";", encoding='unicode_escape')
                    # print(df["Stationsname"][0])
                    wdf = df[["Stationsname", "Geogr.Laenge", "Geogr.Breite", "Stationshoehe"]]
                    wdf = wdf.dropna().head(1)
                    wdf = wdf.rename(columns={"Stationsname": "Cityname", "Geogr.Laenge": "longitude", "Geogr.Breite": "latitude", "Stationshoehe": "altitude"})
                    cty = str(wdf["Cityname"].iloc[0])
                    vals = db.ReadyByCity(sqlCities, "cities", cty)
                    if len(vals) < 1:
                        db.write(sqlCities, "cities", wdf)
        print(cty)
        if cty == "KA":
            cty = "KA" + str(ff)
        
        for ff in zf.namelist():
            if ff.startswith("produkt") and ff.endswith('.txt'):
                with zf.open(ff) as f:
                    df = pd.read_csv(f, header=0, sep=";", encoding="unicode_escape",engine='python')
                    #print(ff)
                    if file_name.startswith("terminwerte_TU"):
                        wdf = df[["TT_TER", "RF_TER", "MESS_DATUM"]]
                        wdf["Cityname"] =cty
                        wdf = wdf.rename({"TT_TER": "temp", "RF_TER": "RF", "MESS_DATUM": "Dat"})
                        db.write(sqltemperature, "temperature", wdf)
                    if file_name.startswith("terminwerte_N"):
                        wdf = df[["N_TER", "CD_TER", "MESS_DATUM"]]
                        wdf["Cityname"] =cty
                        wdf = wdf.rename({"N_TER": "N", "CD_TER": "CD", "MESS_DATUM": "Dat"})
                        db.write(sqlcloudiness, "cloudiness", wdf)
                    if file_name.startswith("terminwerte_FX3"):
                        wdf = df[["FX_911_3", "MESS_DATUM"]]
                        wdf["Cityname"] =cty
                        wdf = wdf.rename({"FX_911_3": "stregth", "MESS_DATUM": "Dat"})
                        db.write(sqlextreme_wind, "extreme_wind", wdf)
                    if file_name.startswith("terminwerte_FX6"):
                        wdf = df[["FX_911_6", "MESS_DATUM"]]
                        wdf["Cityname"] =cty
                        wdf = wdf.rename({"FX_911_6": "stregth", "MESS_DATUM": "Dat"})
                        db.write(sqlextreme_wind, "extreme_wind", wdf)
                    if file_name.startswith("terminwerte_TF"):
                        wdf = df[["VP_TER", "E_TF_TER", "TF_TER", "RF_TER", "MESS_DATUM"]]
                        wdf["Cityname"] =cty
                        wdf = wdf.rename({"VP_TER": "VP", "E_TF_TER": "E_TF", "TF_TER": "TF", "RF_TER": "RF", "MESS_DATUM": "Dat"})
                        db.write(sqlmoisture, "moisture", wdf)
                    if file_name.startswith("terminwerte_PP"):
                        wdf = df[["PP_TER", "MESS_DATUM"]]
                        wdf["Cityname"] =cty
                        wdf = wdf.rename({"PP_TER": "Pressure", "MESS_DATUM": "Dat"})
                        db.write(sqlpressure, "pressure", wdf)
                    if file_name.startswith("terminwerte_FK"):
                        wdf = df[["DK_TER", "FK_TER", "MESS_DATUM"]]
                        wdf["Cityname"] =cty
                        wdf = wdf.rename({"DK_TER": "DK","FK_TER": "FK", "MESS_DATUM": "Dat"})
                        db.write(sqlwind, "wind", wdf)
    os.remove(file_path)
    print(file_path)                      
