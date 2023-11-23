import database_con as db



sqlweather = "./Databases/Weather.db"

db.CreateTable(sqlweather, "CREATE TABLE IF NOT EXISTS weather(Cityname Varchar(20), long double, lat double);")

