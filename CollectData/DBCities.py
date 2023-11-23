import database_con as db



sqlCities = "./Databases/Cities.db"

db.CreateTable(sqlCities, "CREATE TABLE IF NOT EXISTS cities(Cityname Varchar(20), long double, lat double);")
