import database_con as db


sqlwind = "./Databases/Wind.db"
sqlCities = "./Databases/Cities.db"

db.CreateTable(sqlCities, "CREATE TABLE IF NOT EXISTS cities(Cityname Varchar(20), long double, lat double);")
db.CreateTable(sqlwind, "CREATE TABLE IF NOT EXISTS wind(Cityname Varchar(20), long double, lat double);")
