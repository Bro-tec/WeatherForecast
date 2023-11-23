import json
import requests

Response = requests.get("http://api.openweathermap.org/data/2.5/weather?q=Engelskirchen&APPID=559f64d5bfc9ed695823e611c5f57083")

WeatherData = Response.json()
print(json.dumps(WeatherData, indent = 4, sort_keys = True))