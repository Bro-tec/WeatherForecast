# WeatherPredicter
A way to predict weather data



This Code focused to predict data based on weather-data in germany using DWD data.
The data gets retrieverd using Brightsky Api.

following libraries are needed:
- datetime
- asyncio
- pandas
- numpy
- math
- random
- warnings
- matplotlib
- os
- json
- tqdm
- sklearn
- torch
- torchmetrics
- requests
- aiohttp
- asyncio


## First u need to get the get stations.csv
Run the file get_DWD_data.py in the folder CollectData
or
load the data from (https://drive.google.com/drive/folders/1a8JoFlJ9xNWByvjP25ytz66WRIDUyy0E?usp=sharing) and save it into the Weatherpredicter folder


## The next step is to train the models.
Run the files trainDailyModels.py and trainHourlyModes.py in the folder WeatherPredicter
You can run the Normal mode but its recomendet to use the Timestamp mode, if the device has enough memory.
or
load the data from (https://drive.google.com/drive/folders/1a8JoFlJ9xNWByvjP25ytz66WRIDUyy0E?usp=sharing) and save it into the Models folder

## Now You are ready
Run the file predictModels,py in the folder WeatherPredicter and chose the preferences of your Model.
If you saved the Model from the google drive, then you should run the code on normal mode.
