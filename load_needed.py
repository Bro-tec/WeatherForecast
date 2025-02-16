import os
import gdown

if not os.path.isdir("Forecasts"):
    os.mkdir("Forecasts")

if not os.path.isdir("Models"):
    os.mkdir("Models")

if not os.path.isdir("Plots"):
    os.mkdir("Plots")

if not os.path.isdir("Images"):
    os.mkdir("Images")
    gdown.download(
        "https://drive.google.com/file/d/1TyIs5rla8t5mvWgJxY1JKIZEcS8mZ605/view?usp=drive_link",
        "Images/Arrow_down.png",
    )
    gdown.download("", "Images/Arrow_down_left.png")
    gdown.download("", "Images/Arrow_down_right.png")
    gdown.download("", "Images/Arrow_left.png")
    gdown.download("", "Images/Arrow_right.png")
    gdown.download("", "Images/Arrow_up.png")
    gdown.download("", "Images/Arrow_up_left.png")
    gdown.download("", "Images/Arrow_up_right.png")
    gdown.download("", "Images/clear-day.png")
    gdown.download("", "Images/clear-night.png")
    gdown.download("", "Images/cloudy.png")
    gdown.download("", "Images/dry.png")
    gdown.download("", "Images/fog.png")
    gdown.download("", "Images/glaze.png")
    gdown.download("", "Images/hail.png")
    gdown.download("", "Images/ice.png")
    gdown.download("", "Images/Landkarte_Deutschland.png")
    gdown.download("", "Images/moist.png")
    gdown.download("", "Images/None.png")
    gdown.download("", "Images/not_dry.png")
    gdown.download("", "Images/partly-cloudy-day.png")
    gdown.download("", "Images/partly-cloudy-night.png")
    gdown.download("", "Images/rain.png")
    gdown.download("", "Images/rime.png")
    gdown.download("", "Images/sleet.png")
    gdown.download("", "Images/snow.png")
    gdown.download("", "Images/thunderstorm.png")
    gdown.download("", "Images/wet.png")
    gdown.download("", "Images/wind.png")
