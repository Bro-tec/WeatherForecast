import CollectData.get_learning_data as gld
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
import json
from datetime import datetime as dt
from datetime import timedelta as td
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchmetrics.classification import MulticlassConfusionMatrix
import math
import warnings


plt.switch_backend("agg")

warnings.filterwarnings("ignore")

scales = [
    1,
    1,
    1,
    10000,
    40,
    50,
    100,
    100,
    40,
    100,
    100,
    100,
    1,
]


# checking if cuda is used. if not it choses cpu
def check_cuda():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("this device uses " + device + " to train data")
    return torch.device(device)


def clear_cuda():
    torch.cuda.empty_cache()


# creating my lstm
class PyTorch_LSTM(nn.Module):
    def __init__(
        self,
        inputs,
        outputs,
        device,
        h_size=30,
        seq_size=24,
        batchsize=100,
        layer=3,
        dropout=0,
    ):
        super(PyTorch_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=inputs,
            hidden_size=h_size,
            num_layers=layer,
            dropout=dropout,
            batch_first=True,
        ).to(device)
        # self.fc2 = nn.Linear(seq_size, 1).to(device)
        self.fc = nn.Linear(h_size, outputs).to(device)
        self.hidden_state = Variable(
            torch.Tensor(np.zeros((layer, batchsize, h_size)).tolist())
            .type(torch.float32)
            .to(device)
        ).to(device)
        self.cell_state = Variable(
            torch.Tensor(np.zeros((layer, batchsize, h_size)).tolist())
            .type(torch.float32)
            .to(device)
        ).to(device)

    def show_hiddens(self):
        print(
            "hidden_state:\n",
            self.hidden_state.tolist(),
            "\ncell_state:\n",
            self.cell_state.tolist(),
        )

    def set_hiddens(self, batchsize, device):
        self.hidden_state = Variable(
            torch.Tensor(
                np.zeros(
                    (self.hidden_state.shape[0], batchsize, self.hidden_state.shape[-1])
                ).tolist()
            )
            .type(torch.float32)
            .to(device)
        ).to(device)
        self.cell_state = Variable(
            torch.Tensor(
                np.zeros(
                    (self.cell_state.shape[0], batchsize, self.cell_state.shape[-1])
                ).tolist()
            )
            .type(torch.float32)
            .to(device)
        ).to(device)

    def forward(self, input, device, hiddens=True):
        out, (hn, cn) = self.lstm(input, (self.hidden_state, self.cell_state))
        if hiddens:
            self.hidden_state = hn.detach()
            self.cell_state = cn.detach()
        # print("out", out.shape)
        l = self.fc(out[:, -1, :])
        # print("l2", l2.shape)

        return l.type(torch.float64).to(device)


def create_own_Model(
    name,
    input_count,
    output_count,
    features,
    indx,
    city_next=4,
    learning_rate=0.001,
    layer=3,
    hiddensize=7,
    sequences=12,
    dropout=0.1,
    month=True,
    hours=True,
    position=True,
):
    print(
        "\nname:",
        name,
        "\ninput_count:",
        input_count,
        "\noutput_count:",
        output_count,
        "\nfeatures:",
        features,
        "\nindx:",
        indx,
        "\ncity_next:",
        city_next,
        "\nlearning_rate:",
        learning_rate,
        "\nlayer:",
        layer,
        "\nhiddensize:",
        hiddensize,
        "\nsequences:",
        sequences,
        "\ndropout:",
        dropout,
        "\nmonth:",
        month,
        "\nhours:",
        hours,
        "\nposition:",
        position,
    )
    device = torch.device("cpu")
    if os.path.exists(f"./Models/{name}.pth"):
        print("Model does already exist")
        return "error"
    else:
        model = PyTorch_LSTM(
            input_count,
            output_count,
            device,
            h_size=hiddensize,
            seq_size=sequences,
            layer=layer,
            dropout=dropout,
            batchsize=1,
        )
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        others = {
            "model": model,
            "optimizer": optimizer.state_dict(),
            "input_count": input_count,
            "output_count": output_count,
            "features": features,
            "indx": indx,
            "city_next": city_next,
            "learning_rate": learning_rate,
            "layer": layer,
            "hiddensize": hiddensize,
            "sequences": sequences,
            "dropout": dropout,
            "month": month,
            "hours": hours,
            "position": position,
        }

        torch.save(
            others,
            f"./Models/{name}.pth",
        )

        print("Model Created")
    return "done"


def create_history(features=[]):
    history = {"accuracy": [0], "loss": [0], "val_accuracy": [0], "val_loss": [0]}
    i = 0
    if len(features) > 0:
        if (
            "precipitation" in features
            or "precipitation_probability" in features
            or "precipitation_probability_6h" in features
            or "pressure_msl" in features
            or "temperature" in features
            or "sunshine" in features
            or "wind_speed" in features
            or "cloud_cover" in features
            or "dew_point" in features
            or "wind_gust_speed" in features
            or "relative_humidity" in features
            or "visibility" in features
            or "solar" in features
        ):
            i += 1
            history[f"accuracy{i}"] = [0]
            history[f"loss{i}"] = [0]
            history[f"val_accuracy{i}"] = [0]
            history[f"val_loss{i}"] = [0]
        if "wind_direction0" in features:
            i += 1
            history[f"accuracy{i}"] = [0]
            history[f"loss{i}"] = [0]
            history[f"val_accuracy{i}"] = [0]
            history[f"val_loss{i}"] = [0]
        if "wind_gust_direction0" in features:
            i += 1
            history[f"accuracy{i}"] = [0]
            history[f"loss{i}"] = [0]
            history[f"val_accuracy{i}"] = [0]
            history[f"val_loss{i}"] = [0]
        if "icon0" in features:
            i += 1
            history[f"accuracy{i}"] = [0]
            history[f"loss{i}"] = [0]
            history[f"val_accuracy{i}"] = [0]
            history[f"val_loss{i}"] = [0]
        if "condition0" in features:
            i += 1
            history[f"accuracy{i}"] = [0]
            history[f"loss{i}"] = [0]
            history[f"val_accuracy{i}"] = [0]
            history[f"val_loss{i}"] = [0]
    return history


# loading model if already saved or creating a new model
def load_own_Model(name, device):
    history = {
        "accuracy": [0],
        "loss": [0],
        "val_accuracy": [0],
        "val_loss": [0],
    }
    model = {"haha": [1, 2, 3]}
    optimizer = {"haha": [1, 2, 3]}
    others = {"haha": [1, 2, 3]}

    if os.path.exists(f"./Models/{name}.pth"):
        checkpoint = torch.load(f"./Models/{name}.pth", map_location="cpu")

        model = checkpoint["model"]
        model.hidden_state.to(device=device)
        model.cell_state.to(device=device)
        model.lstm.to(device=device)
        model.fc.to(device=device)
        optimizer = optim.Adam(model.parameters(), lr=checkpoint["learning_rate"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        del checkpoint["model"], checkpoint["optimizer"]
        others = checkpoint
        history = create_history(checkpoint["features"])
        print("Model found")
    else:
        print("Data not found or not complete")
        return "error", "error", "error", "error"
    if os.path.exists(f"./Models/{name}_history.json"):
        with open(f"./Models/{name}_history.json", "r") as f:
            history = json.load(f)

    model.train()
    return model, optimizer, history, others


# saving model if saving_mode set to ts or timestamp it will use the number for the model to save it.
# ts helps to choose saved model data before the model started to overfit or not work anymore
def save_own_Model(name, history, model, optimizer, device):
    with open(f"./Models/{name}_history.json", "w") as fp:
        json.dump(history, fp)
    checkpoint = torch.load(f"./Models/{name}.pth", map_location="cpu")
    checkpoint.update({"model": model, "optimizer": optimizer.state_dict()})
    torch.save(
        checkpoint,
        f"./Models/{name}.pth",
    )

    print("Saved model")
    del checkpoint


def cropping(image, x_list, y_list):
    max_x = max(x_list) + 100
    min_x = min(x_list) - 100
    max_y = max(y_list) + 100
    min_y = min(y_list) - 100
    return image.crop((min_x, min_y, max_x, max_y))


def points(image, drw, df, ims, ofs, checks=[], features=[]):
    pos = []
    for i, d in df.iterrows():
        if i == 0:
            if "Time" in d.keys():
                font = ImageFont.truetype("arial.ttf", 20)
                drw.text((550, 15), str(d.Time) + " Uhr", (0, 0, 0), font=font)
            if not (checks[15] or checks[15] == "true") and "icon" in d.keys():
                pos.append([-20,-20])
            if not (checks[16] or checks[16] == "true") and "condition" in d.keys():
                pos.append([0,-20])
            if not (checks[13] or checks[13] == "true") and "g_direction" in d.keys():
                pos.append([10,10])
            if not (checks[14] or checks[14] == "true") and "direction" in d.keys():
                pos.append([-10,10])
            pos = pos + [[-20,10],[20,10],[-30,10],[30,10],[-10,20],[-30, -30],[-20, -30],[-10, -30],[0, -30],[10, -30],[20, -30],[30, -30], [-30, -20], [-30, -10], [30, -20], [30, -10], [-30, 0], [30, 0]]
            
        xy=0
        if checks[19] or checks[19] == "true":
            drw.ellipse(
                xy=(d.lon - 3, d.lat - 3, d.lon + 3, d.lat + 3),
                fill="red",
            )
        # [
        #         c_precipitation.value,
        #         c_precipitation_probability.value,
        #         c_precipitation_probability_6h.value,
        #         c_pressure_msl.value,
        #         c_temperature.value,
        #         c_sunshine.value,
        #         c_wind_speed.value,
        #         c_cloud_cover.value,
        #         c_dew_point.value,
        #         c_wind_gust_speed.value,
        #         c_relative_humidity.value,
        #         c_visibility.value,
        #         c_solar.value,
        #         c_wind_direction.value,
        #         c_wind_gust_direction.value,
        #         c_icon.value,
        #         c_condition.value,
        #         c_months.value,
        #         c_hours.value,
        #         c_pos.value,
        #     ]
        # if (checks[0] or checks[0] == "true") and "precipitation" in d.keys():
        
        
        font1 = ImageFont.truetype("arial.ttf", 10)
        
        if "precipitation" in d.keys():
            if d.precipitation == None:
                d.precipitation = "-"
            drw.text((int(d.lon) + pos[xy][0], int(d.lat) + pos[xy][1]), str(round(d.precipitation, 1)) +" %", (0, 255, 0), font=font1)
            xy +=1
        if "precipitation_probability" in d.keys():
            if d.precipitation_probability == None:
                d.precipitation_probability = "-"
            drw.text((int(d.lon) + pos[xy][0], int(d.lat) + pos[xy][1]), str(round(d.precipitation_probability, 1)) +" %", (0, 255, 255), font=font1)
            xy +=1
        if "precipitation_probability_6h" in d.keys():
            if d.precipitation_probability_6h == None:
                d.precipitation_probability_6h = "-"
            drw.text((int(d.lon) + pos[xy][0], int(d.lat) + pos[xy][1]), str(round(d.precipitation_probability_6h, 1)) +" %", (255, 192, 203), font=font1)
            xy +=1
        if "pressure_msl" in d.keys():
            if d.pressure_msl == None:
                d.pressure_msl = "-"
            drw.text((int(d.lon) + pos[xy][0], int(d.lat) + pos[xy][1]), str(round(d.pressure_msl, 1)) +" %", (0, 0, 255), font=font1)
            xy +=1
        if "temperature" in d.keys():
            if d.temperature == None:
                d.temperature = "-"
            drw.text((int(d.lon) + pos[xy][0], int(d.lat) + pos[xy][1]), str(round(d.temperature, 1)) +" %", (255, 165, 0), font=font1)
            xy +=1
        if "sunshine" in d.keys():
            if d.sunshine == None:
                d.sunshine = "-"
            drw.text((int(d.lon) + pos[xy][0], int(d.lat) + pos[xy][1]), str(round(d.sunshine, 1)) +" %", (255, 255, 0), font=font1)
            xy +=1
        if "wind_speed" in d.keys():
            if d.wind_speed == None:
                d.wind_speed = "-"
            drw.text((int(d.lon) + pos[xy][0], int(d.lat) + pos[xy][1]), str(round(d.wind_speed, 1)) +" km/h", (64, 224, 208), font=font1)
            xy +=1
        if "cloud_cover" in d.keys():
            if d.cloud_cover == None:
                d.cloud_cover = "-"
            drw.text((int(d.lon) + pos[xy][0], int(d.lat) + pos[xy][1]), str(round(d.cloud_cover, 1)) +" %", (173, 216, 230), font=font1)
            xy +=1
        if "dew_point" in d.keys():
            if d.dew_point == None:
                d.dew_point = "-"
            drw.text((int(d.lon) + pos[xy][0], int(d.lat) + pos[xy][1]), str(round(d.dew_point, 1)) +" %",  (165, 42, 42), font=font1)
            xy +=1
        if "wind_gust_speed" in d.keys():
            if d.wind_gust_speed == None:
                d.wind_gust_speed = "-"
            drw.text((int(d.lon) + pos[xy][0], int(d.lat) + pos[xy][1]), str(round(d.wind_gust_speed, 1)) +" km/h", (255, 127, 80), font=font1)
            xy +=1
        if "relative_humidity" in d.keys():
            if d.relative_humidity == None:
                d.relative_humidity = "-"
            drw.text((int(d.lon) + pos[xy][0], int(d.lat) + pos[xy][1]), str(round(d.relative_humidity, 1)) +" %", (250, 128, 114), font=font1)
            xy +=1
        if "visibility" in d.keys():
            if d.visibility == None:
                d.visibility = "-"
            drw.text((int(d.lon) + pos[xy][0], int(d.lat) + pos[xy][1]), str(round(d.visibility, 1)) +" %", (128, 128, 0), font=font1)
            xy +=1
        if "solar" in d.keys():
            if d.solar == None:
                d.solar = "-"
            drw.text((int(d.lon) + pos[xy][0], int(d.lat) + pos[xy][1]), str(round(d.solar, 1)) +" %", (0, 100, 0), font=font1)
            xy +=1
        if "icon" in d.keys():
            if d.icon == None:
                d.icon = "None"
            ix = ofs.index(d.icon)
            image.paste(ims[ix], (int(d.lon) - 20, int(d.lat) - 20), mask=ims[ix])
        if "condition" in d.keys():
            if d.condition == None:
                d.condition = "None"
            ix = ofs.index(d.condition)
            image.paste(ims[ix], (int(d.lon), int(d.lat) - 20), mask=ims[ix])
        if "g_direction" in d.keys():
            if d.g_direction == None:
                d.g_direction = "None"
            ix = ofs.index(d.g_direction)
            image.paste(ims[ix], (int(d.lon) + 10, int(d.lat) + 10), mask=ims[ix])
        if "direction" in d.keys():
            if d.direction == None:
                d.direction = "None"
            ix = ofs.index(d.direction)
            image.paste(ims[ix], (int(d.lon) - 10, int(d.lat) + 10), mask=ims[ix])
    return image


def l_to_px(list, rs, ps):
    min_l = min(list)
    max_l = max(list) - min_l
    print(min_l, max_l)
    print(list[0] - min_l)
    print((list[0] - min_l) / max_l)
    print(rs)
    list = [((x - min_l) / max_l * rs) + ps for x in list]
    return list


def load_images():
    mypath = "./Images"
    onlyfiles = [
        f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))
    ]
    ims = [
        Image.open(mypath + "/" + of).convert("RGBA").resize((20, 20))
        for of in onlyfiles
    ]
    onlyfiles = [of.replace(".png", "") for of in onlyfiles]
    return ims, onlyfiles


def show_image(
    outs, hours=1, name="germany_points", name_ending=".png", checks=[], features=[]
):
    amount = int(len(outs) / hours)
    for h in range(hours):
        im = Image.open("Images/Landkarte_Deutschland.png").convert("RGBA")
        stations = gld.load_stations_csv()
        stations["lat"] = [-x for x in stations["lat"].to_list()]
        stations["lon"] = l_to_px(stations["lon"].to_list(), 648, 50)
        stations["lat"] = l_to_px(stations["lat"].to_list(), 904, 60)
        outs_stations = pd.merge(
            outs[amount * h : amount * (h + 1)],
            stations,
            on=["ID"],
            suffixes=("_x", ""),
        )
        ims, ofs = load_images()
        drw = ImageDraw.Draw(im)

        im = points(im, drw, outs_stations, ims, ofs, checks=checks, features=features)
        im.save(f"Forecasts/{name}_{h}{name_ending}", "PNG")
        
def mock_show_image(
    cities, name="germany_points", name_ending=".png", checks=[], features=[], i=0
):
    if len(checks) ==0:
        checks = [False for i in range(20)]
        checks[19] = True
    im = Image.open("Images/Landkarte_Deutschland.png").convert("RGBA")
    stations = gld.load_stations_csv()
    outs_stations = gld.load_stations_by_IDs(cities)
    stations["lat"] = [-x for x in stations["lat"].to_list()]
    stations["lon"] = l_to_px(stations["lon"].to_list(), 648, 50)
    stations["lat"] = l_to_px(stations["lat"].to_list(), 904, 60)
    outs_stations = pd.merge(
        outs_stations,
        stations,
        on=["ID"],
        suffixes=("_x", ""),
    )
    ims, ofs = load_images()
    drw = ImageDraw.Draw(im)

    im = points(im, drw, outs_stations, ims, ofs, checks=checks, features=features)
    im.save(f"Forecasts/mock_{name}{i}{name_ending}", "PNG")



def approximation(all):
    x = np.arange(0, len(all))
    y = list(all)
    [b, m] = np.polynomial.polynomial.polyfit(x, y, 1)
    return b, m


# plotting Evaluation via Accuracy, Loss and MulticlassConfusionMatrix
def plotting_hist(history, metrics, name, min_amount=2, epoche=0):
    icons = [
        None,
        "clear-day",
        "clear-night",
        "partly-cloudy-day",
        "partly-cloudy-night",
        "cloudy",
        "fog",
        "wind",
        "rain",
        "sleet",
        "snow",
        "hail",
        "thunderstorm",
        "dry",
        "moist",
        "wet",
        "rime",
        "ice",
        "glaze",
        "not dry",
        "reserved",
    ]
    directions = [
        "Arrow_up",
        "Arrow_up_right",
        "Arrow_right",
        "Arrow_down_right",
        "Arrow_down",
        "Arrow_left_down",
        "Arrow_left",
        "Arrow_up_left",
        "None",
    ]
    name_tag = f"{name}_plot"
    p = 1
    plots = 2
    if "accuracy1" in history:
        plots += 1
    if "accuracy2" in history:
        plots += 1
    if "accuracy3" in history:
        plots += 1
    if "accuracy4" in history:
        plots += 1
    if "accuracy5" in history:
        plots += 1
    # summarize history for accuracy
    fig, ax = plt.subplots(plots, figsize=(20, 12), sharey="row")
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.title("model accuracy")
    # ax[0].plot(history["accuracy"][1:])
    # ax[0].plot(history["val_accuracy"][1:])
    if "accuracy1" in history:
        ax[0].plot(history["accuracy1"][1:])
        ax[0].plot(history["val_accuracy1"][1:])
    if "accuracy2" in history:
        ax[0].plot(history["accuracy2"][1:])
        ax[0].plot(history["val_accuracy2"][1:])
    if "accuracy3" in history:
        ax[0].plot(history["accuracy3"][1:])
        ax[0].plot(history["val_accuracy3"][1:])
    if "accuracy4" in history:
        ax[0].plot(history["accuracy4"][1:])
        ax[0].plot(history["val_accuracy4"][1:])
    if "accuracy5" in history:
        ax[0].plot(history["accuracy5"][1:])
        ax[0].plot(history["val_accuracy5"][1:])
    ax[1].plot(history["accuracy"][1:])
    ax[1].plot(history["val_accuracy"][1:])
    if "accuracy1" in history:
        p += 1
        ax[p].plot(history["accuracy1"][1:])
        ax[p].plot(history["val_accuracy1"][1:])
        ax[p].grid(axis="y")
    if "accuracy2" in history:
        p += 1
        ax[p].plot(history["accuracy2"][1:])
        ax[p].plot(history["val_accuracy2"][1:])
        ax[p].grid(axis="y")
    if "accuracy3" in history:
        p += 1
        ax[p].plot(history["accuracy3"][1:])
        ax[p].plot(history["val_accuracy3"][1:])
        ax[p].grid(axis="y")
    if "accuracy4" in history:
        p += 1
        ax[p].plot(history["accuracy4"][1:])
        ax[p].plot(history["val_accuracy4"][1:])
        ax[p].grid(axis="y")
    if "accuracy5" in history:
        p += 1
        ax[p].plot(history["accuracy5"][1:])
        ax[p].plot(history["val_accuracy5"][1:])
        ax[p].grid(axis="y")
    ax[0].grid(axis="y")
    ax[1].grid(axis="y")
    ax[0].legend(
        [
            "train",
            "test",
        ],
        loc="upper left",
    )
    ax[1].legend(
        [
            "train",
            "test",
        ],
        loc="upper left",
    )
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(f"./Plots/{name_tag}_accuracy.png")
    plt.close()

    # summarize history for loss
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")

    history["loss"] = [
        -_ if _ < 0 else _ for _ in (pd.Series(history["loss"])).tolist()
    ]
    history["val_loss"] = [
        -_ if _ < 0 else _ for _ in (pd.Series(history["val_loss"])).tolist()
    ]
    if "loss1" in history:
        history["loss1"] = [
            -_ if _ < 0 else _ for _ in (pd.Series(history["loss1"])).tolist()
        ]
        history["val_loss1"] = [
            -_ if _ < 0 else _ for _ in (pd.Series(history["val_loss1"])).tolist()
        ]
    if "loss2" in history:
        history["loss2"] = [
            -_ if _ < 0 else _ for _ in (pd.Series(history["loss2"])).tolist()
        ]
        history["val_loss2"] = [
            -_ if _ < 0 else _ for _ in (pd.Series(history["val_loss2"])).tolist()
        ]
    if "loss3" in history:
        history["loss3"] = [
            -_ if _ < 0 else _ for _ in (pd.Series(history["loss3"])).tolist()
        ]
        history["val_loss3"] = [
            -_ if _ < 0 else _ for _ in (pd.Series(history["val_loss3"])).tolist()
        ]
    if "loss4" in history:
        history["loss4"] = [
            -_ if _ < 0 else _ for _ in (pd.Series(history["loss4"])).tolist()
        ]
        history["val_loss4"] = [
            -_ if _ < 0 else _ for _ in (pd.Series(history["val_loss4"])).tolist()
        ]
    if "loss5" in history:
        history["loss5"] = [
            -_ if _ < 0 else _ for _ in (pd.Series(history["loss5"])).tolist()
        ]
        history["val_loss5"] = [
            -_ if _ < 0 else _ for _ in (pd.Series(history["val_loss5"])).tolist()
        ]

    b, m = approximation(history["loss"][1:])
    f = [b + (m * x) for x in range(1, len(history["loss"]))]
    b1, m1, b2, m2, b3, m3, b4, m4, b5, m5 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    f1, f2, f3, f4, f5 = [], [], [], [], []
    if "loss1" in history:
        b1, m1 = approximation(history["loss1"][1:])
        f1 = [b1 + (m1 * x) for x in range(1, len(history["loss1"]))]
    if "loss2" in history:
        b2, m2 = approximation(history["loss2"][1:])
        f2 = [b2 + (m2 * x) for x in range(1, len(history["loss2"]))]
    if "loss3" in history:
        b3, m3 = approximation(history["loss3"][1:])
        f3 = [b3 + (m3 * x) for x in range(1, len(history["loss3"]))]
    if "loss4" in history:
        b4, m4 = approximation(history["loss4"][1:])
        f4 = [b4 + (m4 * x) for x in range(1, len(history["loss4"]))]
    if "loss5" in history:
        b5, m5 = approximation(history["loss5"][1:])
        f5 = [b5 + (m5 * x) for x in range(1, len(history["loss4"]))]

    p = 1
    # summarize history for loss
    fig, ax = plt.subplots(plots, figsize=(20, 12), sharey="row")

    if "loss1" in history:
        ax[0].plot(history["loss1"][1:])
        ax[0].plot(history["val_loss1"][1:])
    if "loss2" in history:
        ax[0].plot(history["loss2"][1:])
        ax[0].plot(history["val_loss2"][1:])
    if "loss3" in history:
        ax[0].plot(history["loss3"][1:])
        ax[0].plot(history["val_loss3"][1:])
    if "loss4" in history:
        ax[0].plot(history["loss4"][1:])
        ax[0].plot(history["val_loss4"][1:])
    if "loss5" in history:
        ax[0].plot(history["loss5"][1:])
        ax[0].plot(history["val_loss5"][1:])
    ax[0].grid(axis="y")
    ax[1].plot(history["loss"][1:], alpha=0.8)
    ax[1].plot(history["val_loss"][1:], alpha=0.75)
    ax[1].plot(f)
    ax[1].grid(axis="y")
    ax[1].legend(
        [
            "train / Epoche:" + str(epoche + 1),
            "test",
            "linear train: {:.5f}x".format(m),
        ],
        loc="upper left",
    )
    if "loss1" in history:
        p += 1
        ax[p].plot(history["loss1"][1:], alpha=0.8)
        ax[p].plot(history["val_loss1"][1:], alpha=0.75)
        ax[p].plot(f1)
        ax[p].grid(axis="y")
        ax[p].legend(
            [
                "train / Epoche:" + str(epoche + 1),
                "test",
                "linear train: {:.5f}x".format(m1),
            ],
            loc="upper left",
        )
    if "loss2" in history:
        p += 1
        ax[p].plot(history["loss2"][1:], alpha=0.8)
        ax[p].plot(history["val_loss2"][1:], alpha=0.75)
        ax[p].plot(f2)
        ax[p].grid(axis="y")
        ax[p].legend(
            [
                "train / Epoche:" + str(epoche + 1),
                "test",
                "linear train: {:.5f}x".format(m2),
            ],
            loc="upper left",
        )
    if "loss3" in history:
        p += 1
        ax[p].plot(history["loss3"][1:], alpha=0.8)
        ax[p].plot(history["val_loss3"][1:], alpha=0.75)
        ax[p].plot(f3)
        ax[p].grid(axis="y")
        ax[p].legend(
            [
                "train / Epoche:" + str(epoche + 1),
                "test",
                "linear train: {:.5f}x".format(m3),
            ],
            loc="upper left",
        )
    if "loss4" in history:
        p += 1
        ax[p].plot(history["loss4"][1:], alpha=0.8)
        ax[p].plot(history["val_loss4"][1:], alpha=0.75)
        ax[p].plot(f4)
        ax[p].grid(axis="y")
        ax[p].legend(
            [
                "train / Epoche:" + str(epoche + 1),
                "test",
                "linear train: {:.5f}x".format(m4),
            ],
            loc="upper left",
        )
    if "loss5" in history:
        p += 1
        ax[p].plot(history["loss4"][1:], alpha=0.8)
        ax[p].plot(history["val_loss4"][1:], alpha=0.75)
        ax[p].plot(f5)
        ax[p].grid(axis="y")
        ax[p].legend(
            [
                "train / Epoche:" + str(epoche + 1),
                "test",
                "linear train: {:.5f}x".format(m5),
            ],
            loc="upper left",
        )
    plt.savefig(f"./Plots/{name_tag}_loss.png")
    plt.close()

    # summarize MulticlassConfusionMatrix
    fig, ax = plt.subplots(
        4,
        2,
        figsize=(8, 16),
        sharey="row",
    )
    plt.title("Confusion Matrix")
    metric_names = [
        "Train wind direction",
        "Test wind direction",
        "Train gust direction",
        "Test gust direction",
        "Train Icon",
        "Test Icon",
        "Train Condition",
        "Test Condition",
    ]

    for i in range(0, 4):
        j = math.floor(i / 2)
        k = i % 2
        ax[j, k].title.set_text(metric_names[i])
        metrics[i].plot(
            ax=ax[0, 0],
        )
        ax[j, k].xaxis.set_ticklabels(directions)
        ax[j, k].yaxis.set_ticklabels(directions)
    for i in range(4, 8):
        j = math.floor(i / 2)
        k = i % 2
        ax[j, k].title.set_text(metric_names[i])
        metrics[i].plot(
            ax=ax[j, k],
        )
        ax[j, k].xaxis.set_ticklabels(icons)
        ax[j, k].yaxis.set_ticklabels(icons)
    fig.set_figwidth(30)
    fig.set_figheight(30)
    plt.savefig(f"./Plots/{name_tag}_matrix.png")
    plt.close()
    
    del metrics


def loss_indexer(indx, lnf):
    if len(indx) != 0 or lnf != 0:
        val1, val2, val3, val4, val5 = False, False, False, False, False
        if True in [True for i in range(13) if indx[i] != 100]:
            val1 = True
        if indx[13] != 100:
            val2 = True
        if indx[14] != 100:
            val3 = True
        if indx[15] != 100:
            val4 = True
        if indx[16] != 100:
            val5 = True
        val_b_list = [val1, val2, val3, val4, val5]
        ix = [
            0,
            1,
            indx[13],
            indx[13],
            indx[14],
            indx[14],
            indx[15],
            indx[15],
            indx[16],
            indx[16],
        ]
        remember = -1
        for i in range(0, len(val_b_list)):
            if val_b_list[i]:
                if remember > 0:
                    ix[remember] = ix[i * 2]
                remember = (i * 2) + 1
            else:
                ix[i * 2] = 0
                ix[(i * 2) + 1] = 1
        ix[remember] = lnf
        return ix, val1, val2, val3, val4, val5
    return 0, 0, 0, 0, 0, 0


# unscaling the output because it usually doesnt get over 1
def unscale_output(output, ids, times, continueing=False, features=[], indx=[]):
    icons = [
        None,
        "clear-day",
        "clear-night",
        "partly-cloudy-day",
        "partly-cloudy-night",
        "cloudy",
        "fog",
        "wind",
        "rain",
        "sleet",
        "snow",
        "hail",
        "thunderstorm",
        "dry",
        "moist",
        "wet",
        "rime",
        "ice",
        "glaze",
        "not dry",
        "reserved",
    ]
    directions = [
        "Arrow_up",
        "Arrow_up_right",
        "Arrow_right",
        "Arrow_down_right",
        "Arrow_down",
        "Arrow_down_left",
        "Arrow_left",
        "Arrow_up_left",
        "None",
    ]
    output[:, 0] *= 100
    print(output[:, 1])
    global scales
    for idx in range(len(indx) - 4):
        if indx[idx] != 100:
            output[:, indx[idx]] *= scales[idx]

    ix, val1, val2, val3, val4, val5 = loss_indexer(indx, len(features))

    inds_o_direction = torch.argmax(output[:, ix[2] : ix[3]], dim=1)
    inds_o_g_direction = torch.argmax(output[:, ix[4] : ix[5]], dim=1)
    inds_o_icon = torch.argmax(output[:, ix[6] : ix[7]], dim=1)
    print("ix[8] : ix[9]: ", ix[8], ix[9])
    inds_o_condition = torch.argmax(output[:, ix[8] : ix[9]], dim=1)
    direction = [directions[i] for i in inds_o_direction]
    g_direction = [directions[i] for i in inds_o_g_direction]
    icon = [icons[i] for i in inds_o_icon]
    condition = [icons[i] for i in inds_o_condition]

    outs = pd.DataFrame(
        {
            "direction": direction,
            "g_direction": g_direction,
            "icon": icon,
            "condition": condition,
        }
    )
    for i in range(ix[0], ix[1]):
        print(output[:, i].detach().clone())
        outs[features[i]] = output[:, i].detach().clone().cpu().numpy()
    if continueing:
        outs["ID"] = ids
        outs["Time"] = times
        return outs

    return outs


# scaling label to check if output was good enough
def scale_label(output, indx):
    global scales
    for idx in range(len(indx) - 4):
        if indx[idx] != 100:
            output[:, indx[idx]] /= scales[idx]
    return output


def scale_features(output, indx, cities_next):
    global scales
    for i in range(cities_next):
        for idx in range(len(indx) - 4):
            if indx[idx] != 100:
                output[:, :, indx[idx] + (i * cities_next)] /= scales[idx]
    return output


def clean_torch(val):
    val[val != val] = 0.0
    return val


# whole training of the LSTM features and labels need to be 2d shaped
def train_LSTM(
    name,
    feature,
    label,
    model,
    optimizer,
    indx,
    history,
    device,
    epoch_count=1,
    epoch=0,
    batchsize=0,
    cities_next=5,
):
    print("feature: ", feature.shape)
    print("label: ", label.shape)

    ix, val1, val2, val3, val4, val5 = loss_indexer(indx, len(feature))

    MSEloss_fn = nn.MSELoss().to(device)
    CE1loss_fn = nn.CrossEntropyLoss().to(device)
    CE2loss_fn = nn.CrossEntropyLoss().to(device)
    CE3loss_fn = nn.CrossEntropyLoss().to(device)
    CE4loss_fn = nn.CrossEntropyLoss().to(device)
    print(batchsize)
    if batchsize == 0:
        batchsize == len(feature)

    X_train, X_test, y_train, y_test = train_test_split(
        feature, label, test_size=0.05, random_state=42
    )
    X_train_tensors = Variable(torch.Tensor(X_train).to(device)).to(device)
    X_test_tensors = Variable(torch.Tensor(X_test).to(device)).to(device)
    y_train_tensors = Variable(torch.Tensor(y_train)).to(device)
    y_test_tensors = Variable(torch.Tensor(y_test)).to(device)

    X_train_tensors = scale_features(X_train_tensors, indx, cities_next)
    y_train_tensors = scale_label(y_train_tensors, indx)
    X_test_tensors = scale_features(X_test_tensors, indx, cities_next)
    y_test_tensors = scale_label(y_test_tensors, indx)
    metrics = [
        MulticlassConfusionMatrix(num_classes=9).to(device) for i in range(4)
    ] + [MulticlassConfusionMatrix(num_classes=21).to(device) for i in range(4)]

    
    loss_list = []
    acc_list1 = []
    loss_list1 = []
    acc_list2 = []
    loss_list2 = []
    acc_list3 = []
    loss_list3 = []
    acc_list4 = []
    loss_list4 = []
    acc_list5 = []
    loss_list5 = []
    labels = Variable(torch.Tensor(np.array(y_train.tolist())).to(device)).to(device)
    val_labels = Variable(torch.Tensor(np.array(y_test.tolist())).to(device)).to(device)
    val_loss_list = []
    val_loss_list1 = []
    val_acc_list1 = []
    val_loss_list2 = []
    val_acc_list2 = []
    val_loss_list3 = []
    val_acc_list3 = []
    val_loss_list4 = []
    val_acc_list4 = []
    val_loss_list5 = []
    val_acc_list5 = []

    # trains the value using each input and label
    for batches in tqdm(range(math.ceil(X_train_tensors.shape[0] / batchsize))):
        model.train()
        output = 0
        scaled_batch = 0
        # print("batches", batches)

        if batches >= math.ceil(X_train_tensors.shape[0] / batchsize) - 1:
            model.set_hiddens(X_train_tensors[batches * batchsize :].shape[0], device)
            # print("runs")
            output = model.forward(
                X_train_tensors[batches * batchsize :], device, hiddens=False
            )
            scaled_batch = y_train_tensors[batches * batchsize :].detach().clone()
            # train_label = labels[batches * batchsize :]
        else:
            model.set_hiddens(batchsize, device)
            output = model.forward(
                X_train_tensors[batches * batchsize : (batches + 1) * batchsize],
                device,
                hiddens=False,
            )
            scaled_batch = (
                y_train_tensors[batches * batchsize : (batches + 1) * batchsize]
                .detach()
                .clone()
            )
            # train_label = labels[batches * batchsize : (batches + 1) * batchsize]

        torch_outputs = output.detach().clone()
        # print("output", output.shape, "\nscaled_batch", scaled_batch.shape)
        loss1 = MSEloss_fn(
            output[:, ix[0] : ix[1]], scaled_batch[:, ix[0] : ix[1]].double()
        )
        loss2 = CE1loss_fn(
            output[:, ix[2] : ix[3]],
            torch.argmax(scaled_batch[:, ix[2] : ix[3]], dim=1),
        )
        loss3 = CE2loss_fn(
            output[:, ix[4] : ix[5]],
            torch.argmax(scaled_batch[:, ix[4] : ix[5]], dim=1),
        )
        loss4 = CE3loss_fn(
            output[:, ix[6] : ix[7]],
            torch.argmax(scaled_batch[:, ix[6] : ix[7]], dim=1),
        )
        loss5 = CE4loss_fn(
            output[:, ix[8] : ix[9]],
            torch.argmax(scaled_batch[:, ix[8] : ix[9]], dim=1),
        )
        # calculates the loss of the loss functions
        loss_l = []
        if val1:
            loss_l.append(loss1)
        if val2:
            loss_l.append(loss2)
        if val3:
            loss_l.append(loss3)
        if val4:
            loss_l.append(loss4)
        if val5:
            loss_l.append(loss5)

        loss = sum(loss_l)

        loss.backward()
        # improve from loss, this is the actual backpropergation
        optimizer.step()
        # caluclate the gradient, manually setting to 0
        optimizer.zero_grad()

        loss_list.append(float(loss.item()))
        loss_list1.append(float(loss1.item()))
        loss_list2.append(float(loss2.item()))
        loss_list3.append(float(loss3.item()))
        loss_list4.append(float(loss4.item()))
        loss_list5.append(float(loss5.item()))

        size = torch_outputs.shape[0]

        compare = (
            (
                torch.pow(
                    torch_outputs[:, ix[0] : ix[1]] - scaled_batch[:, ix[0] : ix[1]], 2
                )
            )
            .float()
            .sum()
        )
        if compare <= 0:
            return 100
        else:
            acc_list1.append(100 / (1 + (compare / size * 3)))
        inds_o_direction = torch.argmax(torch_outputs[:, ix[2] : ix[3]], dim=1)
        inds_s_direction = torch.argmax(scaled_batch[:, ix[2] : ix[3]], dim=1)
        inds_o_g_direction = torch.argmax(torch_outputs[:, ix[4] : ix[5]], dim=1)
        inds_s_g_direction = torch.argmax(scaled_batch[:, ix[4] : ix[5]], dim=1)
        inds_o_icon = torch.argmax(torch_outputs[:, ix[6] : ix[7]], dim=1)
        inds_s_icon = torch.argmax(scaled_batch[:, ix[6] : ix[7]], dim=1)
        inds_o_condition = torch.argmax(torch_outputs[:, ix[8] : ix[9]], dim=1)
        inds_s_condition = torch.argmax(scaled_batch[:, ix[8] : ix[9]], dim=1)
        acc_list2.append(
            (inds_o_direction == inds_s_direction).sum().item() / size * 100
        )
        acc_list3.append(
            (inds_o_g_direction == inds_s_g_direction).sum().item() / size * 100
        )
        acc_list4.append((inds_o_icon == inds_s_icon).sum().item() / size * 100)
        acc_list5.append(
            (inds_o_condition == inds_s_condition).sum().item() / size * 100
        )
        metrics[0].update(
            inds_o_direction,
            inds_s_direction,
        )
        metrics[2].update(
            inds_o_g_direction,
            inds_s_g_direction,
        )
        metrics[4].update(
            inds_o_icon,
            inds_s_icon,
        )
        metrics[6].update(
            inds_o_condition,
            inds_s_condition,
        )

    acc_list = []
    if val1:
        acc_list = acc_list + acc_list1
    if val2:
        acc_list = acc_list + acc_list2
    if val3:
        acc_list = acc_list + acc_list3
    if val4:
        acc_list = acc_list + acc_list4
    if val5:
        acc_list = acc_list + acc_list5
    my_acc = torch.FloatTensor(acc_list).to(device)
    my_acc = clean_torch(my_acc).mean()
    my_acc1 = torch.FloatTensor(acc_list1).to(device)
    my_acc1 = clean_torch(my_acc1).mean()
    my_acc2 = torch.FloatTensor(acc_list2).to(device)
    my_acc2 = clean_torch(my_acc2).mean()
    my_acc3 = torch.FloatTensor(acc_list3).to(device)
    my_acc3 = clean_torch(my_acc3).mean()
    my_acc4 = torch.FloatTensor(acc_list4).to(device)
    my_acc4 = clean_torch(my_acc4).mean()
    my_acc5 = torch.FloatTensor(acc_list5).to(device)
    my_acc5 = clean_torch(my_acc5).mean()
    my_loss = torch.FloatTensor(loss_list).to(device)
    my_loss = clean_torch(my_loss).mean()
    my_loss1 = torch.FloatTensor(loss_list1).to(device)
    my_loss1 = clean_torch(my_loss1).mean()
    my_loss2 = torch.FloatTensor(loss_list2).to(device)
    my_loss2 = clean_torch(my_loss2).mean()
    my_loss3 = torch.FloatTensor(loss_list3).to(device)
    my_loss3 = clean_torch(my_loss3).mean()
    my_loss4 = torch.FloatTensor(loss_list4).to(device)
    my_loss4 = clean_torch(my_loss4).mean()
    my_loss5 = torch.FloatTensor(loss_list5).to(device)
    my_loss5 = clean_torch(my_loss5).mean()
        
    # showing training
    history_loss_i = 0
    if str(my_loss).lower() != "nan":
        history["loss"].append(float(my_loss))
    else:
        history["loss"].append(history["loss"][-1])

    if val1:
        history_loss_i += 1
        if str(my_loss1).lower() != "nan":
            history[f"loss{history_loss_i}"].append(float(my_loss1))
        else:
            history[f"loss{history_loss_i}"].append(
                history[f"loss{history_loss_i}"][-1]
            )

    if val2:
        history_loss_i += 1
        if str(my_loss2).lower() != "nan":
            history[f"loss{history_loss_i}"].append(float(my_loss2))
        else:
            history[f"loss{history_loss_i}"].append(
                history[f"loss{history_loss_i}"][-1]
            )

    if val3:
        history_loss_i += 1
        if str(my_loss3).lower() != "nan":
            history[f"loss{history_loss_i}"].append(float(my_loss3))
        else:
            history[f"loss{history_loss_i}"].append(
                history[f"loss{history_loss_i}"][-1]
            )

    if val4:
        history_loss_i += 1
        if str(my_loss4).lower() != "nan":
            history[f"loss{history_loss_i}"].append(float(my_loss4))
        else:
            history[f"loss{history_loss_i}"].append(
                history[f"loss{history_loss_i}"][-1]
            )

    if val5:
        history_loss_i += 1
        if str(my_loss5).lower() != "nan":
            history[f"loss{history_loss_i}"].append(float(my_loss5))
        else:
            history[f"loss{history_loss_i}"].append(
                history[f"loss{history_loss_i}"][-1]
            )

    history_acc_i = 0
    if str(my_acc).lower() != "nan":
        history["accuracy"].append(float(my_acc))
    else:
        history["accuracy"].append(history["accuracy"][-1])
    if val1:
        history_acc_i += 1
        if str(my_acc1).lower() != "nan":
            history[f"accuracy{history_acc_i}"].append(float(my_acc1))
        else:
            history[f"accuracy{history_acc_i}"].append(
                history[f"accuracy{history_acc_i}"][-1]
            )
    if val2:
        history_acc_i += 1
        if str(my_acc2).lower() != "nan":
            history[f"accuracy{history_acc_i}"].append(float(my_acc2))
        else:
            history[f"accuracy{history_acc_i}"].append(
                history[f"accuracy{history_acc_i}"][-1]
            )
    if val3:
        history_acc_i += 1
        if str(my_acc3).lower() != "nan":
            history[f"accuracy{history_acc_i}"].append(float(my_acc3))
        else:
            history[f"accuracy{history_acc_i}"].append(
                history[f"accuracy{history_acc_i}"][-1]
            )
    if val4:
        history_acc_i += 1
        if str(my_acc4).lower() != "nan":
            history[f"accuracy{history_acc_i}"].append(float(my_acc4))
        else:
            history[f"accuracy{history_acc_i}"].append(
                history[f"accuracy{history_acc_i}"][-1]
            )
    if val5:
        history_acc_i += 1
        if str(my_acc5).lower() != "nan":
            history[f"accuracy{history_acc_i}"].append(float(my_acc5))
        else:
            history[f"accuracy{history_acc_i}"].append(
                history[f"accuracy{history_acc_i}"][-1]
            )
    print(
        "\nEpoch {}/{}, Loss: {:.5f}, Accuracy: {:.5f} \n".format(
            epoch + 1, epoch_count, my_loss, my_acc
        )
    )

    # testing trained model on unused values
    model.eval()
    for batches in tqdm(range(math.ceil(X_test_tensors.shape[0] / batchsize))):
        output = 0
        scaled_batch = 0
        with torch.no_grad():
            if batches >= math.ceil(X_test_tensors.shape[0] / batchsize) - 1:
                model.set_hiddens(
                    X_test_tensors[batches * batchsize :].shape[0], device
                )
                # print("runs")
                output = model.forward(
                    X_test_tensors[batches * batchsize :], device, hiddens=False
                )
                scaled_batch = y_test_tensors[batches * batchsize :]
            else:
                model.set_hiddens(batchsize, device)
                output = model.forward(
                    X_test_tensors[batches * batchsize : (batches + 1) * batchsize],
                    device,
                    hiddens=False,
                )
                scaled_batch = y_test_tensors[
                    batches * batchsize : (batches + 1) * batchsize
                ]

            val_torch_outputs = output.detach().clone()

            # loss
            val_loss1 = MSEloss_fn(
                output[:, ix[0] : ix[1]], scaled_batch[:, ix[0] : ix[1]].double()
            )
            val_loss2 = CE1loss_fn(
                output[:, ix[2] : ix[3]],
                torch.argmax(scaled_batch[:, ix[2] : ix[3]], dim=1),
            )
            val_loss3 = CE2loss_fn(
                output[:, ix[4] : ix[5]],
                torch.argmax(scaled_batch[:, ix[4] : ix[5]], dim=1),
            )
            val_loss4 = CE3loss_fn(
                output[:, ix[6] : ix[7]],
                torch.argmax(scaled_batch[:, ix[6] : ix[7]], dim=1),
            )
            val_loss5 = CE4loss_fn(
                output[:, ix[8] : ix[9]],
                torch.argmax(scaled_batch[:, ix[8] : ix[9]], dim=1),
            )
            val_loss_list1.append(float(val_loss1.item()))
            val_loss_list2.append(float(val_loss2.item()))
            val_loss_list3.append(float(val_loss3.item()))
            val_loss_list4.append(float(val_loss4.item()))
            val_loss_list4.append(float(val_loss5.item()))

            val_loss_l = []
            if val1:
                val_loss_l.append(val_loss1)
            if val2:
                val_loss_l.append(val_loss2)
            if val3:
                val_loss_l.append(val_loss3)
            if val4:
                val_loss_l.append(val_loss4)
            if val5:
                val_loss_l.append(val_loss5)

            val_loss = sum(val_loss_l)

            val_loss_list.append(float(val_loss.item()))

        size = val_torch_outputs.shape[0]
        compare = (
            (
                torch.pow(
                    val_torch_outputs[:, ix[0] : ix[1]]
                    - scaled_batch[:, ix[0] : ix[1]],
                    2,
                )
            )
            .float()
            .sum()
        )
        if compare <= 0:
            return 100
        else:
            val_acc_list1.append(100 / (1 + (compare / size * 3)))
        inds_o_direction = torch.argmax(val_torch_outputs[:, ix[2] : ix[3]], dim=1)
        inds_s_direction = torch.argmax(scaled_batch[:, ix[2] : ix[3]], dim=1)
        inds_o_g_direction = torch.argmax(val_torch_outputs[:, ix[4] : ix[5]], dim=1)
        inds_s_g_direction = torch.argmax(scaled_batch[:, ix[4] : ix[5]], dim=1)
        inds_o_icon = torch.argmax(val_torch_outputs[:, ix[6] : ix[7]], dim=1)
        inds_s_icon = torch.argmax(scaled_batch[:, ix[6] : ix[7]], dim=1)
        inds_o_condition = torch.argmax(val_torch_outputs[:, ix[8] : ix[9]], dim=1)
        inds_s_condition = torch.argmax(scaled_batch[:, ix[8] : ix[9]], dim=1)
        val_acc_list2.append(
            (inds_o_direction == inds_s_direction).sum().item() / size * 100
        )
        val_acc_list3.append(
            (inds_o_direction == inds_s_direction).sum().item() / size * 100
        )
        val_acc_list4.append((inds_o_icon == inds_s_icon).sum().item() / size * 100)
        val_acc_list5.append(
            (inds_o_condition == inds_s_condition).sum().item() / size * 100
        )

        metrics[1].update(
            inds_o_direction,
            inds_s_direction,
        )
        metrics[3].update(
            inds_o_g_direction,
            inds_s_g_direction,
        )
        metrics[5].update(
            inds_o_icon,
            inds_s_icon,
        )
        metrics[7].update(
            inds_o_condition,
            inds_s_condition,
        )
    val_acc_list = []
    if val1:
        val_acc_list = val_acc_list + val_acc_list1
    if val2:
        val_acc_list = val_acc_list + val_acc_list2
    if val3:
        val_acc_list = val_acc_list + val_acc_list3
    if val4:
        val_acc_list = val_acc_list + val_acc_list4
    if val5:
        val_acc_list = val_acc_list + val_acc_list5

    my_val_acc = torch.FloatTensor(val_acc_list).to(device)
    my_val_acc = clean_torch(my_val_acc).mean()
    my_val_acc1 = torch.FloatTensor(val_acc_list1).to(device)
    my_val_acc1 = clean_torch(my_val_acc1).mean()
    my_val_acc2 = torch.FloatTensor(val_acc_list2).to(device)
    my_val_acc2 = clean_torch(my_val_acc2).mean()
    my_val_acc3 = torch.FloatTensor(val_acc_list3).to(device)
    my_val_acc3 = clean_torch(my_val_acc3).mean()
    my_val_acc4 = torch.FloatTensor(val_acc_list4).to(device)
    my_val_acc4 = clean_torch(my_val_acc4).mean()
    my_val_acc5 = torch.FloatTensor(val_acc_list5).to(device)
    my_val_acc5 = clean_torch(my_val_acc5).mean()
    my_val_loss = torch.FloatTensor(val_loss_list).to(device)
    my_val_loss = clean_torch(my_val_loss).mean()
    my_val_loss1 = torch.FloatTensor(val_loss_list1).to(device)
    my_val_loss1 = clean_torch(my_val_loss1).mean()
    my_val_loss2 = torch.FloatTensor(val_loss_list2).to(device)
    my_val_loss2 = clean_torch(my_val_loss2).mean()
    my_val_loss3 = torch.FloatTensor(val_loss_list3).to(device)
    my_val_loss3 = clean_torch(my_val_loss3).mean()
    my_val_loss4 = torch.FloatTensor(val_loss_list4).to(device)
    my_val_loss4 = clean_torch(my_val_loss4).mean()
    my_val_loss5 = torch.FloatTensor(val_loss_list5).to(device)
    my_val_loss5 = clean_torch(my_val_loss5).mean()
    
    history_loss_i = 0
    if str(my_val_loss).lower() != "nan":
        history["val_loss"].append(float(my_val_loss))
    else:
        history["val_loss"].append(history["val_loss"][-1])

    if val1:
        history_loss_i += 1
        if str(my_val_loss1).lower() != "nan":
            history[f"val_loss{history_loss_i}"].append(float(my_val_loss1))
        else:
            history[f"val_loss{history_loss_i}"].append(
                history[f"val_loss{history_loss_i}"][-1]
            )

    if val2:
        history_loss_i += 1
        if str(my_val_loss2).lower() != "nan":
            history[f"val_loss{history_loss_i}"].append(float(my_val_loss2))
        else:
            history[f"val_loss{history_loss_i}"].append(
                history[f"val_loss{history_loss_i}"][-1]
            )

    if val3:
        history_loss_i += 1
        if str(my_val_loss3).lower() != "nan":
            history[f"val_loss{history_loss_i}"].append(float(my_val_loss3))
        else:
            history[f"val_loss{history_loss_i}"].append(
                history[f"val_loss{history_loss_i}"][-1]
            )

    if val4:
        history_loss_i += 1
        if str(my_val_loss4).lower() != "nan":
            history[f"val_loss{history_loss_i}"].append(float(my_val_loss4))
        else:
            history[f"val_loss{history_loss_i}"].append(
                history[f"val_loss{history_loss_i}"][-1]
            )

    if val5:
        history_loss_i += 1
        if str(my_val_loss5).lower() != "nan":
            history[f"val_loss{history_loss_i}"].append(float(my_val_loss5))
        else:
            history[f"val_loss{history_loss_i}"].append(
                history[f"val_loss{history_loss_i}"][-1]
            )

    history_acc_i = 0
    if str(my_val_acc).lower() != "nan":
        history["val_accuracy"].append(float(my_val_acc))
    else:
        history["val_accuracy"].append(history["val_accuracy"][-1])

    if val1:
        history_acc_i += 1
        if str(my_val_acc1).lower() != "nan":
            history[f"val_accuracy{history_acc_i}"].append(float(my_val_acc1))
        else:
            history[f"val_accuracy{history_acc_i}"].append(
                history[f"val_accuracy{history_acc_i}"][-1]
            )

    if val2:
        history_acc_i += 1
        if str(my_val_acc2).lower() != "nan":
            history[f"val_accuracy{history_acc_i}"].append(float(my_val_acc2))
        else:
            history[f"val_accuracy{history_acc_i}"].append(
                history[f"val_accuracy{history_acc_i}"][-1]
            )

    if val3:
        history_acc_i += 1
        if str(my_val_acc3).lower() != "nan":
            history[f"val_accuracy{history_acc_i}"].append(float(my_val_acc3))
        else:
            history[f"val_accuracy{history_acc_i}"].append(
                history[f"val_accuracy{history_acc_i}"][-1]
            )

    if val4:
        history_acc_i += 1
        if str(my_val_acc4).lower() != "nan":
            history[f"val_accuracy{history_acc_i}"].append(float(my_val_acc4))
        else:
            history[f"val_accuracy{history_acc_i}"].append(
                history[f"val_accuracy{history_acc_i}"][-1]
            )

    if val5:
        history_acc_i += 1
        if str(my_val_acc5).lower() != "nan":
            history[f"val_accuracy{history_acc_i}"].append(float(my_val_acc5))
        else:
            history[f"val_accuracy{history_acc_i}"].append(
                history[f"val_accuracy{history_acc_i}"][-1]
            )

    print(
        "\nEpoch {}/{}, val Loss: {:.8f}, val Accuracy: {:.5f}".format(
            # epoch + 1, epoch_count, convert_loss(my_val_loss), my_val_acc
            epoch + 1,
            epoch_count,
            my_val_loss,
            my_val_acc,
        )
    )
    del (X_train_tensors, X_test_tensors, y_train_tensors, y_test_tensors, CE4loss_fn,CE3loss_fn,CE2loss_fn,CE1loss_fn,MSEloss_fn, labels, val_labels,my_acc,my_loss,my_acc1,my_loss1,my_acc2,my_loss2,my_acc3,my_loss3,my_acc4,my_loss4,my_acc5,my_loss5,my_val_acc,my_val_loss,my_val_loss1,my_val_acc1,my_val_loss2,my_val_acc2,my_val_loss3,my_val_acc3,my_val_loss4,my_val_acc4,my_val_loss5,my_val_acc5,output, scaled_batch,
        inds_o_direction,inds_s_direction,inds_o_g_direction,inds_s_g_direction,inds_o_icon,inds_s_icon,inds_o_condition,inds_s_condition)
    

    return model, history, metrics, optimizer


# plotting predicted data only for hourly named models
# future predictions has no labels so they can't be printed
def plotting_Prediction_hourly(
    all_input, output, plot_text, hourly=[], hourly24=[], mode="normal", t=0
):
    titles = ["Temperature", "Wind direction", "Wind speed", "Visibility"]
    input = [
        all_input[3],
        all_input[4],
        all_input[5],
        all_input[9],
        all_input[12],
        all_input[16],
    ]
    x = [1, 2, 25]
    name = "Hourly"
    fig, axs = plt.subplots(len(titles) + 1, 1, figsize=(12, 8))
    for i in range(len(titles)):
        axs[i].set_title(titles[i])
        axs[i].plot(x, [input[i], output[0][0][i].item(), output[1][0][i].item()])
        if plot_text != "future":
            axs[i].plot(x, [input[i], hourly[i], hourly24[i]])
        axs[len(titles)].text(
            0,
            0.3,
            f"icon: real-{hourly[4]} / prediction-{output[0][0][4]}, condition: real-{hourly[4]}/ prediction-{output[0][0][5]}",
            fontsize=15,
        )
        axs[len(titles)].text(
            0,
            0.6,
            f"24 icon: real-{hourly24[4]} / prediction-{output[1][0][4]}, condition: real-{hourly24[4]}/ prediction-{output[1][0][5]}",
            fontsize=15,
        )

    if plot_text != "future":
        axs[0].legend(["predicted", "real"], loc="upper left")
    fig.suptitle(f"prediction {plot_text}")
    plt.tight_layout()
    if mode == "timestep" or mode == "ts":
        plt.savefig(f"./Plots/{name}_{t}_prediction_{plot_text}.png")
    else:
        plt.savefig(f"./Plots/{name}_prediction_{plot_text}.png")


# aktively used reshaping and predicting
def prediction(model, train, label, device, features=[], indx=[], cities_next=4):
    ix, val1, val2, val3, val4, val5 = loss_indexer(indx, len(features))
    X_test_tensors = torch.reshape(
        Variable(torch.Tensor(train).to(device)).to(device),
        (1, train.shape[0], train.shape[1]),
    ).to(device)
    y_test_tensors = []
    if len(label) > 0:
        y_test_tensors = torch.reshape(
            Variable(torch.Tensor(label)).to(device),
            (1, label.shape[0], label.shape[1]),
        ).to(device)
    X_test_tensors = scale_features(X_test_tensors, indx, cities_next)
    if len(label) > 0:
        y_test_tensors = scale_label(y_test_tensors, indx)
    model.eval()
    val_acc_list1 = []
    val_acc_list2 = []
    val_acc_list3 = []
    val_acc_list4 = []
    val_acc_list5 = []
    output = 0
    scaled_batch = 0
    with torch.no_grad():
        model.set_hiddens(1, device)
        output = model.forward(X_test_tensors, device, hiddens=False)

        if len(label) > 0:
            scaled_batch = y_test_tensors

        val_torch_outputs = output.detach().clone()

    size = val_torch_outputs.shape[0]

    if len(label) > 0:
        compare = (
            (torch.pow(val_torch_outputs[:, :3] - scaled_batch[:, :3], 2)).float().sum()
        )

        if compare <= 0:
            return 100
        else:
            val_acc_list1.append(100 / (1 + (compare / size * 3)))
        inds_o_direction = torch.argmax(val_torch_outputs[:, ix[2] : ix[3]], dim=1)
        inds_s_direction = torch.argmax(scaled_batch[:, ix[2] : ix[3]], dim=1)
        inds_o_g_direction = torch.argmax(val_torch_outputs[:, ix[4] : ix[5]], dim=1)
        inds_s_g_direction = torch.argmax(scaled_batch[:, ix[4] : ix[5]], dim=1)
        inds_o_icon = torch.argmax(val_torch_outputs[:, ix[6] : ix[7]], dim=1)
        inds_s_icon = torch.argmax(scaled_batch[:, ix[6] : ix[7]], dim=1)
        inds_o_condition = torch.argmax(val_torch_outputs[:, ix[8] : ix[9]], dim=1)
        inds_s_condition = torch.argmax(scaled_batch[:, ix[8] : ix[9]], dim=1)
        val_acc_list2.append(
            (inds_o_direction == inds_s_direction).sum().item() / size * 100
        )
        val_acc_list3.append(
            (inds_o_direction == inds_s_direction).sum().item() / size * 100
        )
        val_acc_list4.append((inds_o_icon == inds_s_icon).sum().item() / size * 100)
        val_acc_list5.append(
            (inds_o_condition == inds_s_condition).sum().item() / size * 100
        )

        val_acc_list = []
        if val1:
            val_acc_list = val_acc_list + val_acc_list1
        if val2:
            val_acc_list = val_acc_list + val_acc_list2
        if val3:
            val_acc_list = val_acc_list + val_acc_list3
        if val4:
            val_acc_list = val_acc_list + val_acc_list4
        if val5:
            val_acc_list = val_acc_list + val_acc_list5

        my_val_acc = torch.FloatTensor(val_acc_list).to(device)
        my_val_acc = clean_torch(my_val_acc).mean()
        my_val_acc1 = torch.FloatTensor(val_acc_list1).to(device)
        my_val_acc1 = clean_torch(my_val_acc1).mean()
        my_val_acc2 = torch.FloatTensor(val_acc_list2).to(device)
        my_val_acc2 = clean_torch(my_val_acc2).mean()
        my_val_acc3 = torch.FloatTensor(val_acc_list3).to(device)
        my_val_acc3 = clean_torch(my_val_acc3).mean()
        my_val_acc4 = torch.FloatTensor(val_acc_list4).to(device)
        my_val_acc4 = clean_torch(my_val_acc4).mean()
        my_val_acc5 = torch.FloatTensor(val_acc_list5).to(device)
        my_val_acc5 = clean_torch(my_val_acc5).mean()
    return output


# https://www.geeksforgeeks.org/python-last-occurrence-of-some-element-in-a-list/
def last_occurrence(lst, val):
    index = -1
    while True:
        try:
            index = lst.index(val, index + 1)
        except ValueError:
            return index


def future_prediction(model_name, device, id=[], checks=[], hours=1, show_all=True):
    model, optimizer, history, others = load_own_Model(model_name, device)
    ids, train, label = gld.get_predictDataHourly(
        dt.now(),
        id=id,
        seq=others["sequences"],
        feature_labels=others["features"],
        next_city_amount=others["city_next"],
        month=others["month"],
        hours=others["hours"],
        position=others["position"],
        forecast=hours,
    )
    uids = list(set(ids))
    if type(model) == "str":
        print("choose an other name")
        return []
    output_list = torch.zeros(1, others["output_count"]).to(device)
    id_list = []
    time_list = []
    vals = []
    for ui in uids:
        t = train[last_occurrence(ids, ui), :, -5][-1] + 1
        if t >= 24:
            t -= 24
        output = prediction(
            model,
            train[last_occurrence(ids, ui)],
            [],
            device,
            features=others["features"],
            indx=others["indx"],
            cities_next=others["city_next"],
        )
        id_list.append(ui)
        time_list.append(int(t))
        vals.append(
            [
                train[last_occurrence(ids, ui), :, -3][-1],
                train[last_occurrence(ids, ui), :, -2][-1],
                train[last_occurrence(ids, ui), :, -1][-1],
            ]
        )
        output_list = torch.cat([output_list, output.detach().clone()], dim=0)
    outputs, ids, times = gld.continue_prediction(
        model,
        output_list[1:],
        id_list,
        time_list,
        vals,
        hours - 1,
        dt.now(),
        device,
        prediction=prediction,
        show_all=show_all,
        ids=id,
        next_city_amount=others["city_next"],
    )
    output_np = unscale_output(
        outputs,
        ids,
        times,
        continueing=True,
        features=others["features"],
        indx=others["indx"],
    )
    return output_np


def check_prediction(model_name, device, id=[], hours=1):
    train_tensors = Variable(torch.Tensor(train).to(device)).to(device)
    input_final = torch.reshape(train_tensors, (1, 1, train_tensors.shape[-1])).to(
        device
    )
    ids, train, label = gld.get_predictDataHourly(
        dt.now(),
        id=id,
    )
    uids = list(set(ids))
    print("trainLabel: ", train.shape, label.shape)
    model, optimizer, history, others = load_own_Model(model_name, device)
    if type(model) == "str":
        print("choose an other name")
        return []
    output_list = torch.zeros(1, 55)
    for ui in uids:
        t = max(train[last_occurrence(ids, ui), :, -5]) + 1
        if t >= 24:
            t -= 24
        # print(train[last_occurrence(ids, ui), :, -5])
        output = prediction(model, train[last_occurrence(ids, ui)], [], device)
        output[54] = ui
        output[55] = t
        output_list = torch.cat(output, dim=1)

    return output_list


def forecast_weather(model_name, h, ids, show_all, name="", checks=[], features=[]):
    if str(show_all) == "false":
        show_all = False
    elif str(show_all) == "true":
        show_all = True
    device = check_cuda()
    outs = future_prediction(
        str(model_name), device, id=ids, hours=int(h), show_all=show_all, checks=checks
    )
    show_image(outs, hours=int(h), name=name, checks=checks, features=features)
