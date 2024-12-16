import CollectData.get_learning_data as gld
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import json
from datetime import datetime as dt
from datetime import timedelta as td
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle

# from sklearn import preprocessing as pr
from sklearn.model_selection import train_test_split

# from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchmetrics.classification import MulticlassConfusionMatrix
import math
import warnings
import random

plt.switch_backend("agg")

warnings.filterwarnings("ignore")


# checking if cuda is used. if not it choses cpu
def check_cuda():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("this device uses " + device + " to train data")
    return torch.device(device)


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


# loading model if already saved or creating a new model
def load_own_Model(
    name,
    device,
    input_count=230,
    output_count=46,
    learning_rate=0.001,
    layer=3,
    hiddensize=7,
    sequences=24,
    dropout=0.1,
    batchsize=0,
    prediction=False,
):
    history = {
        "accuracy": [0],
        "loss": [0],
        "val_accuracy": [0],
        "val_loss": [0],
        "accuracy1": [0],
        "loss1": [0],
        "val_accuracy1": [0],
        "val_loss1": [0],
        "accuracy2": [0],
        "loss2": [0],
        "val_accuracy2": [0],
        "val_loss2": [0],
        "accuracy3": [0],
        "loss3": [0],
        "val_accuracy3": [0],
        "val_loss3": [0],
        "accuracy4": [0],
        "loss4": [0],
        "val_accuracy4": [0],
        "val_loss4": [0],
        # "argmax_accuracy": [0],
        # "val_argmax_accuracy": [0],
    }
    model = {"haha": [1, 2, 3]}

    if os.path.exists(f"./Models/{name}.pth"):
        checkpoint = torch.load(f"./Models/{name}.pth", map_location=device)
        model = checkpoint["model"]
    elif prediction:
        print("Model not found")
        return "error", "error", "error"
    else:
        print("Data not found or not complete")
        model = PyTorch_LSTM(
            input_count,
            output_count,
            device,
            h_size=hiddensize,
            seq_size=sequences,
            layer=layer,
            dropout=dropout,
            batchsize=batchsize,
        )
    if os.path.exists(f"./Models/{name}_history.json"):
        with open(f"./Models/{name}_history.json", "r") as f:
            history = json.load(f)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if os.path.exists(f"./Models/{name}.pth"):
        checkpoint = torch.load(f"./Models/{name}.pth", map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Model found")
    model.train()
    # metric = MulticlassConfusionMatrix(num_classes=21).to(device)
    return model, optimizer, history


# saving model if saving_mode set to ts or timestamp it will use the number for the model to save it.
# ts helps to choose saved model data before the model started to overfit or not work anymore
def save_own_Model(name, history, model, optimizer):
    with open(f"./Models/{name}_history.json", "w") as fp:
        json.dump(history, fp)
    torch.save(
        {"model": model, "optimizer": optimizer.state_dict()},
        f"./Models/{name}.pth",
    )
    print("Saved model")


def cropping(image, x_list, y_list):
    max_x = max(x_list) + 100
    min_x = min(x_list) - 100
    max_y = max(y_list) + 100
    min_y = min(y_list) - 100
    return image.crop((min_x, min_y, max_x, max_y))


def points(image, drw, df, ims, ofs):
    for i, d in df.iterrows():
        print(d)
        drw.ellipse(
            xy=(d.lon - 3, d.lat - 3, d.lon + 3, d.lat + 3),
            fill="red",
        )
        if d.icon is not None:
            ix = ofs.index(d.icon)
            image.paste(ims[ix], (int(d.lon) - 20, int(d.lat) - 20), mask=ims[ix])
        if d.condition is not None:
            ix = ofs.index(d.condition)
            image.paste(ims[ix], (int(d.lon), int(d.lat) - 20), mask=ims[ix])
        if d.direction is not None:
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
    print(onlyfiles)
    return ims, onlyfiles


def show_image(outs):
    im = Image.open("Images/Landkarte_Deutschland.png").convert("RGBA")
    stations = gld.load_stations_csv()
    stations["lat"] = [-x for x in stations["lat"].to_list()]
    stations["lon"] = l_to_px(stations["lon"].to_list(), 648, 50)
    stations["lat"] = l_to_px(stations["lat"], 904, 60)
    outs_stations = pd.merge(outs, stations, on=["ID"])
    ims, ofs = load_images()
    # print(stations["lon"], stations["lat"])
    drw = ImageDraw.Draw(im)
    im = points(im, drw, outs_stations, ims, ofs)
    # im = cropping(im, outs_stations["lon"], outs_stations["lat"])
    im.show()
    return im


def approximation(all):
    x = np.arange(0, len(all))
    y = list(all)
    [b, m] = np.polynomial.polynomial.polyfit(x, y, 1)
    # print("b: ", b, ", m: ", m)
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
    # summarize history for accuracy
    fig, ax = plt.subplots(6, figsize=(20, 12), sharey="row")
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.title("model accuracy")
    # ax[0].plot(history["accuracy"][1:])
    # ax[0].plot(history["val_accuracy"][1:])
    ax[0].plot(history["accuracy1"][1:])
    ax[0].plot(history["val_accuracy1"][1:])
    ax[0].plot(history["accuracy2"][1:])
    ax[0].plot(history["val_accuracy2"][1:])
    ax[0].plot(history["accuracy3"][1:])
    ax[0].plot(history["val_accuracy3"][1:])
    ax[0].plot(history["accuracy4"][1:])
    ax[0].plot(history["val_accuracy4"][1:])
    ax[1].plot(history["accuracy"][1:])
    ax[1].plot(history["val_accuracy"][1:])
    ax[2].plot(history["accuracy1"][1:])
    ax[2].plot(history["val_accuracy1"][1:])
    ax[3].plot(history["accuracy2"][1:])
    ax[3].plot(history["val_accuracy2"][1:])
    ax[4].plot(history["accuracy3"][1:])
    ax[4].plot(history["val_accuracy3"][1:])
    ax[5].plot(history["accuracy4"][1:])
    ax[5].plot(history["val_accuracy4"][1:])
    # if len(history["accuracy"]) > min_amount:
    #     ax[2].plot(history["accuracy"][(-1 * min_amount) :])
    #     ax[3].plot(history["val_accuracy"][(-1 * min_amount) :])
    ax[0].grid(axis="y")
    ax[1].grid(axis="y")
    ax[2].grid(axis="y")
    ax[3].grid(axis="y")
    ax[4].grid(axis="y")
    ax[5].grid(axis="y")
    ax[0].legend(
        [
            "train",
            "test",
        ],
        # ["svm train", "svm test", "rf train", "rf test"],
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
    history["loss1"] = [
        -_ if _ < 0 else _ for _ in (pd.Series(history["loss1"])).tolist()
    ]
    history["val_loss1"] = [
        -_ if _ < 0 else _ for _ in (pd.Series(history["val_loss1"])).tolist()
    ]
    history["loss2"] = [
        -_ if _ < 0 else _ for _ in (pd.Series(history["loss2"])).tolist()
    ]
    history["val_loss2"] = [
        -_ if _ < 0 else _ for _ in (pd.Series(history["val_loss2"])).tolist()
    ]
    history["loss3"] = [
        -_ if _ < 0 else _ for _ in (pd.Series(history["loss3"])).tolist()
    ]
    history["val_loss3"] = [
        -_ if _ < 0 else _ for _ in (pd.Series(history["val_loss3"])).tolist()
    ]
    history["loss4"] = [
        -_ if _ < 0 else _ for _ in (pd.Series(history["loss4"])).tolist()
    ]
    history["val_loss4"] = [
        -_ if _ < 0 else _ for _ in (pd.Series(history["val_loss4"])).tolist()
    ]

    b, m = approximation(history["loss"][1:])
    f = [b + (m * x) for x in range(1, len(history["loss"]))]
    b1, m1 = approximation(history["loss1"][1:])
    f1 = [b1 + (m1 * x) for x in range(1, len(history["loss1"]))]
    b2, m2 = approximation(history["loss2"][1:])
    f2 = [b2 + (m2 * x) for x in range(1, len(history["loss2"]))]
    b3, m3 = approximation(history["loss3"][1:])
    f3 = [b3 + (m3 * x) for x in range(1, len(history["loss3"]))]
    b4, m4 = approximation(history["loss4"][1:])
    f4 = [b4 + (m4 * x) for x in range(1, len(history["loss4"]))]

    # summarize history for loss
    fig, ax = plt.subplots(6, figsize=(20, 12), sharey="row")

    ax[0].plot(history["loss"][1:])
    ax[0].plot(history["val_loss"][1:])
    ax[0].plot(history["loss1"][1:])
    ax[0].plot(history["val_loss1"][1:])
    ax[0].plot(history["loss2"][1:])
    ax[0].plot(history["val_loss2"][1:])
    ax[0].plot(history["loss3"][1:])
    ax[0].plot(history["val_loss3"][1:])
    ax[0].plot(history["loss4"][1:])
    ax[0].plot(history["val_loss4"][1:])
    ax[0].grid(axis="y")
    ax[1].plot(history["loss"][1:], alpha=0.8)
    ax[1].plot(history["val_loss"][1:], alpha=0.75)
    ax[1].plot(f)
    ax[1].grid(axis="y")
    ax[1].legend(
        [
            "train / Epoche:" + str(epoche + 1),
            "test",
            # "linear train: {:.1f} + {:.5f}x".format(b * 10, m),
            "linear train: {:.5f}x".format(m),
        ],
        loc="upper left",
    )
    ax[2].plot(history["loss1"][1:], alpha=0.8)
    ax[2].plot(history["val_loss1"][1:], alpha=0.75)
    ax[2].plot(f1)
    ax[2].grid(axis="y")
    ax[2].legend(
        [
            "train / Epoche:" + str(epoche + 1),
            "test",
            # "linear train: {:.1f} + {:.5f}x".format(b * 10, m),
            "linear train: {:.5f}x".format(m1),
        ],
        loc="upper left",
    )
    ax[3].plot(history["loss2"][1:], alpha=0.8)
    ax[3].plot(history["val_loss2"][1:], alpha=0.75)
    ax[3].plot(f2)
    ax[3].grid(axis="y")
    ax[3].legend(
        [
            "train / Epoche:" + str(epoche + 1),
            "test",
            # "linear train: {:.1f} + {:.5f}x".format(b * 10, m),
            "linear train: {:.5f}x".format(m2),
        ],
        loc="upper left",
    )
    ax[4].plot(history["loss3"][1:], alpha=0.8)
    ax[4].plot(history["val_loss3"][1:], alpha=0.75)
    ax[4].plot(f3)
    ax[4].grid(axis="y")
    ax[4].legend(
        [
            "train / Epoche:" + str(epoche + 1),
            "test",
            # "linear train: {:.1f} + {:.5f}x".format(b * 10, m),
            "linear train: {:.5f}x".format(m3),
        ],
        loc="upper left",
    )
    ax[5].plot(history["loss4"][1:], alpha=0.8)
    ax[5].plot(history["val_loss4"][1:], alpha=0.75)
    ax[5].plot(f4)
    ax[5].grid(axis="y")
    ax[5].legend(
        [
            "train / Epoche:" + str(epoche + 1),
            "test",
            # "linear train: {:.1f} + {:.5f}x".format(b * 10, m),
            "linear train: {:.5f}x".format(m4),
        ],
        loc="upper left",
    )
    plt.savefig(f"./Plots/{name_tag}_loss.png")
    plt.close()

    # summarize MulticlassConfusionMatrix
    fig, ax = plt.subplots(
        3,
        2,
        figsize=(8, 16),
        sharey="row",
    )
    plt.title("Confusion Matrix")
    metric_names = [
        "Train direction",
        "Test Icon",
        "Train Icon",
        "Test Icon",
        "Train Condition",
        "Test Condition",
    ]
    ax[0, 0].title.set_text(metric_names[0])
    metrics[0].plot(
        ax=ax[0, 0],
        # cmap="Blues",
        # colorbar=False,
    )
    ax[0, 0].xaxis.set_ticklabels(directions)
    ax[0, 0].yaxis.set_ticklabels(directions)
    ax[0, 1].title.set_text(metric_names[1])
    metrics[1].plot(
        ax=ax[0, 1],
        # cmap="Blues",
        # colorbar=False,
    )
    ax[0, 1].xaxis.set_ticklabels(directions)
    ax[0, 1].yaxis.set_ticklabels(directions)
    for i in range(2, len(metrics)):
        j = math.floor(i / 2)
        k = i % 2
        ax[j, k].title.set_text(metric_names[i])
        metrics[i].plot(
            ax=ax[j, k],
            # cmap="Blues",
            # colorbar=False,
        )
        ax[j, k].xaxis.set_ticklabels(icons)
        ax[j, k].yaxis.set_ticklabels(icons)
    fig.set_figwidth(30)
    fig.set_figheight(30)
    plt.savefig(f"./Plots/{name_tag}_matrix.png")
    plt.close()


# unscaling the output because it usually doesnt get over 1
def unscale_output(output):
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
    output[:, 0] *= 100
    if output[:, 1] <= 0:
        output[:, 1] = 0
    else:
        output[:, 1] *= 100
    output[:, 2] *= 10000

    inds_o_direction = torch.argmax(output[:, 3:12], dim=1)
    inds_o_icon = torch.argmax(output[:, 12:33], dim=1)
    inds_o_condition = torch.argmax(output[:, 33:54], dim=1)
    direction = [directions[i] for i in inds_o_direction]
    icon = [icons[i] for i in inds_o_icon]
    condition = [icons[i] for i in inds_o_condition]

    return pd.DataFrame(
        {
            "temperature": output[:, 0].tolist(),
            "wind_speed": output[:, 1].tolist(),
            "visibility": output[:, 2].tolist(),
            "direction": direction,
            "icon": icon,
            "condition": condition,
        }
    )


# scaling label to check if output was good enough
def scale_label(output):
    output[:, 0] /= 100
    output[:, 1] /= 100
    output[:, 2] /= 10000
    # output[:, :, 4] /= 21
    # output[:, :, 5] /= 21
    return output


def scale_features(output):
    for i in range(5):
        output[:, :, i * 5] /= 100
        output[:, :, 1 + i * 5] /= 100
        output[:, :, 2 + i * 5] /= 10000
        # output[:, :, 4 + i * 5] /= 21
        # output[:, :, 5 + i * 5] /= 21
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
    history,
    device,
    epoch_count=1,
    epoch=0,
    batchsize=0,
):
    MSEloss_fn = nn.MSELoss().to(device)
    CE1loss_fn = nn.CrossEntropyLoss().to(device)
    CE2loss_fn = nn.CrossEntropyLoss().to(device)
    CE3loss_fn = nn.CrossEntropyLoss().to(device)
    if batchsize == 0:
        batchsize == len(feature)

    X_train, X_test, y_train, y_test = train_test_split(
        feature, label, test_size=0.02, random_state=42
    )
    X_train_tensors = Variable(torch.Tensor(X_train).to(device)).to(device)
    X_test_tensors = Variable(torch.Tensor(X_test).to(device)).to(device)
    y_train_tensors = Variable(torch.Tensor(y_train)).to(device)
    y_test_tensors = Variable(torch.Tensor(y_test)).to(device)

    # print("X_train_tensors", X_train_tensors.shape)
    # print("y_train_tensors", y_train_tensors.shape)
    # print("X_test_tensors", X_test_tensors.shape)
    # print("y_test_tensors", y_test_tensors.shape)

    X_train_tensors = scale_features(X_train_tensors)
    y_train_tensors = scale_label(y_train_tensors)
    X_test_tensors = scale_features(X_test_tensors)
    y_test_tensors = scale_label(y_test_tensors)
    metrics = [
        MulticlassConfusionMatrix(num_classes=9).to(device) for i in range(2)
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
    labels = Variable(torch.Tensor(np.array(y_train.tolist())).to(device)).to(
        device
    )  # .flatten()
    val_labels = Variable(torch.Tensor(np.array(y_test.tolist())).to(device)).to(
        device
    )  # .flatten()
    val_loss_list = []
    val_loss_list1 = []
    val_acc_list1 = []
    val_loss_list2 = []
    val_acc_list2 = []
    val_loss_list3 = []
    val_acc_list3 = []
    val_loss_list4 = []
    val_acc_list4 = []

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
        loss1 = MSEloss_fn(output[:, :3], scaled_batch[:, :3].double())
        loss2 = CE1loss_fn(output[:, 3:12], torch.argmax(scaled_batch[:, 3:12], dim=1))
        loss3 = CE2loss_fn(
            output[:, 12:33], torch.argmax(scaled_batch[:, 12:33], dim=1)
        )
        loss4 = CE3loss_fn(
            output[:, 33:54], torch.argmax(scaled_batch[:, 33:54], dim=1)
        )
        # calculates the loss of the loss functions
        loss = loss1 + loss2 + loss3 + loss4
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

        size = torch_outputs.shape[0]

        compare = (
            (torch.pow(torch_outputs[:, :3] - scaled_batch[:, :3], 2)).float().sum()
        )
        if compare <= 0:
            return 100
        else:
            acc_list1.append(100 / (1 + (compare / size * 3)))
        inds_o_direction = torch.argmax(torch_outputs[:, 3:12], dim=1)
        inds_s_direction = torch.argmax(scaled_batch[:, 3:12], dim=1)
        inds_o_icon = torch.argmax(torch_outputs[:, 12:33], dim=1)
        inds_s_icon = torch.argmax(scaled_batch[:, 12:33], dim=1)
        inds_o_condition = torch.argmax(torch_outputs[:, 33:54], dim=1)
        inds_s_condition = torch.argmax(scaled_batch[:, 33:54], dim=1)
        acc_list2.append(
            (inds_o_direction == inds_s_direction).sum().item() / size * 100
        )
        acc_list3.append((inds_o_icon == inds_s_icon).sum().item() / size * 100)
        acc_list4.append(
            (inds_o_condition == inds_s_condition).sum().item() / size * 100
        )
        metrics[0].update(
            inds_o_direction,
            inds_s_direction,
        )
        # print(
        #     inds_o_direction,
        #     "\n",
        #     inds_s_direction,
        # )
        metrics[2].update(
            inds_o_icon,
            inds_s_icon,
        )
        metrics[3].update(
            inds_o_condition,
            inds_s_condition,
        )

    my_acc = torch.FloatTensor(acc_list1 + acc_list2 + acc_list3 + acc_list4).to(device)
    my_acc = clean_torch(my_acc).mean()
    my_acc1 = torch.FloatTensor(acc_list1).to(device)
    my_acc1 = clean_torch(my_acc1).mean()
    my_acc2 = torch.FloatTensor(acc_list2).to(device)
    my_acc2 = clean_torch(my_acc2).mean()
    my_acc3 = torch.FloatTensor(acc_list3).to(device)
    my_acc3 = clean_torch(my_acc3).mean()
    my_acc4 = torch.FloatTensor(acc_list4).to(device)
    my_acc4 = clean_torch(my_acc4).mean()
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

    # showing training
    if my_loss != "NaN":
        history["loss"].append(float(my_loss))
    else:
        history["loss"].append(history["loss"][-1])

    if my_loss1 != "NaN":
        history["loss1"].append(float(my_loss1))
    else:
        history["loss1"].append(history["loss1"][-1])

    if my_loss2 != "NaN":
        history["loss2"].append(float(my_loss2))
    else:
        history["loss2"].append(history["loss2"][-1])

    if my_loss3 != "NaN":
        history["loss3"].append(float(my_loss3))
    else:
        history["loss3"].append(history["loss3"][-1])

    if my_loss4 != "NaN":
        history["loss4"].append(float(my_loss4))
    else:
        history["loss4"].append(history["loss4"][-1])

    if my_acc != "NaN":
        history["accuracy"].append(float(my_acc))
    else:
        history["accuracy"].append(history["accuracy"][-1])
    if my_acc1 != "NaN":
        history["accuracy1"].append(float(my_acc1))
    else:
        history["accuracy1"].append(history["accuracy1"][-1])
    if my_acc2 != "NaN":
        history["accuracy2"].append(float(my_acc2))
    else:
        history["accuracy2"].append(history["accuracy2"][-1])
    if my_acc3 != "NaN":
        history["accuracy3"].append(float(my_acc3))
    else:
        history["accuracy3"].append(history["accuracy3"][-1])
    if my_acc4 != "NaN":
        history["accuracy4"].append(float(my_acc4))
    else:
        history["accuracy4"].append(history["accuracy4"][-1])
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
                # test_label = val_labels[batches * batchsize :]
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
                # test_label = val_labels[batches * batchsize : (batches + 1) * batchsize]

            val_torch_outputs = output.detach().clone()
            # del output, input

            # loss
            val_loss1 = MSEloss_fn(output[:, :3], scaled_batch[:, :3].double())
            val_loss2 = CE1loss_fn(
                output[:, 3:12], torch.argmax(scaled_batch[:, 3:12], dim=1)
            )
            val_loss3 = CE2loss_fn(
                output[:, 12:33], torch.argmax(scaled_batch[:, 12:33], dim=1)
            )
            val_loss4 = CE3loss_fn(
                output[:, 33:54], torch.argmax(scaled_batch[:, 33:54], dim=1)
            )
            val_loss_list1.append(float(val_loss1.item()))
            val_loss_list2.append(float(val_loss2.item()))
            val_loss_list3.append(float(val_loss3.item()))
            val_loss_list4.append(float(val_loss4.item()))
            val_loss = val_loss1 + val_loss2 + val_loss3 + val_loss4
            val_loss_list.append(float(val_loss.item()))

        size = val_torch_outputs.shape[0]
        compare = (
            (torch.pow(val_torch_outputs[:, :3] - scaled_batch[:, :3], 2)).float().sum()
        )
        if compare <= 0:
            return 100
        else:
            val_acc_list1.append(100 / (1 + (compare / size * 3)))
        inds_o_direction = torch.argmax(val_torch_outputs[:, 3:12], dim=1)
        inds_s_direction = torch.argmax(scaled_batch[:, 3:12], dim=1)
        inds_o_icon = torch.argmax(val_torch_outputs[:, 12:33], dim=1)
        inds_s_icon = torch.argmax(scaled_batch[:, 12:33], dim=1)
        inds_o_condition = torch.argmax(val_torch_outputs[:, 33:54], dim=1)
        inds_s_condition = torch.argmax(scaled_batch[:, 33:54], dim=1)
        val_acc_list2.append(
            (inds_o_direction == inds_s_direction).sum().item() / size * 100
        )
        val_acc_list3.append((inds_o_icon == inds_s_icon).sum().item() / size * 100)
        val_acc_list4.append(
            (inds_o_condition == inds_s_condition).sum().item() / size * 100
        )
        # print(
        #     inds_o_direction,
        #     "\n",
        #     inds_s_direction,
        # )
        metrics[1].update(
            inds_o_direction,
            inds_s_direction,
        )
        metrics[4].update(
            inds_o_icon,
            inds_s_icon,
        )
        metrics[5].update(
            inds_o_condition,
            inds_s_condition,
        )

    my_val_acc = torch.FloatTensor(
        val_acc_list1 + val_acc_list2 + val_acc_list3 + val_acc_list4
    ).to(device)
    my_val_acc = clean_torch(my_val_acc).mean()
    my_val_acc1 = torch.FloatTensor(val_acc_list1).to(device)
    my_val_acc1 = clean_torch(my_val_acc1).mean()
    my_val_acc2 = torch.FloatTensor(val_acc_list2).to(device)
    my_val_acc2 = clean_torch(my_val_acc2).mean()
    my_val_acc3 = torch.FloatTensor(val_acc_list3).to(device)
    my_val_acc3 = clean_torch(my_val_acc3).mean()
    my_val_acc4 = torch.FloatTensor(val_acc_list4).to(device)
    my_val_acc4 = clean_torch(my_val_acc4).mean()
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

    if my_val_loss != "NaN":
        history["val_loss"].append(float(my_val_loss))
    else:
        history["val_loss"].append(history["val_loss"][-1])
    if my_val_loss1 != "NaN":
        history["val_loss1"].append(float(my_val_loss1))
    else:
        history["val_loss1"].append(history["val_loss1"][-1])
    if my_val_loss2 != "NaN":
        history["val_loss2"].append(float(my_val_loss2))
    else:
        history["val_loss2"].append(history["val_loss2"][-1])
    if my_val_loss3 != "NaN":
        history["val_loss3"].append(float(my_val_loss3))
    else:
        history["val_loss3"].append(history["val_loss3"][-1])
    if my_val_loss4 != "NaN":
        history["val_loss4"].append(float(my_val_loss4))
    else:
        history["val_loss4"].append(history["val_loss4"][-1])

    if my_val_acc != "NaN":
        history["val_accuracy"].append(float(my_val_acc))
    else:
        history["val_accuracy"].append(history["val_accuracy"][-1])
    if my_val_acc1 != "NaN":
        history["val_accuracy1"].append(float(my_val_acc1))
    else:
        history["val_accuracy1"].append(history["val_accuracy1"][-1])
    if my_val_acc2 != "NaN":
        history["val_accuracy2"].append(float(my_val_acc2))
    else:
        history["val_accuracy2"].append(history["val_accuracy2"][-1])
    if my_val_acc3 != "NaN":
        history["val_accuracy3"].append(float(my_val_acc3))
    else:
        history["val_accuracy3"].append(history["val_accuracy3"][-1])
    if my_val_acc4 != "NaN":
        history["val_accuracy4"].append(float(my_val_acc4))
    else:
        history["val_accuracy4"].append(history["val_accuracy4"][-1])
    print(
        "\nEpoch {}/{}, val Loss: {:.8f}, val Accuracy: {:.5f}".format(
            # epoch + 1, epoch_count, convert_loss(my_val_loss), my_val_acc
            epoch + 1,
            epoch_count,
            my_val_loss,
            my_val_acc,
        )
    )

    del (
        MSEloss_fn,
        CE1loss_fn,
        CE2loss_fn,
        CE3loss_fn,
        X_train_tensors,
        y_train_tensors,
        X_test_tensors,
        y_test_tensors,
    )

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
def prediction(model, train, label, device):
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
    X_test_tensors = scale_features(X_test_tensors)
    if len(label) > 0:
        y_test_tensors = scale_label(y_test_tensors)
    model.eval()
    val_acc_list1 = []
    val_acc_list2 = []
    val_acc_list3 = []
    val_acc_list4 = []
    output = 0
    scaled_batch = 0
    with torch.no_grad():
        model.set_hiddens(1, device)
        # print("runs")
        output = model.forward(X_test_tensors, device, hiddens=False)

        if len(label) > 0:
            scaled_batch = y_test_tensors
        # test_label = val_labels[batches * batchsize :]

        val_torch_outputs = output.detach().clone()
        # del output, input

    size = val_torch_outputs.shape[0]

    if len(label) > 0:
        compare = (
            (torch.pow(val_torch_outputs[:, :3] - scaled_batch[:, :3], 2)).float().sum()
        )
        if compare <= 0:
            return 100
        else:
            val_acc_list1.append(100 / (1 + (compare / size * 3)))
        inds_o_direction = torch.argmax(val_torch_outputs[:, 3:12], dim=1)
        inds_s_direction = torch.argmax(scaled_batch[:, 3:12], dim=1)
        inds_o_icon = torch.argmax(val_torch_outputs[:, 12:33], dim=1)
        inds_s_icon = torch.argmax(scaled_batch[:, 12:33], dim=1)
        inds_o_condition = torch.argmax(val_torch_outputs[:, 33:54], dim=1)
        inds_s_condition = torch.argmax(scaled_batch[:, 33:54], dim=1)
        val_acc_list2.append(
            (inds_o_direction == inds_s_direction).sum().item() / size * 100
        )
        val_acc_list3.append((inds_o_icon == inds_s_icon).sum().item() / size * 100)
        val_acc_list4.append(
            (inds_o_condition == inds_s_condition).sum().item() / size * 100
        )
        my_val_acc = torch.FloatTensor(
            val_acc_list1 + val_acc_list2 + val_acc_list3 + val_acc_list4
        ).to(device)
        my_val_acc = clean_torch(my_val_acc).mean()
        my_val_acc1 = torch.FloatTensor(val_acc_list1).to(device)
        my_val_acc1 = clean_torch(my_val_acc1).mean()
        my_val_acc2 = torch.FloatTensor(val_acc_list2).to(device)
        my_val_acc2 = clean_torch(my_val_acc2).mean()
        my_val_acc3 = torch.FloatTensor(val_acc_list3).to(device)
        my_val_acc3 = clean_torch(my_val_acc3).mean()
        my_val_acc4 = torch.FloatTensor(val_acc_list4).to(device)
        my_val_acc4 = clean_torch(my_val_acc4).mean()
    output = unscale_output(output=output)
    return output


# https://www.geeksforgeeks.org/python-last-occurrence-of-some-element-in-a-list/
def last_occurrence(lst, val):
    index = -1
    while True:
        try:
            index = lst.index(val, index + 1)
        except ValueError:
            return index


def future_prediction(model_name, device, id=[], hours=1):
    ids, train, label = gld.get_predictDataHourly(dt.now(), id=id)
    print(ids)
    uids = list(set(ids))
    print(train.shape, label.shape)
    model, optimizer, history = load_own_Model(model_name, device)
    if type(model) == "str":
        print("choose an other name")
        return []
    output_list = pd.DataFrame()
    for ui in uids:
        t = max(train[last_occurrence(ids, ui), :, -5]) + 1
        if t >= 24:
            t -= 24
        # print(train[last_occurrence(ids, ui), :, -5])
        output = prediction(model, train[last_occurrence(ids, ui)], [], device)
        output["ID"] = ui
        output["Time"] = t
        output_list = pd.concat([output_list, output])
    print(output_list)
    return output_list


def check_prediction(model_name, device, id=[], hours=1):
    train_tensors = Variable(torch.Tensor(train).to(device)).to(device)
    input_final = torch.reshape(train_tensors, (1, 1, train_tensors.shape[-1])).to(
        device
    )
    ids, train, label = gld.get_predictDataHourly(dt.now(), id=id)
    print(ids)
    uids = list(set(ids))
    print(train.shape, label.shape)
    model, optimizer, history = load_own_Model(model_name, device)
    if type(model) == "str":
        print("choose an other name")
        return []
    output_list = pd.DataFrame()
    for ui in uids:
        t = max(train[last_occurrence(ids, ui), :, -5]) + 1
        if t >= 24:
            t -= 24
        # print(train[last_occurrence(ids, ui), :, -5])
        output = prediction(model, train[last_occurrence(ids, ui)], [], device)
        output["ID"] = ui
        output["Time"] = t
        output_list = pd.concat([output_list, output])
    print(output_list)
    return output_list


# prediction main code specially for hourly models
def predictHourly(date, device, mode="normal", model_num=0, id="", city="", time=-1):
    out_list = []
    if date <= dt.now() - td(days=2):
        train, label, label24 = gld.get_predictDataHourly(date, id=id)
        if isinstance(train, str):
            print("error occured please retry with other ID/Name")
            return
        print("\ntraining count", train.shape)

        model, optimizer, loss_fn, metric, history = load_own_Model(
            "Hourly", device, loading_mode=mode, t=model_num
        )
        model24, optimizer24, loss_fn24, metric24, history24 = load_own_Model(
            "Hourly24", device, loading_mode=mode, t=model_num
        )
        if not id == "" or city == "":
            out_list.append(prediction(model, train[time], "Hourly", device))
            out_list.append(prediction(model24, train[time], "Hourly", device))
            plotting_Prediction_hourly(
                train[time],
                out_list,
                "test",
                hourly=label[time],
                hourly24=label24[time],
                mode=mode,
                t=model_num,
            )
    else:
        train = gld.get_predictDataHourly(date, id=id)
        if isinstance(train, str):
            print("error occured please retry with other ID/Name")
            return
        print("\ntraining count", train.shape)
        model, optimizer, loss_fn, metric, history = load_own_Model(
            "Hourly", device, loading_mode=mode, t=model_num
        )
        model24, optimizer24, loss_fn24, metric24, history24 = load_own_Model(
            "Hourly24", device, loading_mode=mode, t=model_num
        )
        if not id == "" or city == "":
            out_list.append(prediction(model, train[time], "Hourly", device))
            out_list.append(prediction(model24, train[time], "Hourly", device))
            plotting_Prediction_hourly(
                train[time], out_list, "future", mode=mode, t=model_num
            )


if __name__ == "__main__":
    device = check_cuda()
    outs = future_prediction(
        "working", device, id=["00020", "00044", "00096", "00294", "00757"]
    )

    im = show_image(outs)
