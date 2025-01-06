import matplotlib.pyplot as plt
import os
import json
from datetime import datetime as dt
from datetime import timedelta as td
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle

# import asyncio

# import random

# from sklearn import preprocessing as pr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score

# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchmetrics.classification import MulticlassConfusionMatrix
import seaborn as sns
import math
from progress.bar import Bar

# import warnings

plt.switch_backend("agg")

# warnings.simplefilter("ignore")
# os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses
# warnings.filterwarnings("ignore")

gam = 100
c_svm = 100
n_e = 3000
color = ["red", "grey", "blue"]
hidden_output_size = 0
# random_learning = True
batch_bar = Bar("Processing", max=20)


# checking if cuda is used. if not it choses cpu
def check_cuda():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("this device uses " + device + " to train data")
    return torch.device(device)


def convert_loss(loss):
    loss = float(loss) - 0.5
    if loss < 0:
        loss *= -1
    loss *= 2
    return loss


# creating my lstm
class PyTorch_LSTM(nn.Module):
    def __init__(self, inputs, outputs, h_size, device, layer=3, dropout=0):
        super(PyTorch_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=inputs,
            hidden_size=h_size,
            num_layers=layer,
            dropout=dropout,
            # batch_first=False,
        ).to(device)
        # self.fcl = nn.Linear(layer, 3).to(device)
        self.fc = nn.Linear(h_size, outputs).to(device)
        # self.soft = nn.Softmax().to(device)
        # self.sig = nn.Sigmoid().to(device)
        # self.relu = nn.ReLU().to(device)
        # self.hidden_state = Variable(torch.zeros((layer, 1, h_size)).to(device)).to(
        #     device
        # )
        # self.cell_state = Variable(torch.zeros((layer, 1, h_size)).to(device)).to(
        #     device
        # )
        self.hidden_state = Variable(
            torch.Tensor(np.zeros((layer, h_size)).tolist())
            .type(torch.float32)
            .to(device)
        ).to(device)
        self.cell_state = Variable(
            torch.Tensor(np.zeros((layer, h_size)).tolist())
            .type(torch.float32)
            .to(device)
        ).to(device)
        # self.hidden_state = Variable(
        #     torch.rand((layer, 1, h_size)).to(device)
        # ).to(device)
        # self.cell_state = Variable(torch.rand((layer, 1, h_size)).to(device)).to(
        #     device
        # )

    def show_hiddens(self):
        print(
            "hidden_state:\n",
            self.hidden_state.tolist(),
            "\ncell_state:\n",
            self.cell_state.tolist(),
        )

    def forward(self, input, device, hiddens=True):
        out, (hn, cn) = self.lstm(input, (self.hidden_state, self.cell_state))
        if hiddens:
            self.hidden_state = hn.detach()
            self.cell_state = cn.detach()
        # print("out", out.shape)
        l2 = self.fc(out)
        # print("l2", l2.shape)
        # out1 = self.fc(hn)
        # out1 = torch.reshape(out1, (1, out1.shape[0], out1.shape[1], out1.shape[2])).to(
        #     device
        # )
        # out2 = torch.cat(
        #     (
        #         torch.flatten(hn.type(torch.float64).to(device)),
        #         torch.flatten(out1.type(torch.float64).to(device)),
        #     )
        # )
        # print("out2", out2.shape)
        # print("out", new_out.shape)
        # if multi_target:
        #     return out1.type(torch.float64).to(device), out2.type(torch.float64).to(
        #         device
        #     )

        return l2.type(torch.float64).to(
            device
        )  # , out2.type(torch.float64).to(device)


# loading model if already saved or creating a new model
def load_own_Model(
    name,
    device,
    input_count,
    output_count,
    hiddensize=30,
    learning_rate=0,
    layer=3,
    weights=[],
    split_loss_number=0.5,
    floating_point=0.01,
    dropout=0.1,
    epoche="",
    optimizer_classes="adagrad",
    loss_classes="bce",
):
    global hidden_output_size, random_learning
    hidden_output_size = (layer * hiddensize) + layer

    history = {
        "accuracy": [0],
        "loss": [0.5],
        "val_accuracy": [0],
        "val_loss": [0.5],
        "argmax_accuracy": [0],
        "val_argmax_accuracy": [0],
        "svm_accuracy": [0],
        "val_svm_accuracy": [0],
        "rf_accuracy": [0],
        "val_rf_accuracy": [0],
        "gauss_accuracy": [0],
        "val_gauss_accuracy": [0],
        "gaussrbf_accuracy": [0],
        "val_gaussrbf_accuracy": [0],
        "nb_accuracy": [0],
        "val_nb_accuracy": [0],
        "C": [1],
    }
    model = {"haha": [1, 2, 3]}
    # model.show_hiddens()
    if os.path.exists(f"./Models/{name}_history.json"):
        with open(f"./Models/{name}_history.json", "r") as f:
            history = json.load(f)
    if os.path.exists(f"./Models/{name}.pth"):
        print("Model found")
        # model = PyTorch_LSTM(input_count, hiddensize, history, device, layer=layer)
        # model.load_state_dict(torch.load(f"./Models/{name}.pth", map_location=device))
        # model.eval()
        model = torch.load(f"./Models/{name}.pth", map_location=device)
        model.train()
        # model.show_hiddens()
    else:
        model = PyTorch_LSTM(
            input_count,
            output_count,
            hiddensize,
            device,
            layer=layer,
            dropout=dropout,
        )
    print(model.show_hiddens())

    svm = SVC(kernel="rbf", gamma=gam, C=c_svm)
    if os.path.exists(f"./Models/{name}_svm.sav"):
        svm = pickle.load(open(f"./Models/{name}_svm.sav", "rb"))
        print("SVM found")
    else:
        print("SVM not found or not complete")
        svm.fit(
            np.random.rand(3, hidden_output_size),
            # np.random.rand(3, hiddensize * layer + layer),
            np.array([1, 2, 3]),
        )

    # gauss = GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)
    # if os.path.exists(f"./Models/{name}_gauss.sav"):
    #     if not training:
    #         gauss = pickle.load(open(f"./Models/{name}_gauss.sav", "rb"))
    #     print("gauss found")
    # else:
    #     print("gauss not found or not complete")
    #     gauss.fit(
    #         np.random.rand(3, hidden_output_size),
    #         # np.random.rand(3, hiddensize * layer + layer),
    #         np.array([1, 2, 3]),
    #     )

    # gaussrbf = GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)
    # if os.path.exists(f"./Models/{name}_gaussrbf.sav"):
    #     if not training:
    #         gaussrbf = pickle.load(open(f"./Models/{name}_gaussrbf.sav", "rb"))
    #     print("gaussrbf found")
    # else:
    #     print("gaussrbf not found or not complete")
    #     gaussrbf.fit(
    #         np.random.rand(3, hidden_output_size),
    #         # np.random.rand(3, hiddensize * layer + layer),
    #         np.array([1, 2, 3]),
    #     )

    # nb = GaussianNB()
    # if os.path.exists(f"./Models/{name}_nb.sav"):
    #     if not training:
    #         nb = pickle.load(open(f"./Models/{name}_nb.sav", "rb"))
    #     print("NB found")
    # else:
    #     print("NB not found or not complete")
    #     nb.fit(
    #         np.random.rand(3, hidden_output_size),
    #         # np.random.rand(3, hiddensize * layer + layer),
    #         np.array([1, 2, 3]),
    #     )

    # rf = RandomForestClassifier(n_estimators=n_e, random_state=42, max_depth=3)
    # if os.path.exists(f"./Models/{name}_rf.sav"):
    #     if not training:
    #         rf = pickle.load(open(f"./Models/{name}_rf.sav", "rb"))
    #     # print("RF found")
    # else:
    #     print("RF not found or not complete")
    #     rf.fit(
    #         np.random.rand(3, hidden_output_size),
    #         np.array([1, 2, 3]),
    #     )

    # actual_loss = (history["loss"][-1] - 0.5) * 2
    actual_loss = history["loss"][-1]
    if len(history["loss"]) > 150:
        actual_loss = min(history["loss"][-150:])
    # print(actual_loss)

    if learning_rate == 0:
        if str(epoche) == "0":
            learning_rate = 0.00000001 * floating_point
        actual_loss = history["loss"][-1]
        if split_loss_number <= 0.01:
            actual_loss *= 10 ** (len(str(split_loss_number)) - 2)
            split_loss_number *= 10 ** (len(str(split_loss_number)) - 2)
        # if actual_loss >= 0.8:
        # learning_rate = 0.8
        # elif history["loss"][-1] >= 0.7:
        #     learning_rate = 0.05

        print("floating point:", floating_point)
        if (
            actual_loss < 1
            and actual_loss >= split_loss_number
            and split_loss_number < 1
        ):
            learning_rate = floating_point * (actual_loss - (split_loss_number - 0.1))
        elif (
            actual_loss <= split_loss_number
            and actual_loss > 0
            and split_loss_number < 1
        ):
            # if actual_loss < 0.35:
            #     random_learning = False
            learning_rate = (
                (floating_point / 10)
                / (
                    (
                        10 ** int((split_loss_number * 10))
                        - (math.floor(actual_loss * 10))
                    )
                )
                * ((actual_loss - (math.floor(actual_loss * 10) / 10)) * 100)
            )
            # print((floating_point / 10)
            #         )
            # print("/ ",((10 ** int((split_loss_number * 10)) - (math.floor(actual_loss * 10))))
            #         )
            # print("* ", ((actual_loss - (math.floor(actual_loss * 10) / 10)) * 100))
            # print("= ",(floating_point / 10)
            #         / ((10 ** int((split_loss_number * 10)) - (math.floor(actual_loss * 10))))
            #         * ((actual_loss - (math.floor(actual_loss * 10) / 10)) * 100))
        elif (
            actual_loss <= (10 + split_loss_number) and actual_loss >= split_loss_number
        ):  # and random_learning == True:
            learning_rate = floating_point * (actual_loss - (split_loss_number - 1))
        elif actual_loss < split_loss_number and actual_loss > 0:
            # if learning_rate < 0.5:
            #     random_learning = False
            learning_rate = (
                floating_point
                / (10 ** int(split_loss_number - (math.floor(actual_loss))))
                * ((actual_loss - math.floor(actual_loss)) * 10)
            )
            # print(floating_point)
            # print("/ ", (10 ** int(split_loss_number - (math.floor(actual_loss)))))
            # print("* ", ((actual_loss - math.floor(actual_loss)) * 10))
            # print("= ",floating_point
            #     / (10 ** int(5 - (math.floor(actual_loss))))
            #     * ((actual_loss - math.floor(actual_loss)) * 10))
        # elif history["loss"][-1] <= 5:
        #     print(history["loss"][-1])
        #     lossr = 0.01 / (10 ** int(5 - (math.floor(history["loss"][-1]))))
        #     print("2")
        else:
            learning_rate = 0.00000001 * floating_point

        print("loss:", actual_loss, " ,learn rate:", learning_rate)
    print(
        "learn rate:",
        learning_rate,
        "\n",
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if optimizer_classes.lower() == "adamax":
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
    elif optimizer_classes.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_classes.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    elif optimizer_classes.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_classes.lower() == "nadam":
        optimizer = optim.NAdam(model.parameters(), lr=learning_rate)
    elif optimizer_classes.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_classes.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    model.train()
    print("Model is running in Training mode")
    loss_fn = nn.CrossEntropyLoss().to(device)

    if len(weights) != 0:
        # weight_tensor = torch.reshape(torch.tensor(weights).to(device), (1, 3)).to(
        #     device
        # )
        weight_tensor = torch.tensor(weights).to(device)
        if loss_classes.lower() == "bce":
            loss_fn = nn.BCEWithLogitsLoss(weight=weight_tensor).to(device)
        if loss_classes.lower() == "l1":
            loss_fn = nn.L1Loss(weight=weight_tensor).to(device)
        if loss_classes.lower() == "nll":
            loss_fn = nn.NLLLoss(weight=weight_tensor).to(device)
        else:
            loss_fn = nn.CrossEntropyLoss(weight=weight_tensor).to(device)

        del weight_tensor
    else:
        if loss_classes.lower() == "bce":
            loss_fn = nn.BCEWithLogitsLoss().to(device)
        if loss_classes.lower() == "l1":
            loss_fn = nn.L1Loss().to(device)
        if loss_classes.lower() == "nll":
            loss_fn = nn.NLLLoss().to(device)

    # metric = MulticlassConfusionMatrix(num_classes=3).to(device)
    return (
        model,
        [[]],  # svm],  # rf],  # gauss, gaussrbf, nb],
        ["argmax"],  # "svm"],  # "rf"],  # "gauss", "gaussrbf", "nb"],
        optimizer,
        loss_fn,
        # metric,
        history,
    )


# saving model if saving_mode set to ts or timestamp it will use the number for the model to save it.
# ts helps to choose saved model data before the model started to overfit or not work anymore
def save_own_Model(
    name,
    history,
    model,
    classificators,
    cls_names,
    only_cls=False,
    save_classificators=False,
):
    if save_classificators:
        for i in range(1, len(classificators)):
            pickle.dump(
                classificators[i], open(f"./Models/{name}_{cls_names[i]}.sav", "wb")
            )
            # print(f"./Models/{name}_{cls_names[i]}.sav")
        print("Saved Classifications\n")
    else:
        print("Not allowed to save Classifications\n")

    if not only_cls:
        with open(f"./Models/{name}_history.json", "w") as fp:
            json.dump(history, fp)
        # torch.save(model.state_dict(), f"./Models/{name}.pth")
        torch.save(model, f"./Models/{name}.pth")
        history["loss"] = list(filter(lambda num: num != 0, history["loss"]))
        history["val_loss"] = list(filter(lambda num: num != 0, history["val_loss"]))
        if len(history["loss"]) > 1:
            # print(
            #     "save val_loss if ",
            #     min(history["val_loss"][:-1]),
            #     " >= ",
            #     history["val_loss"][-1],
            # )
            # if min(history["val_loss"][:-1]) >= history["val_loss"][-1]:
            #     print("new best val_loss")
            #     torch.save(model, f"./Models/{name}_min_val_Loss.pth")
            # print(
            #     "save loss if ", min(history["loss"][:-1]), " >= ", history["loss"][-1]
            # )
            if min(history["loss"][:-1]) >= history["loss"][-1]:
                print("new best loss")
                torch.save(model, f"./Models/{name}_min_Loss.pth")

        history["argmax_accuracy"] = list(
            filter(lambda num: num != 0, history["argmax_accuracy"])
        )
        history["val_argmax_accuracy"] = list(
            filter(lambda num: num != 0, history["val_argmax_accuracy"])
        )
        if len(history["argmax_accuracy"]) > 1:
            # print(
            #     "save argmax_accuracy if ",
            #     max(history["argmax_accuracy"][:-1]),
            #     " <= ",
            #     history["argmax_accuracy"][-1],
            # )
            if max(history["argmax_accuracy"][:-1]) <= history["argmax_accuracy"][-1]:
                print("new best argmax_accuracy")
                torch.save(model, f"./Models/{name}_max_Accuracy.pth")
        # print("Saved model\n\n")


def set_weights(label):
    weights = []
    min_val = 100000000
    for i in range(0, 3):
        weights.append(np.count_nonzero(label == i))
        if weights[i] < min_val:
            min_val = weights[i]
    for i in range(0, 3):
        weights[i] = min_val / weights[i]
    print("Weights setted")
    return weights


def norm(matrix):
    # matrix **= 2
    # matrix = np.square(matrix)
    max = np.amax(matrix)
    return matrix / max


def approximation(all):
    # print(max(all), min(all))
    # nall, max = norm(all)
    x = np.arange(0, len(all))
    y = list(all)
    [b, m] = np.polynomial.polynomial.polyfit(x, y, 1)
    # print(b,m)
    return b, m
    # return round(np.angle(1 + (m * 1j), deg=True), 5)


def norm_np(matrix):
    max = np.reshape(np.amax(matrix, axis=1), (matrix.shape[0], -1))
    return matrix / max


def approximation_np(all):
    nall = norm_np(all)
    flipped = np.swapaxes(nall, 0, 1)
    x = np.arange(0, flipped.shape[0])
    [b, m] = np.polynomial.polynomial.polyfit(x, flipped, 1)
    col = np.column_stack((b, m))
    return col


def stack_f(arr, stack=True):
    # arr muss durch 10 und durch 3 teilbar sein
    approx_arr = approximation_np(arr)
    for i in range(int(arr.shape[1] / 10)):
        approx_arr = np.column_stack(
            (approx_arr, approximation_np(arr[:, i * 10 : (i + 1) * 10]))
        )
    for i in range(3):
        approx_arr = np.column_stack(
            (
                approx_arr,
                approximation_np(
                    arr[
                        :,
                        int(i * (arr.shape[1] / 3)) : int((i + 1) * (arr.shape[1] / 3)),
                    ]
                ),
            )
        )
    if stack:
        return np.column_stack((arr, approx_arr))
    else:
        return approx_arr


def buysell_logic(arr, under_half=False):
    if len(arr) < 1:
        return []
    # print(buy)
    # print(sell)
    jumper = 0
    jvalue = min(arr[:, 1])
    condition_run = max(arr[:, 1])
    jnp = np.array(arr[arr[:, 1] == jvalue])
    # print(jnp)
    ix = 0

    while jvalue < condition_run:
        # print(jvalue)
        jumper += 1
        if ix < 30:
            ix = 30
        jx = ix - 100
        if jx < 0:
            jx = 0
        # Buy
        matching = []
        if jumper % 2 == 1:
            if under_half:
                matching = arr[
                    (arr[:, 1] > jvalue)
                    & (arr[:, 0] == 0)
                    & (arr[:, 2] < arr[jx:ix, 2].mean())
                ]
            else:
                matching = arr[(arr[:, 1] > jvalue) & (arr[:, 0] == 0)]

        # Sell
        else:
            if jnp[-1, 2] == 0:
                jnp[-1, 2] == 0.000001
            bought = (
                jnp[-1, 2] + (jnp[-1, 2] * 0.05) + (1 / math.ceil(100 / jnp[-1, 2]))
            )
            if under_half:
                matching = arr[
                    (arr[:, 1] > jvalue)
                    & (arr[:, 0] == 2)
                    # & (arr[:, 2] - (arr[:, 2] / 0.05) > jnp[-1, 2])
                    & (arr[:, 2] > bought)
                    & (arr[:, 2] > arr[jx:ix, 2].mean())
                ]
            else:
                matching = arr[
                    (arr[:, 1] > jvalue)
                    & (arr[:, 0] == 2)
                    # & (arr[:, 2] - (arr[:, 2] / 0.05) > jnp[-1, 2])
                    & (arr[:, 2] > bought)
                ]

        if len(matching) > 0:
            jvalue = min(matching[:, 1])
            ix = np.where(matching[:, 1] == jvalue)[0][0]
            jnp = np.concatenate([jnp, arr[arr[:, 1] == jvalue]])
        else:
            return jnp
    return jnp


def buysell_calc(arr):
    v = 0
    counts = 0
    bought = 0
    for label, time, val in arr:
        if val > 0:
            if label == 0:
                counts = math.ceil(100 / (val + (val * 0.05)))
                bought = (counts * (val + (val * 0.05))) + 1
                # v -= bought
            elif label == 2:
                v += (counts * (val + (val * 0.05))) - bought - 1
    return v


# plotting Evaluation via Accuracy, Loss and MulticlassConfusionMatrix
def plotting_hist(
    history,
    metrics,
    metric_names,
    name,
    epoche=0,
    p1=True,
    p2=True,
    dimension=2,
    min_amount=3,
):
    icons = ["buy", "hold", "sell"]
    # ["svm train", "svm test", "rf train", "rf test"]
    name_tag = f"{name}_plot"
    if p1:
        # summarize history for accuracy

        fig, ax = plt.subplots(4, figsize=(20, 5), sharey="row")
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        # ax[0].plot(history["accuracy"], alpha=0.6)
        # ax[0].plot(history["val_accuracy"], alpha=0.6)
        ax[0].plot(history["argmax_accuracy"], alpha=0.6)
        ax[0].plot(history["val_argmax_accuracy"], alpha=0.6)
        # ax[0].plot(history["svm_accuracy"], alpha=0.6)
        # ax[0].plot(history["val_svm_accuracy"], alpha=0.4)
        # ax[0].plot(history["rf_accuracy"], alpha=0.6)
        # ax[0].plot(history["val_rf_accuracy"], alpha=0.6)
        if len(history["accuracy"]) > 150:
            # ax[1].plot(history["accuracy"][-150:], alpha=0.6)
            # ax[1].plot(history["val_accuracy"][-150:], alpha=0.6)
            ax[1].plot(history["argmax_accuracy"][-150:], alpha=0.6)
            ax[1].plot(history["val_argmax_accuracy"][-150:], alpha=0.6)
            # ax[1].plot(history["svm_accuracy"][-150:], alpha=0.6)
            # ax[1].plot(history["val_svm_accuracy"][-150:], alpha=0.4)
            # ax[1].plot(history["rf_accuracy"][-150:], alpha=0.6)
            # ax[1].plot(history["val_rf_accuracy"][-150:], alpha=0.6)
        if len(history["accuracy"]) > min_amount:
            ax[2].plot(history["argmax_accuracy"][(-1 * min_amount) :])
            ax[3].plot(history["val_argmax_accuracy"][(-1 * min_amount) :])
            # ax[2].plot(history["argmax_accuracy"], alpha=0.6)
            # ax[2].plot(history["val_argmax_accuracy"], alpha=0.6)
            # ax[2].plot(history["svm_accuracy"][-3:], alpha=0.6)
            # ax[2].plot(history["val_svm_accuracy"][-3:], alpha=0.6)
            # ax[2].plot(history["rf_accuracy"][-3:], alpha=0.6)
            # ax[2].plot(history["val_rf_accuracy"][-3:], alpha=0.6)
        ax[0].grid(axis="y")
        ax[1].grid(axis="y")
        ax[2].grid(axis="y")
        ax[3].grid(axis="y")
        ax[0].legend(
            [
                # "train",
                # "test",
                "argmax train",
                "argmax test",
                "svm train",
                "svm test",
            ],
            # ["svm train", "svm test", "rf train", "rf test"],
            loc="upper left",
        )
        ax[1].legend(
            [
                # "train",
                # "test",
                "argmax train",
                "argmax test",
                "svm train",
                "svm test",
            ],
            # ["svm train", "svm test", "rf train", "rf test"],
            loc="upper left",
        )
        plt.savefig(f"./Plots/{name_tag}_accuracy.png")
        plt.close()

        history["loss"] = (pd.Series(history["loss"]) * 1000).tolist()
        history["val_loss"] = (pd.Series(history["val_loss"]) * 1000).tolist()
        history["loss"] = [_ if _ < 10 else 10 for _ in history["loss"]]
        history["val_loss"] = [_ if _ < 10 else 10 for _ in history["val_loss"]]

        b, m = approximation(history["loss"])
        f = [
            # (b / 100) + ((m * x) / len(history["loss"]))
            b + (m * x)
            for x in range(len(history["loss"]))
        ]
        b2, m2 = approximation(history["loss"][-150:])
        f2 = [
            # (b / 100) + ((m * x) / len(history["loss"]))
            b2 + (m2 * x)
            for x in range(150)
        ]
        # summarize history for loss
        fig, ax = plt.subplots(4, figsize=(20, 5), sharey="row")

        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        ax[0].plot(history["loss"], alpha=0.8)
        ax[0].plot(history["val_loss"], alpha=0.75)
        ax[0].plot(f)
        ax[0].grid(axis="y")
        ax[0].legend(
            [
                "train / Epoche:" + str(epoche + 1),
                "test",
                "linear train: {:.1f} + {:.5f}x".format(b * 10, m),  # * 10000),
            ],
            loc="upper left",
        )
        ax[1].plot(history["loss"][-150:], alpha=0.8)
        ax[1].plot(history["val_loss"][-150:], alpha=0.75)
        ax[1].plot(f2)
        ax[1].grid(axis="y")
        ax[1].legend(
            [
                "train",
                "test",
                "linear train: {:.1f} + {:.5f}x".format(b2 * 10, m2),  # * 10000),
            ],
            loc="upper left",
        )
        ax[2].plot(history["loss"][(-1 * min_amount) :])
        ax[2].grid(axis="y")
        ax[3].plot(history["val_loss"][(-1 * min_amount) :])
        ax[3].grid(axis="y")
        plt.savefig(f"./Plots/{name_tag}_loss.png")
        plt.close()
    if p2:
        # summarize MulticlassConfusionMatrix
        sz = (dimension * 2) - 1
        fig, ax = plt.subplots(
            int(len(metrics) / dimension),
            dimension,
            figsize=(sz, 7),
            sharey="row",
        )
        plt.title("Confusion Matrix")
        for i, metric in enumerate(metrics):
            j = math.floor(i / dimension)
            k = i % dimension
            if dimension == 1:
                ax[i].title.set_text(metric_names[i])
                metric.plot(
                    ax=ax[i],
                    # cmap="Blues",
                    # colorbar=False,
                )
                ax[i].xaxis.set_ticklabels(icons)
                ax[i].yaxis.set_ticklabels(icons)
            else:
                ax[j, k].title.set_text(metric_names[i])
                metric.plot(
                    ax=ax[j, k],
                    # cmap="Blues",
                    # colorbar=False,
                )
                ax[j, k].xaxis.set_ticklabels(icons)
                ax[j, k].yaxis.set_ticklabels(icons)
        fig.set_figwidth(10)
        fig.set_figheight(10)
        # ax.set_xlabel("Predicted labels")
        # ax.set_ylabel("True labels")
        plt.savefig(f"./Plots/{name}_matrix.png")
        plt.close()


def plotting_plot2(x, ys, name="plot"):
    fig, ax = plt.subplots(int(len(ys)), figsize=(15, int(len(ys) * 5)), sharey="row")
    plt.title(f"{name} accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("ISIN_counts")
    if len(ys) == 1:
        ax.scatter(x, ys[0])
    else:
        for i, y in enumerate(ys):
            ax[i].scatter(x, y)
    plt.savefig(f"./Plots/{name}_acc_plot.png")
    plt.close()


def plotting_scatter(outputs, labels, predicted=[]):
    fig, ax = plt.subplots(
        len(predicted), 3, figsize=(10, 10), subplot_kw=dict(projection="3d")
    )

    # print("outputs", outputs.shape)
    pca = PCA(
        n_components=3,
    )

    principalComponents = pca.fit_transform(outputs)
    # print("principalComponents", principalComponents.shape)

    xl = [45, 225, 135]
    yl = [20, 20, 20]
    zl = [0, 0, 0]

    for i in range(3):
        ax[0, i].scatter(
            principalComponents[:, 0],
            principalComponents[:, 1],
            principalComponents[:, 2],
            c=labels[:],
            cmap=plt.cm.coolwarm,
        )
        ax[0, i].view_init(yl[i], xl[i], zl[i])
        # if use_svm:
        ax[1, i].scatter(
            principalComponents[:, 0],
            principalComponents[:, 1],
            principalComponents[:, 2],
            c=predicted[:],
            cmap=plt.cm.coolwarm,
        )
        ax[1, i].view_init(yl[i], xl[i], zl[i])

        # ax[i].legend(["buy", "hold", "sell"])
        # plt.xticks(rotation=90 * i, ha="right")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")

    plt.title("SVC")

    # pickle.dump(fig, open("./Plots/fig1.pkl", "wb"))

    plt.savefig("./Plots/testing_plot.png", dpi="figure", format=None)

    # plt.show()
    plt.close()


def plotting_scatter_2d(outputs, labels, predicted=[], name="testing"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # , subplot_kw=dict(projection="3d"))

    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(outputs)
    # print("principalComponents", principalComponents.shape)

    y_max = sorted(principalComponents[:, 1])[-15] + 0.001
    y_min = min(principalComponents[:, 1]) - 0.001
    x_min = min(principalComponents[:, 0]) - 0.005
    x_max = sorted(principalComponents[:, 0])[-15] + 0.005
    # c = ["blue", "red", "grey"]
    # c = ["blue", "grey", "red"]

    ax[0].scatter(
        principalComponents[:, 0],
        principalComponents[:, 1],
        c=[color[int(i)] for i in labels[:]],
        alpha=0.4,
    )
    ax[0].set_xlim(x_min, x_max)
    ax[0].set_ylim(y_min, y_max)
    # l, count = np.unique(labels, return_counts=True)
    # _, ls = np.unique(labels, return_counts=False, return_index=True)
    l = np.unique(labels, return_counts=False, return_index=False).tolist()
    # l = [labels[index] for index in sorted(ls.tolist())]
    # _, ps = np.unique(predicted, return_counts=False, return_index=True)
    p = np.unique(predicted, return_counts=False, return_index=False).tolist()
    # p = [labels[index] for index in sorted(ps.tolist())]
    # print(l, p)
    # handles, labels = ax[0].gca().get_legend_handles_labels()
    # print(handles, labels)
    # print(l, "\n", labels, "\n", ls.tolist())
    h = []
    if 0 in l:
        h.append(mpatches.Patch(color=color[0], label="Buy"))
    if 1 in l:
        h.append(mpatches.Patch(color=color[1], label="Hold"))
    if 2 in l:
        h.append(mpatches.Patch(color=color[2], label="Sell"))
    # ax[0].legend()
    ax[0].legend(handles=h)
    # if use_svm:
    ax[1].scatter(
        principalComponents[:, 0],
        principalComponents[:, 1],
        alpha=0.5,
        c=[color[int(i)] for i in predicted[:]],
    )
    # plt.colorbar(ax[0].get_children()[2], ax=ax[0])
    # plt.colorbar(ax[1].get_children()[2], ax=ax[1])
    ax[1].set_xlim(x_min, x_max)
    ax[1].set_ylim(y_min, y_max)

    h2 = []
    if 0 in p:
        h2.append(mpatches.Patch(color=color[0], label="Buy"))
    if 1 in p:
        h2.append(mpatches.Patch(color=color[1], label="Hold"))
    if 2 in p:
        h2.append(mpatches.Patch(color=color[2], label="Sell"))
    # ax[0].legend()
    ax[1].legend(handles=h)

    # print("outs:", outputs.shape)
    # print("pcas:", principalComponents.shape)
    # ax[1].legend(handles=["buy", "hold", "sell"])
    # ax[1].legend()
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.title("SVC")

    # pickle.dump(fig, open("./Plots/fig1.pkl", "wb"))

    plt.savefig(f"./Plots/{name}_scatterplot.png", dpi="figure", format=None)

    # plt.show()
    plt.close()


def plotting_timeseries(label, predictions, cls_names=[], plot_name="timeseries_plot"):
    names = ["How it schould be"]
    for cls_n in cls_names:
        names.append(f"What {cls_n} makes correct")
        names.append(f"How it is in {cls_n}")
    color2 = ["green", "orange", "red"]
    print("Timeseries Plots: ", 1 + (len(cls_names) * 2))
    fig, ax = plt.subplots(
        1 + (len(cls_names) * 2), figsize=(20, (1 + (len(cls_names)) * 2) * 2)
    )

    arr = buysell_logic(label)
    win = buysell_calc(arr)
    arr2 = buysell_logic(label, under_half=True)
    win2 = buysell_calc(arr2)
    ax[0].title.set_text(names[0] + f", Win: {round(win, 2)}, Win2: {round(win2, 2)}")
    ax[0].title.set_size(24)
    ax[0].scatter(
        label[:, 1],
        label[:, 2],
        alpha=0.5,
        c=[color[int(i)] for i in label[:, 0]],
    )
    ax[0].plot(
        arr[:, 1],
        arr[:, 2],
    )
    ax[0].plot(
        arr2[:, 1],
        arr2[:, 2],
    )
    # ax[0].scatter(
    #     arr[:, 1],
    #     arr[:, 2],
    #     alpha=0.5,
    #     c=[color2[i] for i in arr[:, 0]],
    # )
    ax[0].grid(axis="y")

    winsp = []
    winsp2 = []

    for pi, predicted in enumerate(predictions):
        labelf = label.copy()
        ax[(pi * 2) + 1].title.set_text(names[(pi * 2) + 1])
        ax[(pi * 2) + 1].title.set_size(24)

        l = labelf[:, 0] - predicted
        l[l < 0] *= -1
        ax[(pi * 2) + 1].scatter(
            labelf[:, 1],
            labelf[:, 2],
            alpha=0.5,
            c=[color2[int(i)] for i in l],
        )
        ax[(pi * 2) + 1].grid(axis="y")

        bs_label = labelf.copy()
        bs_label[:, 0] = predicted
        arr = buysell_logic(bs_label)
        win_n = buysell_calc(arr)
        arr2 = buysell_logic(bs_label, under_half=True)
        win2_n = buysell_calc(arr2)
        ax[(pi * 2) + 2].title.set_text(
            names[(pi * 2) + 2] + f", Win: {round(win_n, 2)}, Win2: {round(win2_n, 2)}"
        )
        ax[(pi * 2) + 2].title.set_size(24)
        ax[(pi * 2) + 2].scatter(
            labelf[:, 1],
            labelf[:, 2],
            alpha=0.5,
            c=[color[int(i)] for i in predicted],
        )
        ax[(pi * 2) + 2].plot(
            arr[:, 1],
            arr[:, 2],
        )
        ax[(pi * 2) + 2].plot(
            arr2[:, 1],
            arr2[:, 2],
        )
        # ax[(pi * 2) + 2].scatter(
        #     arr[:, 1],
        #     arr[:, 2],
        #     alpha=0.5,
        #     c=[color2[i] for i in arr[:, 0]],
        # )
        ax[(pi * 2) + 2].grid(axis="y")

        if win_n > 0 and win > 0:
            winsp.append((win_n / win) * 100)
        else:
            winsp.append(0)
        if win2_n > 0 and win2 > 0:
            winsp2.append((win2_n / win2) * 100)
        else:
            winsp2.append(0)

    plt.savefig(f"./Plots/{plot_name}.png", dpi="figure", format=None)
    plt.close()

    return winsp, winsp2


def plotting_correlation(corr):
    # corr_matrix = np.corrcoef(corr)
    corr_matrix = pd.DataFrame(
        corr, columns=[str(i) for i in range(corr.shape[1])]
    ).corr()
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        corr_matrix,
        annot=False,
        # fmt=".2f",
        cmap="coolwarm",
        cbar_kws={"label": "Correlation coefficient"},
        square=True,
    )
    plt.title("Heatmap of the feature correlations clean")
    plt.xticks(
        rotation=45, ha="right"
    )  # Rotates the X-axis labels for better readability
    plt.tight_layout()  # Adjusts the layout to avoid overlaps
    plt.savefig("./Plots/correlation_plot.png", dpi="figure", format=None)
    # plt.show()
    plt.close()


# def unscale_output(out):
#     out *= 2
#     if out >= 0.5:
#         out[0] = 2
#     elif out <= -0.5:
#         out[0] = 0
#     else:
#         out[0] = 1
#     return out.type(torch.float64)


def check_classes(predicted):
    calc = 0
    if 0 in predicted:
        calc += 1
    if 1 in predicted:
        calc += 1
    if 2 in predicted:
        calc += 1
    return calc


def create_label(outputs, label, device, classes=3):
    # print("output: ", outputs.shape, "\nlabel: ", label.shape)
    l = torch.zeros(classes, outputs.shape[-1]).type(torch.float64).to(device)
    for c in range(classes):
        o = outputs[label[:, 0, 0] == c, :, :].detach().clone()
        for i in range(o.shape[-1]):
            l[c, i] = o[:, :, i].mean().item()

    l = clean_torch(l)
    new_l = torch.zeros(outputs.shape).type(torch.float64).to(device)

    for i in range(label.shape[0]):
        new_l[i, :, :] = l[int(label[i, :, :].item())]

    return new_l, l
    # torch.reshape(new_l, (outputs.shape[0], label.shape[1], outputs.shape[-1])).to(
    #     device
    # )


def gat_new_label(l, outputs, label, device, classes=3):
    l = clean_torch(l)
    new_l = torch.zeros(outputs.shape).type(torch.float64).to(device)

    for i in range(label.shape[0]):
        new_l[i, :, :] = l[int(label[i, :, :].item())]

    return new_l
    # torch.reshape(new_l, (outputs.shape[0], label.shape[1], outputs.shape[-1])).to(
    #     device
    # )


def scale_label(labels, device):
    l = []
    for label in labels:
        if label == 0:
            l.append([1, 0, 0])
        elif label == 1:
            l.append([0, 1, 0])
        elif label == 2:
            l.append([0, 0, 1])
        else:
            print("error- label: ", label)
            l.append([0, 0, 0])
    # return

    return torch.reshape(
        torch.Tensor(l).type(torch.float64).to(device), (len(l), 3)
    ).to(device)


def clean_torch(val):
    val[val != val] = 0.0
    return val


def add_metric_row(
    classifications,
    cls_names,
    cls_state_names,
    outputs,
    labels,
    max,
    history,
    metrics,
    row,
    device,
    name="",
    name_end="",
    return_predictions=False,
    fit_cls=False,
    show_pr=True,
    save_hist=True,
    fit_name="Unknown_Name",
    predicts=[],
    max_acc_list=[],
):
    max_acc = accuracy_score(labels, max) * 100
    max_acc_list[0].append(max_acc)
    if show_pr:
        print(
            "\n\n"
            + cls_state_names[len(classifications) * row]
            + " -> Accuracy: {:.12f}".format(np.mean(max_acc_list[0]))
        )

    metrics[len(classifications) * row].update(
        torch.Tensor(max).to(device), torch.Tensor(labels).to(device)
    )
    if save_hist:
        history[f"{name}{cls_names[0]}{name_end}"].append(
            float(np.mean(max_acc_list[0]))
        )

    for i in range(1, len(classifications)):
        tstart = dt.now()
        cc = check_classes(labels)
        if fit_cls and cc >= 3:
            classifications[i] = classifications[i].fit(outputs, labels)
            pickle.dump(
                classifications[i],
                open(f"./Models/{fit_name}_{cls_names[i]}.sav", "wb"),
            )
        predicted_cls = classifications[i].predict(outputs)
        predicts[i - 1].extend(predicted_cls)
        # classes = check_classes(predicted)
        cls_acc = accuracy_score(labels, predicted_cls) * 100
        max_acc_list[i].append(cls_acc)
        classes_cls = check_classes(predicted_cls)
        if show_pr:
            print(
                cls_state_names[len(classifications) * row + i],
                "-> Accuracy: {:.12f}, Classes: {}".format(
                    np.mean(max_acc_list[i]), classes_cls
                ),
            )
        if save_hist:
            if classes_cls == 3:
                history[f"{name}{cls_names[i]}{name_end}"].append(float(cls_acc))
            else:
                history[f"{name}{cls_names[i]}{name_end}"].append(
                    history[f"{name}{cls_names[i]}{name_end}"][-1]
                )

        metrics[len(classifications) * row + i].update(
            torch.Tensor(predicted_cls).to(device), torch.Tensor(labels).to(device)
        )

        tend = dt.now() - tstart
        if show_pr:
            if tend > td(seconds=10):
                print("       -> ", tend)

    if return_predictions:
        return metrics, predicts
    return metrics


# whole training of the LSTM features and labels need to be 2d shaped
def train_LSTM(
    feature,
    label,
    model,
    classificators,
    cls_names,
    optimizer,
    loss_fn,
    # metric,
    history,
    model_kind,
    device,
    epoch_count=1,
    epoch=0,
    batchsize=0,
    use_svm=True,
    use_plot=True,
    name="Unknown_Name",
    extra_train=False,
):
    n_state = ["train", "test"]
    cls_state_names = []
    for j in n_state:
        for i in cls_names:
            cls_state_names.append(f"{j} {i}")

    scaled = Normalizer().fit_transform(feature[~np.isnan(feature).any(axis=1)])
    new_scaled = np.zeros((1, 1))
    if model_kind == 0:
        new_scaled = stack_f(scaled)  # normal
    elif model_kind == 1:
        new_scaled = scaled  # Second
    elif model_kind == 2:
        new_scaled = stack_f(scaled, stack=False)  # Third
    else:
        print("Error, newscaled gescheitert")
        return

    print("New training count: ", new_scaled.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        new_scaled, label, test_size=0.04, random_state=42
    )
    del new_scaled

    print("X_train_tensors", X_train.shape)
    print("X_test_tensors", X_test.shape)
    print("y_train_tensors", y_train.shape)
    print("y_test_tensors", y_test.shape)
    X_train_tensors = Variable(torch.Tensor(X_train).to(device)).to(device)
    X_test_tensors = Variable(torch.Tensor(X_test).to(device)).to(device)
    # y_train_tensors = Variable(torch.Tensor(y_train[:, 0].tolist())).to(device)
    # y_test_tensors = Variable(torch.Tensor(y_test[:, 0].tolist())).to(device)

    X_train_tensors = torch.reshape(
        X_train_tensors,
        (
            X_train_tensors.shape[0],
            X_train_tensors.shape[-1],
        ),
    ).to(device)
    X_test_tensors = torch.reshape(
        X_test_tensors,
        (
            X_test_tensors.shape[0],
            X_test_tensors.shape[-1],
        ),
    ).to(device)

    # y_train_tensors = (
    #     torch.reshape(y_train_tensors, (y_train_tensors.shape[0], 1, 1))
    #     .to(device)
    #     .type(torch.float64)
    #     .to(device)
    # )
    # y_test_tensors = (
    #     torch.reshape(y_test_tensors, (y_test_tensors.shape[0], 1, 1, 1))
    #     .to(device)
    #     .type(torch.float64)
    #     .to(device)
    # )

    X_train_tensors = clean_torch(X_train_tensors)
    X_test_tensors = clean_torch(X_test_tensors)
    # y_train_tensors = clean_torch(y_train_tensors)
    # y_test_tensors = clean_torch(y_test_tensors)

    metrics = [
        MulticlassConfusionMatrix(num_classes=3).to(device)
        for i in range(len(cls_state_names))
    ]

    scaled_label = scale_label(y_train[:, 0].tolist(), device=device)
    scaled_val_label = scale_label(y_test[:, 0].tolist(), device=device)

    predicted = []
    acc_list = []
    loss_list = []
    # outputs = Variable(torch.zeros((1, hidden_output_size)).to(device)).to(device)
    # outputs = np.zeros((1, layer * outs + layer))
    torch_outputs = Variable(torch.zeros((1, 1, 1)).to(device)).to(device)
    # val_outputs = np.zeros((1, hidden_output_size))
    # val_outputs = Variable(torch.zeros((1, hidden_output_size)).to(device)).to(device)
    val_torch_outputs = Variable(torch.zeros((1, 1, 1)).to(device)).to(device)
    max_list = np.zeros(1)
    labels = np.array(y_train[:, 0].tolist())  # .flatten()
    val_loss_list = []
    val_acc_list = []
    predicts = [[] for i in range(len(classificators) - 1)]
    predicted = [[] for i in range(len(classificators) - 1)]
    max_acc_list = [[] for i in range(len(classificators))]
    val_max_acc_list = [[] for i in range(len(classificators))]
    # trains the value using each input and label
    print("\nEpoch {}/{},".format(epoch + 1, epoch_count))
    for batches in tqdm(range(math.ceil(X_train_tensors.shape[0] / batchsize))):
        model.train()
        output = 0
        scaled_batch = 0
        labels_batch = []
        if batches >= math.ceil(X_train_tensors.shape[0] / batchsize) - 1:
            # print("runs")
            output = model.forward(
                X_train_tensors[batches * batchsize :],
                device,
            )
            scaled_batch = scaled_label[batches * batchsize :].detach().clone()
            labels_batch = labels[batches * batchsize :]
        else:
            output = model.forward(
                X_train_tensors[batches * batchsize : (batches + 1) * batchsize],
                device,
            )
            scaled_batch = (
                scaled_label[batches * batchsize : (batches + 1) * batchsize]
                .detach()
                .clone()
            )
            labels_batch = labels[batches * batchsize : (batches + 1) * batchsize]

        torch_outputs = output.detach().clone()
        # print("output", output.shape)
        loss = loss_fn(
            # torch.max(output.squeeze(-1), dim=1).indices.type(torch.float64),
            output,
            # torch.max(scaled_batch, dim=1).indices.type(torch.float64),
            scaled_batch,
        )

        # calculates the loss of the loss function
        loss.backward()
        # improve from loss, this is the actual backpropergation
        optimizer.step()

        optimizer.zero_grad()
        # outputs = outputs.cpu().detach().numpy()
        loss_list.append(float(loss.item()))

        if extra_train:
            model.eval()
            if batches <= 0:
                print("\n\nExtra Training")

            with torch.no_grad():
                if batches >= math.ceil(X_train_tensors.shape[0] / batchsize) - 1:
                    # print("runs")
                    output = model.forward(
                        X_train_tensors[batches * batchsize :],
                        device,
                    )
                else:
                    output = model.forward(
                        X_train_tensors[
                            batches * batchsize : (batches + 1) * batchsize
                        ],
                        device,
                    )
            torch_outputs = output.detach().clone()

            # if epoch >= epoch_count - 1:
            #     outputs = Variable(torch.zeros((1, hidden_output_size)).to(device)).to(device)
            #     torch_outputs = Variable(torch.zeros((1, 1, 1)).to(device)).to(device)
            #     if multi_target:
            #         torch_outputs = Variable(torch.zeros((1, layer, 1, 1)).to(device)).to(
            #             device
            #         )
            #     for input, one_label in tqdm(
            #         zip(X_train_tensors, y_train_tensors), total=X_train_tensors.shape[0]
            #     ):
            #         # input_final = torch.reshape(input, (input.shape[0], 1, input.shape[1])).to(
            #         #     device
            #         # )
            #         # predicted values
            #         # sigmoid_output,
            #         output, hidden_output = model.forward(
            #             input, device, multi_target=multi_target, hiddens=hiddens
            #         )
            #         # print(output)
            #         # print(output.shape)
            #         # print(hidden_output.shape)
            #         # inplementig data into MulticlassConfusionMatrix
            #         torch_outputs = torch.cat((torch_outputs, output))
            #         outputs = torch.cat(
            #             (
            #                 outputs,
            #                 torch.reshape(
            #                     hidden_output.clone(),
            #                     # (1, hidden_output.shape[0]  hidden_output.shape[-1]),
            #                     (1, hidden_output.shape[-1]),
            #                 ),
            #             ),
            #             axis=0,
            #         )
            if True in torch.isnan(output):
                print(" \n", output)
        #         print(outputs[-1], " \n", output)

        # outputs = outputs.cpu().detach().numpy()
        # compare = torch_outputs[1:] - scaled_label
        max = torch.reshape(torch_outputs.detach().clone(), (torch_outputs.shape[0], 3))
        max = torch.argmax(max, dim=1).cpu().detach().numpy()
        max_list = np.concatenate([max_list, max])
        compare = (
            torch.reshape(torch_outputs.detach().clone(), (torch_outputs.shape[0], 3))
            - scaled_batch.detach().clone()
        )
        compare[compare < 0] *= -1
        acc_list.append(
            100
            - (
                (compare).float().sum()
                * 100
                / (scaled_batch.shape[0] * scaled_batch.shape[-1])
                / 2
            )
            # / (layer * outs)
        )

        # print("torch_outputs", torch_outputs.shape)
        # print("labels_batch", len(labels_batch))

        sh_pr = False
        if batches >= math.ceil(X_train_tensors.shape[0] / batchsize) - 1:
            sh_pr = True

        if use_svm:
            fc = False
            if batches == 0:
                fc = True
            # plotting_correlation(outputs[1:])
            metrics, predicts = add_metric_row(
                classificators,
                cls_names,
                cls_state_names,
                torch.reshape(
                    torch_outputs,
                    (torch_outputs.shape[0], 3),
                )
                .cpu()
                .detach()
                .clone()
                .numpy(),
                labels_batch,
                max,
                history,
                metrics,
                0,
                device,
                name_end="_accuracy",
                return_predictions=True,
                fit_cls=fc,
                fit_name=name,
                show_pr=sh_pr,
                predicts=predicts,
                max_acc_list=max_acc_list,
                save_hist=sh_pr,
            )
        else:  # plotting_correlation(outputs[1:])
            _, predicts = add_metric_row(
                classificators,
                cls_names,
                cls_state_names,
                torch.reshape(
                    torch_outputs,
                    (torch_outputs.shape[0], 3),
                )
                .cpu()
                .detach()
                .clone()
                .numpy(),
                labels_batch,
                max,
                history,
                metrics,
                0,
                device,
                name_end="_accuracy",
                return_predictions=True,
                show_pr=sh_pr,
                predicts=predicts,
                max_acc_list=max_acc_list,
                save_hist=sh_pr,
            )

    # acc_list.append((out_max == label).float() * 100)
    # setting nan values to 0 to calculate the mean
    my_acc = torch.FloatTensor(acc_list).to(device)
    my_acc = clean_torch(my_acc).mean()
    my_loss = torch.FloatTensor(loss_list).to(device)
    my_loss = clean_torch(my_loss).mean()

    # showing training
    if my_loss != "NaN":
        # history["loss"].append(convert_loss(float(my_loss)))
        history["loss"].append(float(my_loss) / 100)
    else:
        history["loss"].append(history["loss"][-1])

    if my_acc != "NaN":
        # history["accuracy"].append(convert_loss(float(my_acc)))
        history["accuracy"].append(float(my_acc))
    else:
        history["accuracy"].append(history["accuracy"][-1])

    print(
        "\nEpoch {}/{}, Loss: {:.8f}, Accuracy: {:.5f}".format(
            # epoch + 1, epoch_count, convert_loss(my_loss), my_acc
            epoch + 1,
            epoch_count,
            my_loss,
            my_acc,
        )
    )

    # if use_plot:
    if False:
        plotting_scatter_2d(
            outputs[1:], labels, predicted=predicted_svm, name="testing_svm"
        )
        plotting_scatter_2d(
            outputs[1:], labels, predicted=predicted_rf, name="testing_rf"
        )

    # testing trained model on unused values
    model.eval()
    val_labels = np.array(y_test[:, 0].tolist())
    for batches in tqdm(range(math.ceil(X_test_tensors.shape[0] / batchsize))):
        # print(
        #     "Batch: ",
        #     batches + 1,
        #     " / ",
        #     math.ceil(X_test_tensors.shape[0] / batchsize),
        # )
        output = 0
        scaled_batch = 0
        val_labels_batch = 0
        with torch.no_grad():
            if batches >= math.ceil(X_test_tensors.shape[0] / batchsize) - 1:
                # print("runs")
                output = model.forward(
                    X_test_tensors[batches * batchsize :],
                    device,
                )
                scaled_batch = scaled_val_label[batches * batchsize :]
                val_labels_batch = val_labels[batches * batchsize :]
            else:
                output = model.forward(
                    X_test_tensors[batches * batchsize : (batches + 1) * batchsize],
                    device,
                )
                scaled_batch = scaled_val_label[
                    batches * batchsize : (batches + 1) * batchsize
                ]

                val_labels_batch = val_labels[
                    batches * batchsize : (batches + 1) * batchsize
                ]
                # out_max = torch.reshape(
                #     torch.argmax(output.detach().clone()).to(device), (1, 1)
                # ).to(device)
            val_torch_outputs = output.detach().clone()
            # del output, input

            # loss
            val_loss = loss_fn(
                # torch.max(val_torch_outputs.squeeze(-1), dim=1).indices.type(torch.float64),
                output,
                # torch.max(scaled_batch.detach().clone(), dim=1).indices.type(torch.float64),
                scaled_batch,
            )
            val_loss_list.append(float(val_loss.item()))
        val_max = torch.reshape(
            val_torch_outputs.detach().clone(), (val_torch_outputs.shape[0], 3)
        )
        val_max = torch.argmax(val_max, dim=1).cpu().detach().numpy()
        max_list = np.concatenate([max_list, val_max])
        compare = (
            torch.reshape(
                val_torch_outputs.detach().clone(),
                (scaled_batch.shape[0], 3),
            )
            - scaled_batch.detach().clone()
        )
        # compare = val_torch_outputs[1:] - scaled_val_label
        compare[compare < 0] *= -1
        val_acc_list.append(
            100
            - (
                (compare).float().sum()
                * 100
                # / (layer * outs)
                / (scaled_val_label.shape[0] * scaled_val_label.shape[1])
                / 2
            )
        )

        sh_pr = False
        if batches >= math.ceil(X_test_tensors.shape[0] / batchsize) - 1:
            sh_pr = True
        fc = False
        if batches == 0:
            fc = True
        # val_labels = np.array(y_test[:, 0].tolist())
        metrics, predicted = add_metric_row(
            classificators,
            cls_names,
            cls_state_names,
            torch.reshape(
                val_torch_outputs,
                (val_torch_outputs.shape[0], 3),
            )
            .cpu()
            .detach()
            .clone()
            .numpy(),
            val_labels_batch,
            val_max,
            history,
            metrics,
            1,
            device,
            name="val_",
            name_end="_accuracy",
            return_predictions=True,
            show_pr=sh_pr,
            predicts=predicted,
            max_acc_list=val_max_acc_list,
            save_hist=fc,
        )
    # val_acc_list.append((out_max == test_label).float() * 100)

    my_val_acc = torch.FloatTensor(val_acc_list).to(device)
    # setting nan values to 0 to calculate the mean
    my_val_acc = clean_torch(my_val_acc).mean()
    my_val_loss = torch.FloatTensor(val_loss_list).to(device)
    my_val_loss = clean_torch(my_val_loss).mean()

    if my_val_loss != "NaN":
        # history["val_loss"].append(convert_loss(float(my_val_loss)))
        history["val_loss"].append(float(my_val_loss) / 100)
    else:
        history["val_loss"].append(history["val_loss"][-1])

    if my_val_acc != "NaN":
        # history["val_accuracy"].append(convert_loss(float(my_val_acc)))
        history["val_accuracy"].append(float(my_val_acc))
    else:
        history["val_accuracy"].append(history["val_accuracy"][-1])
    print(
        "\nEpoch {}/{}, val Loss: {:.8f}, val Accuracy: {:.5f}".format(
            # epoch + 1, epoch_count, convert_loss(my_val_loss), my_val_acc
            epoch + 1,
            epoch_count,
            my_val_loss,
            my_val_acc,
        )
    )
    # del val_outputs

    for i in range(len(predicts)):
        # predicts[i] = np.concatenate((predicts[i], predicted[i]), axis=0)
        predicts[i] = np.concatenate([predicts[i], predicted[i]])

        if use_plot:
            plotting_timeseries(
                np.concatenate((y_train.tolist(), y_test.tolist())),
                [max_list[1:]] + predicts,
                cls_names=cls_names,
            )

    del (
        predicted,
        torch_outputs,
        val_torch_outputs,
        my_acc,
        my_loss,
        my_val_acc,
        my_val_loss,
        X_train_tensors,
        X_test_tensors,
        scaled_label,
        scaled_val_label,
    )

    torch.cuda.empty_cache()
    print("\n")

    return (model, history, classificators, cls_names, metrics, cls_state_names)


# plotting predicted data only for hourly named models
# future predictions has no labels so they can't be printed


# aktively used reshaping and predicting
def predict_lstm(model, train, device, hiddens=True):
    train_tensors = Variable(torch.Tensor(train).to(device)).to(device)
    # input_final = torch.reshape(train_tensors, (1, 1, train_tensors.shape[-1])).to(
    #     device
    # )
    output, hidden_output = model.forward(train_tensors, device, hiddens=hiddens)
    return output, hidden_output
