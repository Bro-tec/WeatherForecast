import CollectData.get_learning_data as gld
import matplotlib.pyplot as plt
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
from sklearn.metrics import accuracy_score
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
        self.rel = nn.ReLU()
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

    def forward(self, input, device, hiddens=True):
        out, (hn, cn) = self.lstm(input, (self.hidden_state, self.cell_state))
        if hiddens:
            self.hidden_state = hn.detach()
            self.cell_state = cn.detach()
        # print("out", out.shape)
        l = self.fc(out[:, -1, :])
        l2 = self.rel(l)
        # print("l2", l2.shape)

        return l2.type(torch.float64).to(device)


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
):
    history = {
        "accuracy": [0],
        "loss": [0.5],
        "val_accuracy": [0],
        "val_loss": [0.5],
        # "argmax_accuracy": [0],
        # "val_argmax_accuracy": [0],
    }
    model = {"haha": [1, 2, 3]}

    if os.path.exists(f"./Models/{name}.pth") and os.path.exists(
        f"./Models/{name}.pth"
    ):
        checkpoint = torch.load(f"./Models/{name}.pth", map_location=device)
        model = checkpoint["model"]
        with open(f"./Models/{name}_history.json", "r") as f:
            history = json.load(f)
        print("Model found")
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if os.path.exists(f"./Models/{name}.pth"):
        checkpoint = torch.load(f"./Models/{name}.pth", map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer"])
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


# plotting Evaluation via Accuracy, Loss and MulticlassConfusionMatrix
def plotting_hist(history, metrics, name):
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
    name_tag = f"{name}_plot"
    # summarize history for accuracy
    plt.plot(history["accuracy"])
    plt.plot(history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(f"./Plots/{name_tag}_accuracy.png")
    plt.close()
    # summarize history for loss
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(f"./Plots/{name_tag}_loss.png")
    plt.close()

    # summarize MulticlassConfusionMatrix
    fig, ax = plt.subplots(
        2,
        2,
        figsize=(4, 7),
        sharey="row",
    )
    plt.title("Confusion Matrix")
    metric_names = ["Train Icon", "Test Icon", "Train Condition", "Test Condition"]
    for i, metric in enumerate(metrics):
        j = math.floor(i / 2)
        k = i % 2
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
    plt.savefig(f"./Plots/{name_tag}_matrix.png")
    plt.close()


# unscaling the output because it usually doesnt get over 1
def unscale_output(output):
    output[:, 0] *= 100
    output[:, 1] *= 360
    output[:, 2] *= 100
    output[:, 3] *= 10000
    return output


# scaling label to check if output was good enough
def scale_label(output):
    output[:, 0] /= 100
    output[:, 1] /= 360
    output[:, 2] /= 100
    output[:, 3] /= 10000
    # output[:, :, 4] /= 21
    # output[:, :, 5] /= 21
    return output


def scale_features(output):
    for i in range(5):
        output[:, :, i * 5] /= 100
        output[:, :, 1 + i * 5] /= 360
        output[:, :, 2 + i * 5] /= 100
        output[:, :, 3 + i * 5] /= 10000
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
    CEloss_fn = nn.CrossEntropyLoss().to(device)
    BCEWL1loss_fn = nn.BCEWithLogitsLoss().to(device)
    BCEWL2loss_fn = nn.BCEWithLogitsLoss().to(device)
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
    metrics = [MulticlassConfusionMatrix(num_classes=21).to(device) for i in range(4)]

    predicted = []
    acc_list = []
    loss_list = []
    labels = Variable(torch.Tensor(np.array(y_train.tolist())).to(device)).to(
        device
    )  # .flatten()
    val_labels = Variable(torch.Tensor(np.array(y_test.tolist())).to(device)).to(
        device
    )  # .flatten()
    val_loss_list = []
    val_acc_list = []
    predicts = [[] for i in range(4)]
    predicted = [[] for i in range(4)]
    max_acc_list = [[] for i in range(4)]
    val_max_acc_list = [[] for i in range(4)]

    # trains the value using each input and label
    for batches in tqdm(range(math.ceil(X_train_tensors.shape[0] / batchsize))):
        model.train()
        output = 0
        scaled_batch = 0
        # print("batches", batches)

        if batches >= math.ceil(X_train_tensors.shape[0] / batchsize) - 1:
            # print("runs")
            output = model.forward(
                X_train_tensors[batches * batchsize :], device, hiddens=False
            )
            scaled_batch = y_train_tensors[batches * batchsize :].detach().clone()
            train_label = labels[batches * batchsize :]
        else:
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
            train_label = labels[batches * batchsize : (batches + 1) * batchsize]

        torch_outputs = output.detach().clone()
        # print("output", output.shape, "\nscaled_batch", scaled_batch.shape)
        loss = CEloss_fn(output[:, :4], scaled_batch[:, :4])
        loss += BCEWL1loss_fn(output[:, 4:25], scaled_batch[:, 4:25])
        loss += BCEWL2loss_fn(output[:, 25:46], scaled_batch[:, 25:46])
        # calculates the loss of the loss function
        loss.backward()
        # improve from loss, this is the actual backpropergation
        optimizer.step()
        # caluclate the gradient, manually setting to 0
        optimizer.zero_grad()

        loss_list.append(float(loss.item()))

        compare = torch_outputs.detach().clone() - scaled_batch.detach().clone()
        compare[compare < 0] *= -1
        acc_list.append(
            100
            - (
                (compare).float().sum()
                * 100
                / (scaled_batch.shape[0] * scaled_batch.shape[-2])
            )
            # / (layer * outs)
        )
        # inplementig data into MulticlassConfusionMatrix
        metric_output = unscale_output(output)
        metrics[0].update(metric_output[:, -42:-21], train_label[:, -42:-21])
        # print("metric", metric_output[:, -21:], label[:, -21:])
        metrics[1].update(metric_output[:, -21:], train_label[:, -21:])
        # label need to be scaled and zero labels need to get a value
        # scaled_label = scale_label(label)

    my_acc = torch.FloatTensor(acc_list).to(device)
    my_acc = clean_torch(my_acc).mean()
    my_loss = torch.FloatTensor(loss_list).to(device)
    my_loss = clean_torch(my_loss).mean()

    # showing training
    if my_loss != "NaN":
        history["loss"].append(float(my_loss) / 100)
    else:
        history["loss"].append(history["loss"][-1])

    if my_acc != "NaN":
        history["accuracy"].append(float(my_acc))
    else:
        history["accuracy"].append(history["accuracy"][-1])
        print(
            "Epoch {}/{}, Loss: {:.5f}, Accuracy: {:.5f}".format(
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
                # print("runs")
                output = model.forward(
                    X_test_tensors[batches * batchsize :], device, hiddens=False
                )
                scaled_batch = y_test_tensors[batches * batchsize :]
                test_label = val_labels[batches * batchsize :]
            else:
                output = model.forward(
                    X_test_tensors[batches * batchsize : (batches + 1) * batchsize],
                    device,
                    hiddens=False,
                )
                scaled_batch = y_test_tensors[
                    batches * batchsize : (batches + 1) * batchsize
                ]
                test_label = val_labels[batches * batchsize : (batches + 1) * batchsize]

            val_torch_outputs = output.detach().clone()
            # del output, input

            # loss
            val_loss = CEloss_fn(output[:, :4], scaled_batch[:, :4])
            val_loss = BCEWL1loss_fn(output[:, 4:25], scaled_batch[:, 4:25])
            val_loss = BCEWL2loss_fn(output[:, 25:46], scaled_batch[:, 25:46])
            val_loss_list.append(float(val_loss.item()))
        compare = val_torch_outputs.detach().clone() - scaled_batch.detach().clone()
        # compare = val_torch_outputs[1:] - scaled_val_label
        compare[compare < 0] *= -1
        val_acc_list.append(
            100
            - (
                (compare).float().sum()
                * 100
                # / (layer * outs)
                / (scaled_batch.shape[0] * scaled_batch.shape[-1])
            )
        )
        metric_output = unscale_output(output)
        # print("metric", metric_output[:, 4:25].shape, test_label[:, 4:25].shape)
        # print("metric", metric_output[:, -21:].shape, test_label[:, -21:].shape)
        metrics[2].update(metric_output[:, -42:-21], test_label[:, -42:-21])
        metrics[3].update(metric_output[:, -21:], test_label[:, -21:])

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
def prediction(model, train, name, device):
    train_tensors = Variable(torch.Tensor(train).to(device)).to(device)
    input_final = torch.reshape(train_tensors, (1, 1, train_tensors.shape[-1])).to(
        device
    )
    output = model.forward(input_final, device)
    output = unscale_output(output=output, name=name)
    return [output]


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


# prediction main code specially for daily models
def predictDaily(date, device, mode="normal", model_num=0, id="", city=""):
    out_list = []
    if date <= dt.now() - td(days=2):
        (
            train,
            label1,
            label2,
            label3,
            label4,
            label5,
            label6,
            label7,
        ) = gld.get_predictDataDaily(date, id=id)
        if isinstance(train, str):
            print("error occured please retry with other ID/Name")
            return
        print("\ntraining count", train.shape)

        model1, optimizer, loss_fn, metric, history = load_own_Model(
            "Daily", device, loading_mode=mode, t=model_num, input_count=7752
        )
        model2, optimizer, loss_fn, metric, history = load_own_Model(
            "Daily", device, loading_mode=mode, t=model_num, input_count=7752
        )
        model3, optimizer, loss_fn, metric, history = load_own_Model(
            "Daily", device, loading_mode=mode, t=model_num, input_count=7752
        )
        model4, optimizer, loss_fn, metric, history = load_own_Model(
            "Daily", device, loading_mode=mode, t=model_num, input_count=7752
        )
        model5, optimizer, loss_fn, metric, history = load_own_Model(
            "Daily", device, loading_mode=mode, t=model_num, input_count=7752
        )
        model6, optimizer, loss_fn, metric, history = load_own_Model(
            "Daily", device, loading_mode=mode, t=model_num, input_count=7752
        )
        model7, optimizer, loss_fn, metric, history = load_own_Model(
            "Daily", device, loading_mode=mode, t=model_num, input_count=7752
        )
        if not id == "" or city == "":
            out_list.append(prediction(model1, train, "Daily", device))
            out_list.append(prediction(model2, train, "Daily", device))
            out_list.append(prediction(model3, train, "Daily", device))
            out_list.append(prediction(model4, train, "Daily", device))
            out_list.append(prediction(model5, train, "Daily", device))
            out_list.append(prediction(model6, train, "Daily", device))
            out_list.append(prediction(model7, train, "Daily", device))
    else:
        train = gld.get_predictDataDaily(date, id=id)
        if isinstance(train, str):
            print("error occured please retry with other ID/Name")
            return
        print("\ntraining count", train.shape)
        model1, optimizer, loss_fn, metric, history = load_own_Model(
            "Daily", device, loading_mode=mode, t=model_num, input_count=7752
        )
        model2, optimizer, loss_fn, metric, history = load_own_Model(
            "Daily", device, loading_mode=mode, t=model_num, input_count=7752
        )
        model3, optimizer, loss_fn, metric, history = load_own_Model(
            "Daily", device, loading_mode=mode, t=model_num, input_count=7752
        )
        model4, optimizer, loss_fn, metric, history = load_own_Model(
            "Daily", device, loading_mode=mode, t=model_num, input_count=7752
        )
        model5, optimizer, loss_fn, metric, history = load_own_Model(
            "Daily", device, loading_mode=mode, t=model_num, input_count=7752
        )
        model6, optimizer, loss_fn, metric, history = load_own_Model(
            "Daily", device, loading_mode=mode, t=model_num, input_count=7752
        )
        model7, optimizer, loss_fn, metric, history = load_own_Model(
            "Daily", device, loading_mode=mode, t=model_num, input_count=7752
        )
        if not id == "" or city == "":
            out_list.append(prediction(model1, train, "Daily", device))
            out_list.append(prediction(model2, train, "Daily", device))
            out_list.append(prediction(model3, train, "Daily", device))
            out_list.append(prediction(model4, train, "Daily", device))
            out_list.append(prediction(model5, train, "Daily", device))
            out_list.append(prediction(model6, train, "Daily", device))
            out_list.append(prediction(model7, train, "Daily", device))
