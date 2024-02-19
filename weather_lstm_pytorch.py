import CollectData.get_learning_data as gld
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime as dt
from datetime import timedelta as td
from tqdm import tqdm

# from sklearn import preprocessing as pr
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchmetrics.classification import MulticlassConfusionMatrix
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
    def __init__(self, inputs, device, layer=3, outputs=6):
        self.output_size = outputs
        self.num_layer = layer
        self.hidden_size = int(round(inputs / 10))
        super(PyTorch_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=inputs,
            hidden_size=self.hidden_size,
            num_layers=layer,
            bidirectional=True,
            batch_first=True,
        ).to(device)
        self.fc = nn.Linear(self.hidden_size * 2, outputs).to(device)
        self.sig = nn.Sigmoid().to(device)

    def forward(self, input, device):
        hidden_state = Variable(
            torch.zeros(self.num_layer * 2, input.size(0), self.hidden_size).to(device)
        ).to(device)
        cell_state = Variable(
            torch.zeros(self.num_layer * 2, input.size(0), self.hidden_size).to(device)
        ).to(device)
        out, (hn, cn) = self.lstm(input, (hidden_state, cell_state))
        # reshaping output to 1d tensor
        out = out[-1, -1, :]
        out = self.fc(out)
        # sigmoid for the labels
        out[4] = self.sig(out[4])
        out[5] = self.sig(out[5])
        return out


# loading model if already saved or creating a new model
def load_own_Model(
    name, device, loading_mode="normal", t=0, input_count=86, learning_rate=0.001
):
    history = {"accuracy": [0], "loss": [0], "val_accuracy": [0], "val_loss": [0]}
    model = PyTorch_LSTM(input_count, device)
    if loading_mode == "timestep" or loading_mode == "ts":
        if os.path.exists(f"./Models/{name}_{str(t)}.pth") and os.path.exists(
            f"./Models/{name}_{str(t)}.pth"
        ):
            model.load_state_dict(torch.load(f"./Models/{name}_{str(t)}.pth"))
            model.eval()
            with open(f"./Models/{name}_history_{str(t)}.json", "r") as f:
                history = json.load(f)
            print("Model found")
        elif os.path.exists(f"./Models/{name}_{str(t-1)}.pth") and os.path.exists(
            f"./Models/{name}_{str(t-1)}.pth"
        ):
            model.load_state_dict(torch.load(f"./Models/{name}_{str(t-1)}.pth"))
            model.eval()
            with open(f"./Models/{name}_history_{str(t-1)}.json", "r") as f:
                history = json.load(f)
            print("Model found")
        else:
            print("Data not found or not complete")

    else:
        if os.path.exists(f"./Models/{name}.pth") and os.path.exists(
            f"./Models/{name}.pth"
        ):
            model.load_state_dict(torch.load(f"./Models/{name}.pth"))
            model.eval()
            with open(f"./Models/{name}_history.json", "r") as f:
                history = json.load(f)
            print("Model found")
        else:
            print("Data not found or not complete")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    # loss_fn = nn.KLDivLoss().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    metric = MulticlassConfusionMatrix(num_classes=21).to(device)
    return model, optimizer, loss_fn, metric, history


# saving model if saving_mode set to ts or timestamp it will use the number for the model to save it.
# ts helps to choose saved model data before the model started to overfit or not work anymore
def save_own_Model(name, history, model, saving_mode="normal", t=0):
    if saving_mode == "timestep" or saving_mode == "ts":
        with open(f"./Models/{name}_history_{str(t)}.json", "w") as fp:
            json.dump(history, fp)
        torch.save(model.state_dict(), f"./Models/{name}_{str(t)}.pth")
        print("Saved model")
    else:
        with open(f"./Models/{name}_history.json", "w") as fp:
            json.dump(history, fp)
        torch.save(model.state_dict(), f"./Models/{name}.pth")
        print("Saved model")


# plotting Evaluation via Accuracy, Loss and MulticlassConfusionMatrix
def plotting_hist(history, metric, name, saving_mode="normal", t=0):
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
    if saving_mode == "timestep" or saving_mode == "ts":
        name_tag = f"{name}_plot_{t}"
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
    fig, ax = metric.plot()
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    fig.set_figwidth(16)
    fig.set_figheight(16)
    plt.title("Confusion Matrix")
    ax.xaxis.set_ticklabels(icons)
    ax.yaxis.set_ticklabels(icons)
    plt.savefig(f"./Plots/{name_tag}_matrix.png")
    plt.close()


# unscaling the output because it usually doesnt get over 1
def unscale_output(output, name):
    if name == "Hourly" or name == "Hourly24":
        output[0] *= 100
        output[1] *= 360
        output[2] *= 100
        output[3] *= 10000
        output[4] *= 21
        output[5] *= 21
    else:
        output[0] *= 100
        output[1] *= 100
        output[2] *= 100
        output[3] *= 100
        output[4] *= 21
        output[5] *= 21
    return output


# scaling label to check if output was good enough
def scale_label(output, name):
    if name == "Hourly" or name == "Hourly24":
        output[0] /= 100
        output[1] /= 360
        output[2] /= 100
        output[3] /= 10000
        output[4] /= 21
        output[5] /= 21
    else:
        output[0] /= 100
        output[1] /= 100
        output[2] /= 100
        output[3] /= 100
        output[4] /= 21
        output[5] /= 21
    return output


# whole training of the LSTM features and labels need to be 2d shaped
def train_LSTM(
    name,
    feature,
    label,
    model,
    optimizer,
    loss_fn,
    metric,
    history,
    device,
    epoch_count=1,
):
    X_train, X_test, y_train, y_test = train_test_split(
        feature, label, test_size=0.02, random_state=42
    )
    X_train_tensors = Variable(torch.Tensor(X_train).to(device)).to(device)
    X_test_tensors = Variable(torch.Tensor(X_test).to(device)).to(device)
    y_train_tensors = Variable(torch.Tensor(y_train)).to(device)
    y_test_tensors = Variable(torch.Tensor(y_test)).to(device)
    X_train_tensors = torch.reshape(
        X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1])
    ).to(device)
    X_test_tensors = torch.reshape(
        X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])
    ).to(device)

    if len(y_train_tensors.shape) >= 3:
        y_train_tensors = torch.reshape(
            y_train_tensors, (y_train_tensors.shape[0], y_train_tensors.shape[-1])
        ).to(device)
        y_test_tensors = torch.reshape(
            y_test_tensors, (y_test_tensors.shape[0], y_test_tensors.shape[-1])
        ).to(device)

    # epoche just repeats the training with excactly the same values
    for epoch in range(epoch_count):
        loss_list = []
        acc_list = []
        val_loss_list = []
        val_acc_list = []
        # trains the value using each input and label
        for input, label in tqdm(
            zip(X_train_tensors, y_train_tensors), total=X_train_tensors.shape[0]
        ):
            input_final = torch.reshape(input, (input.shape[0], 1, input.shape[1])).to(
                device
            )
            # predicted values
            output = model.forward(input_final, device)
            # inplementig data into MulticlassConfusionMatrix
            metric_output = unscale_output(output, name)
            metric.update(metric_output[-2:], label[-2:])
            # label need to be scaled and zero labels need to get a value
            scaled_label = scale_label(label, name)
            scaled_label[scaled_label == 0.0] = random.uniform(0, 1)
            # output[scaled_label==0.0] = 0.0
            # caluclate the gradient, manually setting to 0
            optimizer.zero_grad()
            # obtain the loss function
            loss = loss_fn(output, scaled_label)
            # calculates the loss of the loss function
            loss.backward(retain_graph=True)
            # improve from loss, this is the actual backpropergation
            optimizer.step()
            # calculating loss and counting accuricy
            loss_list.append(float(loss.item()))
            acc_list.append(
                (((output >= scaled_label - 0.05) & (output <= scaled_label + 0.05)))
                .float()
                .sum()
                / 6
                * 100
            )

        my_acc = torch.FloatTensor(acc_list).to(device)
        # setting nan values to 0 to calculate the mean
        my_acc[my_acc != my_acc] = 0
        my_acc = my_acc.mean()
        my_loss = torch.FloatTensor(loss_list).to(device)
        my_loss[my_loss != my_loss] = 0
        my_loss = my_loss.mean()
        # showing training
        history["loss"].append(float(my_loss))
        history["accuracy"].append(float(my_acc))
        print(
            "Epoch {}/{}, Loss: {:.5f}, Accuracy: {:.5f}".format(
                epoch + 1, epoch_count, my_loss, my_acc
            )
        )

        # testing trained model on unused values
        for input, label in tqdm(
            zip(X_test_tensors, y_test_tensors), total=X_test_tensors.shape[0]
        ):
            input_final = torch.reshape(input, (input.shape[0], 1, input.shape[1])).to(
                device
            )
            output = model.forward(input_final, device)  # forward pass

            metric_output = unscale_output(output, name)
            metric.update(metric_output[-2:], label[-2:])
            scaled_label = scale_label(label, name)
            # label need to be scaled and zero labels need to get a value
            scaled_label[scaled_label == 0.0] = random.uniform(0, 1)
            # output[scaled_label==0.0] = 0.0

            loss = loss_fn(output, scaled_label)
            val_loss_list.append(loss.item())
            val_acc_list.append(
                (((output >= scaled_label - 0.05) & (output <= scaled_label + 0.05)))
                .float()
                .sum()
                / 6
                * 100
            )

        my_val_acc = torch.FloatTensor(val_acc_list).to(device)
        # setting nan values to 0 to calculate the mean
        my_val_acc[my_val_acc != my_val_acc] = 0
        my_val_acc = my_val_acc.mean()
        my_val_loss = torch.FloatTensor(val_loss_list).to(device)
        my_val_loss[my_val_loss != my_val_loss] = 0
        my_val_loss = my_val_loss.mean()
        # showing testing
        history["val_loss"].append(float(my_val_loss))
        history["val_accuracy"].append(float(my_val_acc))
        print("val Loss: {:.5f}, val Accuracy: {:.5f}".format(my_val_loss, my_val_acc))
        print("\n")

    return model, history


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
        plt.legend(["predicted", "real"], loc="upper left")
    fig.suptitle(f"prediction {plot_text}")
    plt.tight_layout()
    if mode == "timestep" or mode == "ts":
        plt.savefig(f"./Plots/{name}_{t}_prediction_{plot_text}.png")
    else:
        plt.savefig(f"./Plots/{name}_prediction_{plot_text}.png")


# plotting predicted data only for daily named models
# future predictions has no labels so they can't be printed
def plotting_Prediction_Daily(
    all_input,
    output,
    plot_text,
    label1=[],
    label2=[],
    label3=[],
    label4=[],
    label5=[],
    label6=[],
    label7=[],
    mode="normal",
    t=0,
):
    titles = ["Temperature", "Wind speed"]
    input = [all_input[0][3], all_input[0][5], all_input[0][12], all_input[0][16]]
    x = [i for i in range(8)]
    name = "Daily"
    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    for i in range(len(titles)):
        axs[i].set_title(titles[i])
    axs[0].plot(x, [input[0], *[output[i][0][0].item() for i in range(7)]])
    axs[0].plot(x, [input[0], *[output[i][0][1].item() for i in range(7)]])
    axs[1].plot(x, [input[1], *[output[i][0][2].item() for i in range(7)]])
    axs[1].plot(x, [input[1], *[output[i][0][3].item() for i in range(7)]])
    if plot_text != "future":
        axs[0].plot(
            x,
            [
                input[0],
                label1[0][0][0],
                label2[0][0][0],
                label3[0][0][0],
                label4[0][0][0],
                label5[0][0][0],
                label6[0][0][0],
                label7[0][0][0],
            ],
        )
        axs[0].plot(
            x,
            [
                input[0],
                label1[0][0][1],
                label2[0][0][1],
                label3[0][0][1],
                label4[0][0][1],
                label5[0][0][1],
                label6[0][0][1],
                label7[0][0][1],
            ],
        )
        axs[1].plot(
            x,
            [
                input[1],
                label1[0][0][2],
                label2[0][0][2],
                label3[0][0][2],
                label4[0][0][2],
                label5[0][0][2],
                label6[0][0][2],
                label7[0][0][2],
            ],
        )
        axs[1].plot(
            x,
            [
                input[1],
                label1[0][0][3],
                label2[0][0][3],
                label3[0][0][3],
                label4[0][0][3],
                label5[0][0][3],
                label6[0][0][3],
                label7[0][0][3],
            ],
        )
        axs[2].text(
            0,
            0,
            f"Day1 icon: real-{label1[0][0][4]} / prediction-{output[0][0][4]}, condition: real-{label1[0][0][5]}/ prediction-{output[0][0][5]}",
            fontsize=12,
        )
        axs[2].text(
            0,
            0.1,
            f"Day1 icon: real-{label2[0][0][4]} / prediction-{output[1][0][4]}, condition: real-{label2[0][0][5]}/ prediction-{output[1][0][5]}",
            fontsize=12,
        )
        axs[2].text(
            0,
            0.2,
            f"Day1 icon: real-{label3[0][0][4]} / prediction-{output[2][0][4]}, condition: real-{label3[0][0][5]}/ prediction-{output[2][0][5]}",
            fontsize=12,
        )
        axs[2].text(
            0,
            0.3,
            f"Day1 icon: real-{label4[0][0][4]} / prediction-{output[3][0][4]}, condition: real-{label4[0][0][5]}/ prediction-{output[3][0][5]}",
            fontsize=12,
        )
        axs[2].text(
            0,
            0.4,
            f"Day1 icon: real-{label5[0][0][4]} / prediction-{output[4][0][4]}, condition: real-{label5[0][0][5]}/ prediction-{output[4][0][5]}",
            fontsize=12,
        )
        axs[2].text(
            0,
            0.5,
            f"Day1 icon: real-{label6[0][0][4]} / prediction-{output[5][0][4]}, condition: real-{label6[0][0][5]}/ prediction-{output[5][0][5]}",
            fontsize=12,
        )
        axs[2].text(
            0,
            0.6,
            f"Day1 icon: real-{label7[0][0][4]} / prediction-{output[6][0][4]}, condition: real-{label7[0][0][5]}/ prediction-{output[6][0][5]}",
            fontsize=12,
        )

    if plot_text != "future":
        plt.legend(
            ["min predicted", "max predicted", "min real", "max real"], loc="upper left"
        )
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
        if type(train) == "str":
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
        train, label1, label2, label3, label4, label5, label6, label7 = (
            gld.get_predictDataDaily(date, id=id)
        )
        if type(train) == "str":
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
            plotting_Prediction_Daily(
                train,
                out_list,
                "test",
                label1=label1,
                label2=label2,
                label3=label3,
                label4=label4,
                label5=label5,
                label6=label6,
                label7=label7,
                mode=mode,
                t=model_num,
            )
    else:
        train = gld.get_predictDataDaily(date, id=id)
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
            plotting_Prediction_Daily(train, out_list, "future", mode=mode, t=model_num)
