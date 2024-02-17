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
import seaborn as sns
import warnings
plt.switch_backend('agg')

warnings.filterwarnings('ignore')

# trainData = gld.gen_trainDataHourly()
def check_cuda():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("this device uses " + device + " to train data")
    return torch.device(device)

 
class PyTorch_LSTM(nn.Module):
    def __init__(self, input, device, hidden=360, layer=3, outputs=6):
        self.output_size = outputs
        self.num_layer = layer
        self.hidden_size = hidden
        super(PyTorch_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input, hidden_size=hidden, num_layers=layer, bidirectional=True, batch_first=True).to(device)
        self.fc = nn.Linear(hidden*2, outputs).to(device)
        self.soft = nn.Softmax().to(device)
    def forward(self, input, device):
        hidden_state = Variable(torch.zeros(self.num_layer*2, input.size(0), self.hidden_size).to(device)).to(device)
        cell_state = Variable(torch.zeros(self.num_layer*2, input.size(0), self.hidden_size).to(device)).to(device)
        out, (hn, cn) = self.lstm(input, (hidden_state, cell_state))
        out = out[:,-1,:]
        # out = out.view(-1, ) #reshaping the data for Dense layer next
        out = out[-1,:]
        out = self.soft(out) #softmax
        out = self.fc(out)
        return out

def load_own_Model(name, device, loading_mode="normal", t=0, input_count=86, learning_rate=0.0000001):
    history = {"accuracy":[0], "loss":[0], "val_accuracy":[0], "val_loss":[0]}
    model = PyTorch_LSTM(input_count, device)
    if loading_mode=="timestep" or loading_mode=="ts":
        if os.path.exists(f"./Models/{name}_{str(t)}.pth") and os.path.exists(f"./Models/{name}_{str(t)}.pth"):
            model.load_state_dict(torch.load(f"./Models/{name}_{str(t)}.pth"))
            model.eval()
            with open(f'./Models/{name}_history_{str(t)}.json', 'r') as f:
                history = json.load(f)
            print("Model found")
        elif os.path.exists(f"./Models/{name}_{str(t-1)}.pth") and os.path.exists(f"./Models/{name}_{str(t-1)}.pth"):
            model.load_state_dict(torch.load(f"./Models/{name}_{str(t-1)}.pth"))
            model.eval()
            with open(f'./Models/{name}_history_{str(t-1)}.json', 'r') as f:
                history = json.load(f)
            print("Model found")
        else:
            print("Data not found or not complete")
    else:
        if os.path.exists(f"./Models/{name}.pth") and os.path.exists(f"./Models/{name}.pth"):
            model.load_state_dict(torch.load(f"./Models/{name}.pth"))
            model.eval()
            with open(f'./Models/{name}_history.json', 'r') as f:
                history = json.load(f)
            print("Model found")
        else:
            print("Data not found or not complete")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    loss_fn = nn.KLDivLoss().to(device)
    # loss_fn = nn.MultiMarginLoss().to(device)
    metric = MulticlassConfusionMatrix(num_classes=21).to(device)
    return model, optimizer, loss_fn, metric, history

def save_own_Model(name, history, model, saving_mode="normal", t=0):
    if saving_mode=="timestep" or saving_mode=="ts":
        with open(f"./Models/{name}_history_{str(t)}.json", "w") as fp:  
            json.dump(history, fp)
        torch.save(model.state_dict(), f"./Models/{name}_{str(t)}.pth")
        print("Saved model")
    else:
        with open(f"./Models/{name}_history.json", "w") as fp:  
            json.dump(history, fp)
        torch.save(model.state_dict(), f"./Models/{name}.pth")
        print("Saved model")
    
def plotting_hist(history, metric, name, saving_mode="normal", t=0):
    icons = [None, "clear-day", "clear-night", "partly-cloudy-day", "partly-cloudy-night", "cloudy", "fog", "wind", "rain", "sleet", "snow", "hail", "thunderstorm", "dry", "moist", "wet", "rime", "ice", "glaze", "not dry", "reserved"]
    name_tag = f'{name}_plot'
    if saving_mode=="timestep" or saving_mode=="ts":
        name_tag = f'{name}_plot_{t}'
    # summarize history for accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'./Plots/{name_tag}_accuracy.png')
    plt.close()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'./Plots/{name_tag}_loss.png')
    plt.close()
    
    # plt.figure()
    fig, ax = metric.plot()
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    fig.set_figwidth(20)
    fig.set_figheight(20)
    plt.title('Confusion Matrix')
    ax.xaxis.set_ticklabels(icons)
    ax.yaxis.set_ticklabels(icons)
    plt.savefig(f'./Plots/{name_tag}_matrix.png')
    plt.close()

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

def train_LSTM(name, feature, label, model, optimizer, loss_fn, metric, history, device, epoch_count=1):
    # scaled_X_ss = pr.StandardScaler().fit_transform(feature)
    # scaled_X_rs = pr.RobustScaler().fit_transform(feature)
    # scaled_X_pt = pr.PowerTransformer().fit_transform(feature)
    # scaled_X_n = pr.Normalizer().fit_transform(feature)
    
    # print(feature.shape)
    # print(feature[0])
    # print("StandardScaler")
    # print(scaled_X_ss.shape)
    # print(scaled_X_ss[0])
    # print("RobustScaler")
    # print(scaled_X_rs.shape)
    # print(scaled_X_rs[0])
    # print("PowerTransformer")
    # print(scaled_X_pt.shape)
    # print(scaled_X_pt[0])
    # print("Normalizer")
    # print(scaled_X_n.shape)
    # print(scaled_X_n[0])

    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.02, random_state=42)
    # print("X_train:",X_train.shape, ", X_test:", X_test.shape, ", y_train:", y_train.shape, ", y_test:", y_test.shape)
    
    X_train_tensors = Variable(torch.Tensor(X_train).to(device)).to(device)
    X_test_tensors = Variable(torch.Tensor(X_test).to(device)).to(device)

    y_train_tensors = Variable(torch.Tensor(y_train)).to(device)
    y_test_tensors = Variable(torch.Tensor(y_test)).to(device)
    # print("X_train_tensors:",X_train_tensors.shape, ", X_test_tensors:", X_test_tensors.shape, ", y_train_tensors:", y_train_tensors.shape, ", y_test_tensors:", y_test_tensors.shape)
    
    X_train_tensors = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1])).to(device)
    X_test_tensors = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])).to(device) 
    # print("X_train_tensors:",X_train_tensors.shape, ", X_test_tensors:", X_test_tensors.shape, ", y_train_tensors:", y_train_tensors.shape, ", y_test_tensors:", y_test_tensors.shape)
    
    if len(y_train_tensors.shape) >= 3:
        y_train_tensors = torch.reshape(y_train_tensors, (y_train_tensors.shape[0], y_train_tensors.shape[-1])).to(device)
        y_test_tensors = torch.reshape(y_test_tensors, (y_test_tensors.shape[0], y_test_tensors.shape[-1])).to(device) 
        
    for epoch in range(epoch_count):
        loss_list = []
        acc_list = []
        val_loss_list = []
        val_acc_list = []
        for input, label in tqdm(zip(X_train_tensors, y_train_tensors), total=X_train_tensors.shape[0]):
            input_final = torch.reshape(input, (input.shape[0], 1, input.shape[1])).to(device)
            # print("first label", label)
            
            output = model.forward(input_final, device) #forward pass
            
            # print(output[-2:])
            metric_output = unscale_output(output, name)
            # print(metric_output[-2:], label[-2:])
            metric.update(metric_output[-2:], label[-2:])
            # print("first output", output)
            
            scaled_label = scale_label(label, name)
            output[scaled_label==0.0] = 0.0
            # print("scaled label", scaled_label)
            # print("last output", output)
            optimizer.zero_grad() #caluclate the gradient, manually setting to 0
            # obtain the loss function
            loss = loss_fn(output, scaled_label)
            loss.backward(retain_graph=True) #calculates the loss of the loss function
            optimizer.step() #improve from loss, i.e backprop
            # print(output, " == ", scaled_label)
            loss_list.append(float(loss.item()))
            # print((((output >= scaled_label-0.005)&(output <= scaled_label+0.005) | (scaled_label==0))).float().sum())
            acc_list.append((((output >= scaled_label-0.02)&(output <= scaled_label+0.02))).float().sum()/6)
            
        my_acc = torch.FloatTensor(acc_list).to(device)
        my_acc[my_acc != my_acc] = 0
        my_acc = my_acc.mean()
        my_loss = torch.FloatTensor(loss_list).to(device)
        my_loss[my_loss != my_loss] = 0
        my_loss = my_loss.mean()
        history['loss'].append(float(my_loss)) 
        history['accuracy'].append(float(my_acc)) 
        print("Epoch {}/{}, Loss: {:.5f}, Accuracy: {:.5f}".format(epoch+1,epoch_count, my_loss, my_acc))
    
        for input, label in tqdm(zip(X_test_tensors, y_test_tensors), total=X_test_tensors.shape[0]):
            input_final = torch.reshape(input, (input.shape[0], 1, input.shape[1])).to(device)
            output = model.forward(input_final, device) #forward pass
            
            metric_output = unscale_output(output, name)
            metric.update(metric_output[-2:], label[-2:])
            # print("first output", output)
            
            # print("first label", label)
            scaled_label = scale_label(label, name)
            
            # print("scaled label", scaled_label)
            output[scaled_label==0.0] = 0.0
            
            # print("last output", output)
            
            loss = loss_fn(output, scaled_label)
            val_loss_list.append(loss.item())
            # print(output, " == ", label)
            # print((((output >= scaled_label-0.005)&(output <= scaled_label+0.005) | (scaled_label==0))).float().sum())
            val_acc_list.append((((output >= scaled_label-0.002)&(output <= scaled_label+0.002))).float().sum()/6)
        
        my_val_acc = torch.FloatTensor(val_acc_list).to(device)
        my_val_acc[my_val_acc !=my_val_acc] = 0
        my_val_acc = my_val_acc.mean()
        my_val_loss = torch.FloatTensor(val_loss_list).to(device)
        my_val_loss[my_val_loss != my_val_loss] = 0
        my_val_loss = my_val_loss.mean()
        history['val_loss'].append(float(my_val_loss))
        history['val_accuracy'].append(float(my_val_acc)) 
        print("val Loss: {:.5f}, val Accuracy: {:.5f}".format(my_val_loss, my_val_acc))
        print("\n")
    
    return model, history

def plotting_Prediction_hourly(all_input, output, plot_text, hourly=[], hourly24=[], mode="normal", t=0):
    titles = ["Temperature","Wind direction","Wind speed","Visibility"]
    input = [all_input[3],all_input[4],all_input[5],all_input[9],all_input[12],all_input[16]]
    x = [1, 2, 25]
    name = "Hourly"
    # summarize history for accuracy
    fig, axs = plt.subplots(len(titles), 1, figsize=(12, 8))
    for i in range(len(titles)):
        axs[i].set_title(titles[i])
        axs[i].plot(x,[input[i],output[0][0][i].item(),output[1][0][i].item()])
        if plot_text != "future":
            axs[i].plot(x,[input[i],hourly[i],hourly24[i]])
        # axs[i].ylabel('temp')
        # axs[i].xlabel('time')
    
    if plot_text != "future":
        plt.legend(['predicted', 'real'], loc='upper left')
    fig.suptitle(f'prediction {plot_text}')
    plt.tight_layout()
    if mode=="timestep" or mode=="ts":
        plt.savefig(f'./Plots/{name}_{t}_prediction_{plot_text}.png')
    else:
        plt.savefig(f'./Plots/{name}_prediction_{plot_text}.png')
    
    
def plotting_Prediction_Daily(all_input, output, plot_text, label1=[], label2=[], label3=[], label4=[], label5=[], label6=[], label7=[], mode="normal", t=0):
    titles = ["Temperature","Wind speed"]
    input = [all_input[0][3],all_input[0][5],all_input[0][12],all_input[0][16]]
    x = [i for i in range(8)]
    name = "Daily"
    # summarize history for accuracy
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    for i in range(len(titles)):
        axs[i].set_title(titles[i])
    axs[0].plot(x,[input[0],*[output[i][0][0].item() for i in range(7)]])
    axs[0].plot(x,[input[0],*[output[i][0][1].item() for i in range(7)]])
    axs[1].plot(x,[input[1],*[output[i][0][2].item() for i in range(7)]])
    axs[1].plot(x,[input[1],*[output[i][0][3].item() for i in range(7)]])
    if plot_text != "future":
        axs[0].plot(x,[input[0],label1[0][0][0], label2[0][0][0], label3[0][0][0], label4[0][0][0], label5[0][0][0], label6[0][0][0], label7[0][0][0]])
        axs[0].plot(x,[input[0],label1[0][0][1], label2[0][0][1], label3[0][0][1], label4[0][0][1], label5[0][0][1], label6[0][0][1], label7[0][0][1]])
        axs[1].plot(x,[input[1],label1[0][0][2], label2[0][0][2], label3[0][0][2], label4[0][0][2], label5[0][0][2], label6[0][0][2], label7[0][0][2]])
        axs[1].plot(x,[input[1],label1[0][0][3], label2[0][0][3], label3[0][0][3], label4[0][0][3], label5[0][0][3], label6[0][0][3], label7[0][0][3]])
    # axs[i].ylabel('temp')
    # axs[i].xlabel('time')
    
    if plot_text != "future":
        plt.legend(['min predicted', 'max predicted', 'min real', 'max real'], loc='upper left')
    fig.suptitle(f'prediction {plot_text}')
    plt.tight_layout()
    if mode=="timestep" or mode=="ts":
        plt.savefig(f'./Plots/{name}_{t}_prediction_{plot_text}.png')
    else:
        plt.savefig(f'./Plots/{name}_prediction_{plot_text}.png')
        

def prediction(model, train, name, device):
    train_tensors = Variable(torch.Tensor(train).to(device)).to(device)
    input_final = torch.reshape(train_tensors, (1, 1, train_tensors.shape[-1])).to(device)
    # print(input_final.shape)
    output = model.forward(input_final, device)
    # print("normal: ", output)
    output = unscale_output(output=output, name=name)
    # print("unscaled: ", output)
    return [output]

def predictHourly(date, device, mode="normal", model_num=0, id="", city="", time=-1):
    out_list = []
    if date <= dt.now()-td(days=2):
        train, label, label24 = gld.get_predictDataHourly(date, id=id)
        if type(train)=="str":
            print("error occured please retry with other ID/Name")
            return
        print("\ntraining count", train.shape)
        
        model, optimizer, loss_fn, metric, history = load_own_Model("Hourly", device, loading_mode=mode, t=model_num)
        model24, optimizer24, loss_fn24, metric24, history24 = load_own_Model("Hourly24", device, loading_mode=mode, t=model_num)
        if not id=="" or city=="":
            out_list.append(prediction(model, train[time], "Hourly", device))
            out_list.append(prediction(model24, train[time], "Hourly", device))
            plotting_Prediction_hourly(train[time], out_list, "test", hourly=label[time], hourly24=label24[time], mode=mode, t=model_num)
    else:
        train = gld.get_predictDataHourly(date, id=id)
        model, optimizer, loss_fn, metric, history = load_own_Model("Hourly", device, loading_mode=mode, t=model_num)
        model24, optimizer24, loss_fn24, metric24, history24 = load_own_Model("Hourly24", device, loading_mode=mode, t=model_num)
        if not id=="" or city=="":
            out_list.append(prediction(model, train[time], "Hourly", device))
            out_list.append(prediction(model24, train[time], "Hourly", device))
            plotting_Prediction_hourly(train[time], out_list, "future", mode=mode, t=model_num)

def predictDaily(date, device, mode="normal", model_num=0, id="", city="", time=-1):
    out_list = []
    if date <= dt.now()-td(days=2):
        train, label1, label2, label3, label4, label5, label6, label7 = gld.get_predictDataDaily(date, id=id)
        if type(train)=="str":
            print("error occured please retry with other ID/Name")
            return
        print("\ntraining count", train.shape)
        
        model1, optimizer, loss_fn, metric, history = load_own_Model("Daily", device, loading_mode=mode, t=model_num, input_count=7752)
        model2, optimizer, loss_fn, metric, history = load_own_Model("Daily", device, loading_mode=mode, t=model_num, input_count=7752)
        model3, optimizer, loss_fn, metric, history = load_own_Model("Daily", device, loading_mode=mode, t=model_num, input_count=7752)
        model4, optimizer, loss_fn, metric, history = load_own_Model("Daily", device, loading_mode=mode, t=model_num, input_count=7752)
        model5, optimizer, loss_fn, metric, history = load_own_Model("Daily", device, loading_mode=mode, t=model_num, input_count=7752)
        model6, optimizer, loss_fn, metric, history = load_own_Model("Daily", device, loading_mode=mode, t=model_num, input_count=7752)
        model7, optimizer, loss_fn, metric, history = load_own_Model("Daily", device, loading_mode=mode, t=model_num, input_count=7752)
        if not id=="" or city=="":
            out_list.append(prediction(model1, train, "Daily", device))
            out_list.append(prediction(model2, train, "Daily", device))
            out_list.append(prediction(model3, train, "Daily", device))
            out_list.append(prediction(model4, train, "Daily", device))
            out_list.append(prediction(model5, train, "Daily", device))
            out_list.append(prediction(model6, train, "Daily", device))
            out_list.append(prediction(model7, train, "Daily", device))
            plotting_Prediction_Daily(train, out_list, "test", label1=label1, label2=label2, label3=label3, label4=label4, label5=label5, label6=label6, label7=label7, mode=mode, t=model_num)
    else:
        train = gld.get_predictDataDaily(date, id=id)
        model1, optimizer, loss_fn, metric, history = load_own_Model("Daily", device, loading_mode=mode, t=model_num, input_count=7752)
        model2, optimizer, loss_fn, metric, history = load_own_Model("Daily", device, loading_mode=mode, t=model_num, input_count=7752)
        model3, optimizer, loss_fn, metric, history = load_own_Model("Daily", device, loading_mode=mode, t=model_num, input_count=7752)
        model4, optimizer, loss_fn, metric, history = load_own_Model("Daily", device, loading_mode=mode, t=model_num, input_count=7752)
        model5, optimizer, loss_fn, metric, history = load_own_Model("Daily", device, loading_mode=mode, t=model_num, input_count=7752)
        model6, optimizer, loss_fn, metric, history = load_own_Model("Daily", device, loading_mode=mode, t=model_num, input_count=7752)
        model7, optimizer, loss_fn, metric, history = load_own_Model("Daily", device, loading_mode=mode, t=model_num, input_count=7752)
        if not id=="" or city=="":
            out_list.append(prediction(model1, train, "Daily", device))
            out_list.append(prediction(model2, train, "Daily", device))
            out_list.append(prediction(model3, train, "Daily", device))
            out_list.append(prediction(model4, train, "Daily", device))
            out_list.append(prediction(model5, train, "Daily", device))
            out_list.append(prediction(model6, train, "Daily", device))
            out_list.append(prediction(model7, train, "Daily", device))
            plotting_Prediction_Daily(train, out_list, "future", mode=mode, t=model_num)