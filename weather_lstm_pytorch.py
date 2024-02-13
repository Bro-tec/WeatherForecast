import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
import json
# import torch.utils.data as data
 
# from keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
import warnings
# import tensorflow as tf

warnings.filterwarnings('ignore')

# trainData = gld.gen_trainDataHourly()
def check_cuda():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("this device uses " + device + " to train data")
    return torch.device(device)

 
class PyTorch_LSTM(nn.Module):
    def __init__(self, input, device, hidden=120, layer=1, outputs=6):
        self.output_size = outputs
        self.num_layer = layer
        self.hidden_size = hidden
        super(PyTorch_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input, hidden_size=hidden, num_layers=layer, bidirectional=True, batch_first=True).to(device)
        self.linear = nn.Linear(hidden*2, 30).to(device)
        self.fc = nn.Linear(30, outputs).to(device)
        self.soft = nn.Softmax().to(device)
    def forward(self, input, device):
        hidden_state = Variable(torch.zeros(self.num_layer*2, input.size(0), self.hidden_size).to(device)).to(device)
        cell_state = Variable(torch.zeros(self.num_layer*2, input.size(0), self.hidden_size).to(device)).to(device)
        out, (hn, cn) = self.lstm(input, (hidden_state, cell_state))
        
        out = out[:,-1,:]
        # out = out.view(-1, ) #reshaping the data for Dense layer next
        out = self.soft(out)
        out = self.linear(out) #first Dense
        out = self.soft(out) #relu
        out = self.fc(out)
        return out

def load_own_Model(name, device):
    if os.path.exists(f"./Models/{name}.pth"): #and os.path.exists(f"./Models/{name}_weights.h5"):# and os.path.exists(f"./Models/{name}_history.npy"):
        # model = load_model(f"./Models/{name}.h5")
        # model.load_weights(f"./Models/{name}_weights.h5")
        model = PyTorch_LSTM(85, device)
        model.load_state_dict(torch.load(f"./Models/{name}.pth"))
        model.eval()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        loss_fn = nn.KLDivLoss().to(device)
        # loss_fn = nn.MSELoss()
        # history=np.load(f'./Models/{name}_history.npy',allow_pickle='TRUE').item()
        with open(f'./Models/{name}_history.json', 'r') as f:
            history = json.load(f)
        print("Model found")
        return model, optimizer, loss_fn, history
    elif name == "Hourly" or name == "Hourly24":
        print("Data not found or not complete")
        model = PyTorch_LSTM(85, device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.KLDivLoss().to(device)
        # l = nn.BCELoss()
        # loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
        return model, optimizer, loss_fn, {"accuracy":[0], "loss":[0], "val_accuracy":[0], "val_loss":[0]}
    else:
        # print("Data not found or not complete")
        # model = PyTorch_LSTM(85, 17)
        print("Data not found or not complete")
        model = PyTorch_LSTM(153, device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.KLDivLoss().to(device)
        # loss_fn = nn.MSELoss()
        return model, optimizer, loss_fn, {"accuracy":[0], "loss":[0], "val_accuracy":[0], "val_loss":[0]}

def save_own_Model(name, history, model):
    # np.save(f'./Models/{name}_history.npy',history)
    with open(f"./Models/{name}_history.json", "w") as fp:  
        json.dump(history, fp)
    torch.save(model.state_dict(), f"./Models/{name}.pth")
    # model.save(f"./Models/{name}.h5")
    # model.save_weights(f"./Models/{name}_weights.h5")
    print("Saved model")

# def plotting_hist(history, name, i):
#     # summarize history for accuracy
#     plt.plot(history['accuracy'])
#     plt.plot(history['loss'])
#     # plt.plot(history['val_accuracy'])
#     plt.title('model accuracy-loss')
#     plt.ylabel('y')
#     plt.xlabel('epoch')
#     plt.legend(history.keys(), loc='upper left')
#     # plt.yticks([label/2 for label in plt.yticks()[0]])
#     plt.savefig(f'./Plots/{name}_plot_{i}.png')
#     plt.close()
    
def plotting_hist(history, name, i):
    # summarize history for accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.yticks([label/2 for label in plt.yticks()[0]])
    plt.savefig(f'./Plots/{name}_plot_{i}_accuracy.png')
    plt.close()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.yticks([label/2 for label in plt.yticks()[0]])
    plt.savefig(f'./Plots/{name}_plot_{i}_loss.png')
    plt.close()
    

def train_LSTM(feature, label, model, optimizer, loss_fn, history, device, epoch_count=1):
    scaled_X = StandardScaler().fit_transform(feature)
    scaled_y = MinMaxScaler().fit_transform(label)
    

    X_train, X_test, y_train, y_test = train_test_split(scaled_X, scaled_y, test_size=0.02, random_state=42)
    # print("X_train:",X_train.shape, ", X_test:", X_test.shape, ", y_train:", y_train.shape, ", y_test:", y_test.shape)
    
    X_train_tensors = Variable(torch.Tensor(X_train).to(device)).to(device)
    X_test_tensors = Variable(torch.Tensor(X_test).to(device)).to(device)

    y_train_tensors = Variable(torch.Tensor(y_train)).to(device)
    y_test_tensors = Variable(torch.Tensor(y_test)).to(device)
    # print("X_train_tensors:",X_train_tensors.shape, ", X_test_tensors:", X_test_tensors.shape, ", y_train_tensors:", y_train_tensors.shape, ", y_test_tensors:", y_test_tensors.shape)
    
    X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1])).to(device)
    X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])).to(device) 
    # print("X_train_tensors:",X_train_tensors_final.shape, ", X_test_tensors:", X_test_tensors_final.shape, ", y_train_tensors:", y_train_tensors.shape, ", y_test_tensors:", y_test_tensors.shape)
    
    
    for epoch in range(epoch_count):
        for input, label in tqdm(zip(X_train_tensors_final, y_train_tensors), total=X_train_tensors_final.shape[0]):
            input_final = torch.reshape(input, (input.shape[0], 1, input.shape[1])).to(device)
            outputs = model.forward(input_final, device) #forward pass
            
            optimizer.zero_grad() #caluclate the gradient, manually setting to 0
            # obtain the loss function
            loss = loss_fn(outputs, label)
            loss.backward(retain_graph=True) #calculates the loss of the loss function
            optimizer.step() #improve from loss, i.e backprop
            
        history['loss'].append(float(loss.item())) #+history['loss'][-1])
        output = (outputs>0.5).float()
        correct = (output == y_train_tensors).float().sum()
        acc = y_train_tensors.shape[0]/correct
        history['accuracy'].append(float(acc)) 
        print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch+1,epoch_count, loss, acc))
    
        outputs = model.forward(X_test_tensors_final, device) #forward pass
        loss = loss_fn(outputs, y_test_tensors)
        history['val_loss'].append(float(loss.item())) #+history['loss'][-1])
        output = (outputs>0.5).float()
        correct = (output == y_test_tensors).float().sum()
        acc = y_test_tensors.shape[0]/correct
        history['val_accuracy'].append(float(acc)) 
        print("val Loss: {:.3f}, val Accuracy: {:.3f}".format(loss, acc))
    
    return model, history
        
    # train_history = model.fit( gld.gen_trainDataHourly(), epochs = 1, batch_size = batch_size, validation_data=2, validation_steps=1, verbose = 1 , callbacks = callbacks)
    # train_history = model.fit( X_train, y_train, epochs = epoch_count, batch_size = batch_size, validation_split = 0.01 , verbose = 1 , callbacks = callbacks)
    
    # # print(train_history.history)
    # if len(history) > 0:
    #     for ky in history.keys():
    #         extended_values = train_history.history.get(ky, [])
    #         history[ky] = history.get(ky, []) + extended_values
    #         history[ky] = [0 if v is None or v == 'nan' else v for v in history[ky]]
    #         for ki in range(1,epoch_count+1):
    #             history[ky][-ki] += history[ky][-(epoch_count+1)]
    # else:
    #     history = train_history.history

    # score = model.evaluate( X_test, y_test, batch_size = batch_size)
    # print( "Accuracy: {:0.4}".format( score[1] ))
    # return history, model


