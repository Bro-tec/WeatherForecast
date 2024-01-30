import numpy as np
import matplotlib.pyplot as plt
import torch
import os
# import pandas as pd
# import tensorflow as tf
import CollectData.get_learning_data as gld
from sklearn.model_selection import train_test_split
# from tqdm import tqdm
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
# from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
# from keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
import warnings
# import tensorflow as tf

warnings.filterwarnings('ignore')

# trainData = gld.gen_trainDataHourly()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("this devixe uses " + device + "to train data")

# print(tf.compat.v1.keras.backend.tensorflow_backend._get_available_gpus())

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# net = net.to(device) 
# input = input.to(device) 
# labels = labels.to(device) 

# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
model = []

def load_own_Model(name):
    if  os.path.exists(f"./Models/{name}.h5") and os.path.exists(f"./Models/{name}_weights.h5") and os.path.exists(f"./Models/{name}_history.npy"):
        model = load_model(f"./Models/{name}.h5")
        model.load_weights(f"./Models/{name}_weights.h5")
        history=np.load(f'./Models/{name}_history.npy',allow_pickle='TRUE').item()
        print("Model found")
        return model, history
    elif name == "Hourly" or name == "Hourly":
        print("Data not found or not complete")
        model = Sequential()
        model.add(LSTM( input_shape=(85,1), dropout = 0.2, units=22))
        model.add(Dense(6, activation = 'sigmoid' ))
        model.compile( optimizer = "adam" , loss = 'binary_crossentropy' , metrics = ['accuracy'] )
        return model, {}
    else:
        print("Data not found or not complete")
        model = Sequential()
        model.add(LSTM( input_shape=(240,17,1), dropout = 0.2, units=22))
        model.add(Dense(6, activation = 'sigmoid' ))
        model.compile( optimizer = "adam" , loss = 'binary_crossentropy' , metrics = ['accuracy'] )
        return model, {}

def save_own_Model(name, history, model):
    np.save(f'./Models/{name}_history.npy',history)
    model.save(f"./Models/{name}.h5")
    model.save_weights(f"./Models/{name}_weights.h5")
    print("Saved model")

def plotting_hist(history):
    # summarize history for accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.yticks([label/2 for label in plt.yticks()[0]])
    plt.show()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.yticks([label/2 for label in plt.yticks()[0]])
    plt.show()

early_stopper = EarlyStopping( monitor = 'val_acc' , min_delta = 0.0005, patience = 3 )

reduce_lr = ReduceLROnPlateau( monitor = 'val_loss' , patience = 2 , cooldown = 0)

callbacks = [ reduce_lr , early_stopper]

def train_LSTM(train, label):
    X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.01, random_state=42)

    # train_history = model.fit( gld.gen_trainDataHourly(), epochs = 1, batch_size = batch_size, validation_data=2, validation_steps=1, verbose = 1 , callbacks = callbacks)
    train_history = model.fit( X_train, y_train, epochs = epoch_count, batch_size = batch_size, validation_split = 0.01 , verbose = 1 , callbacks = callbacks)
    
    print(train_history.history)
    if len(history) > 0:
        for ky in history.keys():
            extended_values = train_history.history.get(ky, [])
            history[ky] = history.get(ky, []) + extended_values
            history[ky] = [0 if v is None or v == 'nan' else v for v in history[ky]]
            for ki in range(1,epoch_count+1):
                history[ky][-ki] += history[ky][-(epoch_count+1)]
    else:
        history = train_history.history

    score = model.evaluate( X_test, y_test, batch_size = batch_size)
    print( "Accuracy: {:0.4}".format( score[1] ))
    return history, model

epoch_count = 15
batch_size = 24


# for train, label, label24 in gld.gen_trainDataHourly():
#     model, history = load_own_Model("Hourly")
#     history, model = train_LSTM(train, label)
#     save_own_Model("Hourly", history, model)
#     plotting_hist(history)

#     model, history = load_own_Model("Hourly24")
#     history, model = train_LSTM(train, label)
#     save_own_Model("Hourly24", history, model)
#     plotting_hist(history)

train_np = np.array([])
for train, label_Daily, label_monthly in gld.gen_trainDataDaily():
    

    model, history = load_own_Model("Daily")
    history, model = train_LSTM(train, label_Daily)
    save_own_Model("Daily", history, model)
    plotting_hist(history)

    model, history = load_own_Model("Weekly")
    history, model = train_LSTM(train, label_monthly)
    save_own_Model("Weekly", history, model)
    plotting_hist(history)

    model, history = load_own_Model("Monthly")
    history, model = train_LSTM(train, label_monthly)
    save_own_Model("Monthly", history, model)
    plotting_hist(history)
