import numpy as np
import matplotlib.pyplot as plt
import os
# import pandas as pd
# import tensorflow as tf
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

warnings.filterwarnings('ignore')

def load_own_Model(name):
    if  os.path.exists(f"./Models/{name}.h5") and os.path.exists(f"./Models/{name}_weights.h5") and os.path.exists(f"./Models/{name}_history.npy"):
        model = load_model(f"./Models/{name}.h5")
        model.load_weights(f"./Models/{name}_weights.h5")
        history=np.load(f'./Models/{name}_history.npy',allow_pickle='TRUE').item()
        print("Model found")
        return model, history
    elif name == "Hourly" or name == "Hourly24":
        print("Data not found or not complete")
        model = Sequential()
        model.add(LSTM( input_shape=(85,1), units=22))
        model.add(Dense(6, activation = 'sigmoid' ))
        model.compile( optimizer = "adam" , loss = 'binary_crossentropy' , metrics = ['accuracy'] ) 
        return model, {}
    else:
        print("Data not found or not complete")
        model = Sequential()
        model.add(LSTM( input_shape=(240,17), dropout = 0.2, units=22, activation='linear'))
        model.add(Dense(6, activation = 'sigmoid' ))
        model.compile( optimizer = "adam" , loss = 'binary_crossentropy' , metrics = ['accuracy'] )
        return model, {}
    
    # could 'categorical_crossentropy' work here?
    # In the case of Binary classification: two exclusive classes, you need to use binary cross entropy.
    # In the case of Multi-class classification: more than two exclusive classes, you need to use categorical cross entropy.
    # In the case of Multi-label classification: just non-exclusive classes, you need to use binary cross entropy.
    # found this so i choosed binary cross entropy

    # https://stats.stackexchange.com/questions/260505/should-i-use-a-categorical-cross-entropy-or-binary-cross-entropy-loss-for-binary

def save_own_Model(name, history, model):
    np.save(f'./Models/{name}_history.npy',history)
    model.save(f"./Models/{name}.h5")
    model.save_weights(f"./Models/{name}_weights.h5")
    print("Saved model")

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

def create_callbacks():
    early_stopper = EarlyStopping( monitor = 'val_acc' , min_delta = 0.0005, patience = 3 )
    reduce_lr = ReduceLROnPlateau( monitor = 'val_loss' , patience = 2 , cooldown = 0)
    return [ reduce_lr , early_stopper]

def train_LSTM(train, label, model, history, epoch_count=1, batch_size=10):
    callbacks = create_callbacks()

    X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.01, random_state=42)

    # train_history = model.fit( gld.gen_trainDataHourly(), epochs = 1, batch_size = batch_size, validation_data=2, validation_steps=1, verbose = 1 , callbacks = callbacks)
    train_history = model.fit( X_train, y_train, epochs = epoch_count, batch_size = batch_size, validation_split = 0.01 , verbose = 1 , callbacks = callbacks)
    
    # print(train_history.history)
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


