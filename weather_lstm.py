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
import tensorflow as tf

warnings.filterwarnings('ignore')

# trainData = gld.gen_trainDataHourly()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# print(tf.compat.v1.keras.backend.tensorflow_backend._get_available_gpus())

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# net = net.to(device) 
# input = input.to(device) 
# labels = labels.to(device) 

# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
model = []

def load_own_Model():
    if os.path.exists("./Models/Hourly.h5"):
        model = load_model("./Models/Hourly.h5")
        model.load_weights("./Models/Hourly_weights.h5")
        history=np.load('./Models/Hourly_history.npy',allow_pickle='TRUE').item()
        print("Model found")
        return model, history
    else:
        print("Model not found")
        model = Sequential()
        model.add(LSTM( input_shape=(85,1), dropout = 0.2, units=52))
        model.add(Dense(17, activation = 'sigmoid' ))
        model.compile( optimizer = "adam" , loss = 'binary_crossentropy' , metrics = ['accuracy'] )
        return model, [] 
# model.summary()
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=2)


early_stopper = EarlyStopping( monitor = 'val_acc' , min_delta = 0.0005, patience = 3 )

reduce_lr = ReduceLROnPlateau( monitor = 'val_loss' , patience = 2 , cooldown = 0)

callbacks = [ reduce_lr , early_stopper]





batch_size = 24

X_np = np.zeros(shape=(1, 1))
y_np = np.zeros(shape=(1, 1))
i=1

for train, label in gld.gen_trainDataHourly():

    if i%25 == 1:
        X_np = train
        y_np = label
    else:
        X_np += train
        y_np += label
    print(i)


    if i%25 == 0:
        # Assume X and y are your features and labels
        X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.3, random_state=42)
        # print("X_train",X_train, "X_test",X_test, "y_train",y_train, "y_test",y_test)
        # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
        # print("X_train",len(X_train), "X_test",len(X_test), "X_val", len(X_val), "y_train",len(y_train), "y_test",len(y_test), "y_val", len(y_val))
        # print("X_train",len(X_train), "X_test",len(X_test), "y_train",len(y_train), "y_test",len(y_test))

        model, history = load_own_Model()

        train_history = model.fit( X_train, y_train, epochs = 2, batch_size = batch_size, validation_split = 0.1 , verbose = 1 , callbacks = callbacks)
        
        (len(history))
        if len(history) > 0:
            # print("train_history.history", train_history.history)
            # print("history1", history)
            for ky in history.keys():
                extended_values = train_history.history.get(ky, [])
                history[ky] = history.get(ky, []) + extended_values
                history[ky] = [0 if v is None else v for v in history[ky]]
        else:
            history = dict(train_history.history)
            # print("history1", history)
            # print("train_history", train_history.history)

        score = model.evaluate( X_test, y_test, batch_size = batch_size)

        print( "Accuracy: {:0.4}".format( score[1] ))
        np.save('./Models/Hourly_history.npy',history)
        model.save("./Models/Hourly.h5")
        model.save_weights("./Models/Hourly_weights.h5")
# train_history = model.fit( gld.gen_trainDataHourly(), epochs = 1, batch_size = batch_size, validation_data=2, validation_steps=1, verbose = 1 , callbacks = callbacks)

# history = model.fit_generator(
#             train_generator,
#             steps_per_epoch=24,
#             epochs=5,
#             validation_data=2,
#             validation_steps=1,
#             class_weight=17,
#             initial_epoch=1,
#             max_queue_size=15,
#             workers=8,
#             callbacks=callbacks
#             )
        # score = model.evaluate( gld.gen_trainDataHourly(), batch_size = batch_size)

        X_np = []
        y_np = []

        print(train_history.history.keys())

        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.yticks([label/2 for label in plt.yticks()[0]])
        plt.show()
        # summarize history for loss
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.yticks([label/2 for label in plt.yticks()[0]])
        plt.show()


    i+=1

print("Saved model to disk")