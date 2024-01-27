import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import tensorflow as tf
import CollectData.get_learning_data as gld
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
# from keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error

# trainData = gld.gen_trainDataHourly()


model = Sequential()
model.add(LSTM( input_shape=(85,1), dropout = 0.2, units=52))
model.add(Dense(17, activation = 'sigmoid' ))
# model.summary()
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=2)

model.compile( optimizer = "adam" , loss = 'binary_crossentropy' , metrics = ['accuracy'] )

early_stopper = EarlyStopping( monitor = 'val_acc' , min_delta = 0.0005, patience = 3 )

reduce_lr = ReduceLROnPlateau( monitor = 'val_loss' , patience = 2 , cooldown = 0)

callbacks = [ reduce_lr , early_stopper]





batch_size = 70

X_list = []
y_list = []
i=1

for train, label in gld.gen_trainDataHourly():
    print(i)
    # print(train, label)
    X_list.extend(train) 
    y_list.extend(label)

    if i%3 == 0:
        X_np = np.array(X_list)
        y_np = np.array(y_list)
        X_np = X_np.astype(float)
        y_np = y_np.astype(float)

        # Assume X and y are your features and labels
        X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.3, random_state=42)
        # print("X_train",X_train, "X_test",X_test, "y_train",y_train, "y_test",y_test)
        # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
        # print("X_train",len(X_train), "X_test",len(X_test), "X_val", len(X_val), "y_train",len(y_train), "y_test",len(y_test), "y_val", len(y_val))
        print("X_train",len(X_train), "X_test",len(X_test), "y_train",len(y_train), "y_test",len(y_test))
        

        # encoder = LabelEncoder()

        # encoder.fit(y_train)

        # y_train_transformed = encoder.transform(y_train).reshape(-1,1)

        # y_test_transformed = encoder.transform(y_test).reshape(-1,1)

        # print(y_train_transformed)
        # print(y_test_transformed)

        # X_train_seq = pad_sequences( X_train)

        # X_test_seq = pad_sequences( X_test)

        # break

        train_history = model.fit( X_train, y_train, epochs = 5, validation_split = 0.1 , verbose = 1 , callbacks = callbacks)
        #, batch_size = 24

        score = model.evaluate( X_test , y_test , batch_size = batch_size)

        print( "Accuracy: {:0.4}".format( score[1] ))

        X_list = []
        y_list = []

    i+=1
