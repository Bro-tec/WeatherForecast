import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import CollectData.get_learning_data as gld
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

trainData = gld.gen_trainDataHourly()
for i in range(1):
    train, label = next(trainData)
    print(train.columns)
    print(label)
        