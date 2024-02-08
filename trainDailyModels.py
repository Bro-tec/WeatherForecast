import weather_lstm as wl
import CollectData.get_learning_data as gld
import numpy as np


train_np = np.array([])
for train, label_Daily, i in gld.gen_trainDataDaily():

    name = "Daily"
    model, history = wl.load_own_Model(name)
    history, model = wl.train_LSTM(train, label_Daily, model, history, epoch_count=1)
    wl.save_own_Model(name, history, model)
    wl.plotting_hist(history, name, i)

    # name = "Weekly"
    # model, history = wl.load_own_Model(name)
    # history, model = wl.train_LSTM(train, label_monthly, model )
    # wl.save_own_Model(name, history, model)
    # wl.plotting_hist(history, name, i)

    # name = "Monthly"
    # model, history = wl.load_own_Model(name)
    # history, model = wl.train_LSTM(train, label_monthly, model)
    # wl.save_own_Model(name, history, model)
    # wl.plotting_hist(history, name, i)