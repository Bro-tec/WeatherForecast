import weather_lstm_pytorch as wl
import CollectData.get_learning_data as gld
from datetime import datetime as dt
import numpy as np

start_time = dt.now()
times_list = []

device = wl.check_cuda()

# round about 34:03 min each fit for 200 cities due to data loading (mostly because of concats)
# round about 1 to 10 min each fit for all cities due to async

# if u already trained some days and dont want to retrain them type:
# skip_days=<days> in gld.gen_trainDataHourly_Async
for train, label_Daily1, label_Daily2, label_Daily3, label_Daily4, label_Daily5, label_Daily6, label_Daily7, i in gld.gen_trainDataDaily_Async(skip_days=1500):
    print(train.shape)
    print(label_Daily1.shape)
    
    mode = "timestep"
    name = "Day1"
    model, optimizer, loss_fn, metric, history = wl.load_own_Model(name, device, loading_mode=mode, t=i, input_count=7752) # 1/(5160*1561) ungefair 1e-7
    model, history = wl.train_LSTM(name,train, label_Daily1, model, optimizer, loss_fn, metric, history, device, epoch_count=2)
    wl.save_own_Model(name, history, model, saving_mode=mode, t=i)
    wl.plotting_hist(history, metric, name, i)

    name = "Day2"
    model, optimizer, loss_fn, metric, history = wl.load_own_Model(name, device, loading_mode=mode, t=i, input_count=7752)
    model, history = wl.train_LSTM(name,train, label_Daily2, model, optimizer, loss_fn, metric, history, device, epoch_count=2)
    wl.save_own_Model(name, history, model, saving_mode=mode, t=i)
    wl.plotting_hist(history, metric, name, i)

    name = "Day3"
    model, optimizer, loss_fn, metric, history = wl.load_own_Model(name, device, loading_mode=mode, t=i, input_count=7752)
    model, history = wl.train_LSTM(name,train, label_Daily3, model, optimizer, loss_fn, metric, history, device, epoch_count=2)
    wl.save_own_Model(name, history, model, saving_mode=mode, t=i)
    wl.plotting_hist(history, metric, name, i)
    
    name = "Day4"
    model, optimizer, loss_fn, metric, history = wl.load_own_Model(name, device, loading_mode=mode, t=i, input_count=7752)
    model, history = wl.train_LSTM(name,train, label_Daily4, model, optimizer, loss_fn, metric, history, device, epoch_count=2)
    wl.save_own_Model(name, history, model, saving_mode=mode, t=i)
    wl.plotting_hist(history, metric, name, i)
    
    name = "Day5"
    model, optimizer, loss_fn, metric, history = wl.load_own_Model(name, device, loading_mode=mode, t=i, input_count=7752)
    model, history = wl.train_LSTM(name,train, label_Daily5, model, optimizer, loss_fn, metric, history, device, epoch_count=2)
    wl.save_own_Model(name, history, model, saving_mode=mode, t=i)
    wl.plotting_hist(history, metric, name, i)
    
    name = "Day6"
    model, optimizer, loss_fn, metric, history = wl.load_own_Model(name, device, loading_mode=mode, t=i, input_count=7752)
    model, history = wl.train_LSTM(name,train, label_Daily6, model, optimizer, loss_fn, metric, history, device, epoch_count=2)
    wl.save_own_Model(name, history, model, saving_mode=mode, t=i)
    wl.plotting_hist(history, metric, name, i)
    
    name = "Day7"
    model, optimizer, loss_fn, metric, history = wl.load_own_Model(name, device, loading_mode=mode, t=i, input_count=7752)
    model, history = wl.train_LSTM(name,train, label_Daily7, model, optimizer, loss_fn, metric, history, device, epoch_count=2)
    wl.save_own_Model(name, history, model, saving_mode=mode, t=i)
    wl.plotting_hist(history, metric, name, i)