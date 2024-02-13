import weather_lstm_pytorch as wl
import CollectData.get_learning_data as gld
from datetime import datetime as dt

start_time = dt.now()
times_list = []

device = wl.check_cuda()

# round about 34:03 min each fit due to data loading (mostly because of concats)
for train, label, label24, i in gld.gen_trainDataHourly_Async():
    print("training count", train.shape)

    name = "Hourly"
    model, optimizer, loss_fn, history = wl.load_own_Model(name, device)
    model, history = wl.train_LSTM(train, label, model, optimizer, loss_fn, history, device, epoch_count=2)
    wl.save_own_Model(name, history, model)
    wl.plotting_hist(history, name, i)
    
    
    name = "Hourly24"
    model, optimizer, loss_fn, history = wl.load_own_Model(name, device)
    model, history = wl.train_LSTM(train, label, model, optimizer, loss_fn, history, device, epoch_count=2)
    wl.save_own_Model(name, history, model)
    wl.plotting_hist(history, name, i)
    
    times_list.append(dt.now()-start_time)
    print("actually it took: ", times_list[-1])
    start_time = dt.now()