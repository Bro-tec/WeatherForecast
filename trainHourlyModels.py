import weather_lstm_pytorch as wl
import CollectData.get_learning_data as gld
from datetime import datetime as dt


start_time = dt.now()
times_list = []

device = wl.check_cuda()

# round about 34:03 min each fit for 200 cities due to data loading (mostly because of concats)
# round about 1 to 10 min each fit for all cities due to async

# if u already trained some days and dont want to retrain them type:
# skip_days=<days> in gld.gen_trainDataHourly_Async
for train, label, label24, i in gld.gen_trainDataHourly_Async():
    print("\ntraining count", train.shape)

    mode = "timestep"
    
    name = "Hourly"
    model, optimizer, loss_fn, metric, history = wl.load_own_Model(name, device, loading_mode=mode, t=i, learning_rate=0.000000005 ) # 1/(5160*1561*24) ungefair 5e-9
    model, history = wl.train_LSTM(name,train, label, model, optimizer, loss_fn, metric, history, device, epoch_count=1)
    wl.save_own_Model(name, history, model, saving_mode=mode, t=i)
    wl.plotting_hist(history, metric, name, saving_mode=mode, t=i)
    
    name = "Hourly24"
    model, optimizer, loss_fn, metric, history = wl.load_own_Model(name, device, loading_mode=mode, t=i)
    model, history = wl.train_LSTM(name, train, label, model, optimizer, loss_fn, metric, history, device, epoch_count=1)
    wl.save_own_Model(name, history, model, saving_mode=mode, t=i)
    wl.plotting_hist(history, metric, name, saving_mode=mode, t=i)
    
    times_list.append(dt.now()-start_time)
    print("actually it took: ", times_list[-1])
    start_time = dt.now()
    print("\n\n")