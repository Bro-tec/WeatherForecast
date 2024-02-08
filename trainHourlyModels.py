import weather_lstm as wl
import CollectData.get_learning_data as gld


# round about 34:03 min each fit due to data loading (mostly because of concats)
for train, label, label24, i in gld.gen_trainDataHourly():
    print("training count", train.shape[0])

    name = "Hourly"
    model, history = wl.load_own_Model(name)
    history, model = wl.train_LSTM(train, label, model, history, epoch_count=2, batch_size=240)
    wl.save_own_Model(name, history, model)
    wl.plotting_hist(history, name, i)
    
    
    name = "Hourly24"
    model, history = wl.load_own_Model(name)
    history, model = wl.train_LSTM(train, label24, model, history, epoch_count=2, batch_size=240)
    wl.save_own_Model(name, history, model)
    wl.plotting_hist(history, name, i)