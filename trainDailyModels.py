import weather_lstm_pytorch as wl
import CollectData.get_learning_data as gld
from datetime import datetime as dt

start_time = dt.now()
times_list = []

device = wl.check_cuda()

skip = 0

# round about 34:03 min each fit for 200 cities due to data loading (mostly because of concats)
# round about 1 to 10 min each fit for all cities due to async

# if u already trained some days and dont want to retrain them type:
# skip_days=<days> in gld.gen_trainDataHourly_Async

mode = input(
    "Decide with which mode you want to save your model.\nTimestemp(ts)  - your code will be saved into multiple models to be able to obtain a model before overfitting\nNormal(n)   - if you chose something invalid your model will automatically be setted to Normal\n Choose: "
)
mode = mode.lower()
if mode == "timestep" or mode == "ts":
    skip = ""
    while not skip.isdigit():
        skip = input(
            "Enter the number of the step you want to continue with (, if you just started enter 0): "
        )
    skip = int(skip)
else:
    mode = "normal"


for (
    train,
    label_Daily1,
    label_Daily2,
    label_Daily3,
    label_Daily4,
    label_Daily5,
    label_Daily6,
    label_Daily7,
    i,
) in gld.gen_trainDataDaily_Async(skip_days=skip):
    print(train.shape)
    print(label_Daily1.shape)

    name = "Day1"
    model, optimizer, loss_fn, metric, history = wl.load_own_Model(
        name, device, loading_mode=mode, t=i, input_count=7752
    )  # 1/(5160*1561) ungefair 1e-7
    model, history = wl.train_LSTM(
        name,
        train,
        label_Daily1,
        model,
        optimizer,
        loss_fn,
        metric,
        history,
        device,
        epoch_count=1,
    )
    wl.save_own_Model(name, history, model, saving_mode=mode, t=i)
    wl.plotting_hist(history, metric, name, saving_mode=mode, t=i)

    name = "Day2"
    model, optimizer, loss_fn, metric, history = wl.load_own_Model(
        name, device, loading_mode=mode, t=i, input_count=7752
    )
    model, history = wl.train_LSTM(
        name,
        train,
        label_Daily2,
        model,
        optimizer,
        loss_fn,
        metric,
        history,
        device,
        epoch_count=1,
    )
    wl.save_own_Model(name, history, model, saving_mode=mode, t=i)
    wl.plotting_hist(history, metric, name, saving_mode=mode, t=i)

    name = "Day3"
    model, optimizer, loss_fn, metric, history = wl.load_own_Model(
        name, device, loading_mode=mode, t=i, input_count=7752
    )
    model, history = wl.train_LSTM(
        name,
        train,
        label_Daily3,
        model,
        optimizer,
        loss_fn,
        metric,
        history,
        device,
        epoch_count=1,
    )
    wl.save_own_Model(name, history, model, saving_mode=mode, t=i)
    wl.plotting_hist(history, metric, name, saving_mode=mode, t=i)

    name = "Day4"
    model, optimizer, loss_fn, metric, history = wl.load_own_Model(
        name, device, loading_mode=mode, t=i, input_count=7752
    )
    model, history = wl.train_LSTM(
        name,
        train,
        label_Daily4,
        model,
        optimizer,
        loss_fn,
        metric,
        history,
        device,
        epoch_count=1,
    )
    wl.save_own_Model(name, history, model, saving_mode=mode, t=i)
    wl.plotting_hist(history, metric, name, saving_mode=mode, t=i)

    name = "Day5"
    model, optimizer, loss_fn, metric, history = wl.load_own_Model(
        name, device, loading_mode=mode, t=i, input_count=7752
    )
    model, history = wl.train_LSTM(
        name,
        train,
        label_Daily5,
        model,
        optimizer,
        loss_fn,
        metric,
        history,
        device,
        epoch_count=1,
    )
    wl.save_own_Model(name, history, model, saving_mode=mode, t=i)
    wl.plotting_hist(history, metric, name, saving_mode=mode, t=i)

    name = "Day6"
    model, optimizer, loss_fn, metric, history = wl.load_own_Model(
        name, device, loading_mode=mode, t=i, input_count=7752
    )
    model, history = wl.train_LSTM(
        name,
        train,
        label_Daily6,
        model,
        optimizer,
        loss_fn,
        metric,
        history,
        device,
        epoch_count=1,
    )
    wl.save_own_Model(name, history, model, saving_mode=mode, t=i)
    wl.plotting_hist(history, metric, name, saving_mode=mode, t=i)

    name = "Day7"
    model, optimizer, loss_fn, metric, history = wl.load_own_Model(
        name, device, loading_mode=mode, t=i, input_count=7752
    )
    model, history = wl.train_LSTM(
        name,
        train,
        label_Daily7,
        model,
        optimizer,
        loss_fn,
        metric,
        history,
        device,
        epoch_count=1,
    )
    wl.save_own_Model(name, history, model, saving_mode=mode, t=i)
    wl.plotting_hist(history, metric, name, saving_mode=mode, t=i)
