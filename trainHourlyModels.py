import weather_lstm_pytorch as wl
import CollectData.get_learning_data as gld
from datetime import datetime as dt


start_time = dt.now()
times_list = []

device = wl.check_cuda()

name = "Hourly"
skip = 0
batchsize = 500
epoches = 1
dropout = 0.2
# round about 34:03 min each fit for 200 cities due to data loading (mostly because of concats)
# round about 1 to 10 min each fit for all cities due to async

# if u already trained some days and dont want to retrain them type:
# skip_days=<days> in gld.gen_trainDataHourly_Async
# mode = input(
#     "Decide with which mode you want to save your model.\nTimestemp(ts)  - your code will be saved into multiple models to be able to obtain a model before overfitting\nNormal(n)   - if you chose something invalid your model will automatically be setted to Normal\n Choose: "
# )
# mode = mode.lower()
# if mode == "timestep" or mode == "ts":
#     skip = ""
#     while not skip.isdigit():
#         skip = input(
#             "Enter the number of the step you want to continue with (, if you just started enter 0): "
#         )
#     skip = int(skip)
# else:
#     mode = "normal"

mode = "normal"

# , label24
for train, label, i in gld.gen_trainDataHourly_Async(skip_days=skip):
    print("\ntraining count: ", train.shape)
    print("\nlabel count: ", label.shape)
    # print("\nlabel24 count: ", label24.shape)
    print("\ni count: ", i)

    if train.shape[0] >= 2:
        model, optimizer, loss_fn, history = wl.load_own_Model(
            name,
            device,
            loading_mode=mode,
            t=i,
            dropout=dropout,
        )  # 1/(5160*1561*24) ungefair 5e-9
        for epoch in range(epoches):
            model, history, metrics = wl.train_LSTM(
                name,
                train,
                label,
                model,
                optimizer,
                loss_fn,
                history,
                device,
                epoch_count=1,
                batchsize=batchsize,
            )
            wl.save_own_Model(name, history, model, saving_mode=mode, t=i)
            wl.plotting_hist(history, metrics, name, saving_mode=mode, t=i)

            times_list.append(dt.now() - start_time)
            print("actually it took: ", times_list[-1])
            start_time = dt.now()
            print("\n\n")
