import weather_lstm_pytorch as wl
import CollectData.get_learning_data as gld
from datetime import datetime as dt


start_time = dt.now()
times_list = []

device = wl.check_cuda()

name = "test"
# skip = 2594  # full_feature_layer1
# skip = 6  # new_features
skip = 0  # test
month = True
hours = True
position = True

inputs = 265
outputs = 54
if month:
    inputs += 1
if hours:
    inputs += 1
if position:
    inputs += 3

batchsize = 100
epoches = 9999
repeat = 3
dropout = 0.2
learning_rate = 0.0001
layers = 1

hiddens = 1
sequences = 12
hours_per_city = 24
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

# , label24

model, optimizer, history = wl.load_own_Model(
    name,
    device,
    input_count=inputs,
    output_count=outputs,
    dropout=dropout,
    hiddensize=hiddens,
    sequences=sequences,
    batchsize=batchsize,
    learning_rate=learning_rate,
    layer=layers,
)

for train, label, i, r in gld.gen_trainDataHourly_Async(
    skip_days=skip,
    seq=sequences,
    max_batch=hours_per_city,
    redos=repeat,
    month=month,
    hours=hours,
    position=position,
):
    print("\ntraining count: ", train.shape)
    print("\nlabel count: ", label.shape)
    # print("\nlabel24 count: ", label24.shape)
    print("\n Days count: ", i, ", repeated ", repeat, "\\", r, "\n")

    if train.shape[0] >= 2:
        for epoch in range(epoches):
            model, history, metrics, optimizer = wl.train_LSTM(
                name,
                train,
                label,
                model,
                optimizer,
                history,
                device,
                epoch_count=epoches,
                epoch=epoch,
                batchsize=batchsize,
            )
            wl.save_own_Model(name, history, model, optimizer)
            wl.plotting_hist(history, metrics, name, 10, epoche=epoch)

            times_list.append(dt.now() - start_time)
            print("actually it took: ", times_list[-1])
            start_time = dt.now()
            print("\n")
