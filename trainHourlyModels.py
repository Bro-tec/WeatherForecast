import weather_lstm_pytorch as wl
import CollectData.get_learning_data as gld
from datetime import datetime as dt


start_time = dt.now()
times_list = []

device = wl.check_cuda()

name = ["working"]
# skip = 2594  # full_feature_layer1
# skip = 6  # new_features
# skip = 327  # model_PX
skip = 4641  # working
# skip = 0
month = True
hours = True
position = True

inputs = 270
outputs = 54
if month:
    inputs += 1
if hours:
    inputs += 1

if position:
    inputs += 3

batchsize = 100
epoches = 3
repeat = 3
dropout = 0.1
learning_rate = [0.0001]
layers = [3]

hiddens = [100]
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

for n in range(len(name)):
    print(name[n])
    wl.create_own_Model(
        name[n],
        device,
        input_count=inputs,
        output_count=outputs,
        dropout=dropout,
        hiddensize=hiddens[n],
        sequences=sequences,
        batchsize=batchsize,
        learning_rate=learning_rate[n],
        layer=layers[n],
        month=month,
        hours=hours,
        position=position,
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
        for n in range(len(name)):
            print(name[n])
            model, optimizer, history = wl.load_own_Model(
                name[n],
                device,
            )
            for epoch in range(epoches):
                model, history, metrics, optimizer = wl.train_LSTM(
                    name[n],
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
                wl.save_own_Model(name[n], history, model, optimizer, device)
                wl.plotting_hist(history, metrics, name[n], 9, epoche=epoch)
                del metrics

                times_list.append(dt.now() - start_time)
                print("actually it took: ", times_list[-1])
                start_time = dt.now()
                print("\n")

            del model, optimizer, history
