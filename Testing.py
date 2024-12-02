import weather_lstm_pytorch as wl
import CollectData.get_learning_data as gld
from datetime import datetime as dt


start_time = dt.now()
times_list = []

device = wl.check_cuda()

n = "t"
# skip = 2594  # full_feature_layer1
# skip = 6  # new_features
skip = 0  # test
month = True
hours = True
position = True

inputs = 230
outputs = 46
if month:
    inputs += 1
if hours:
    inputs += 1
if position:
    inputs += 3

batchsizes = [10000, 1000, 500, 200, 100]
epoches = 100
repeat = 1
dropout = 0.1
learning_rates = [0.0001, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
layers = 5

hiddens = 1
sequences = 12
hours_per_city = 24

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
    print("label count: ", label.shape)
    # print("\nlabel24 count: ", label24.shape)
    print("\n Days count: ", i, ", repeated ", repeat, "\\", r, "\n")
    for batchsize in batchsizes:
        for learning_rate in learning_rates:
            for l in range(1, layers):
                for i in range(1, inputs):
                    name = (
                        n
                        + "_e"
                        + str(i)
                        + "_l"
                        + str(l)
                        + "_lr"
                        + str(learning_rate)
                        + "_b"
                        + str(batchsize)
                    )
                    print(name)
                    model, optimizer, history = wl.load_own_Model(
                        name,
                        device,
                        input_count=inputs,
                        output_count=outputs,
                        dropout=dropout,
                        hiddensize=i,
                        sequences=sequences,
                        batchsize=batchsize,
                        learning_rate=learning_rate,
                        layer=layers,
                    )

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
