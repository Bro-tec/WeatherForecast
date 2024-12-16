import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import json
import pandas as pd


# https://www.geeksforgeeks.org/python-check-if-list-is-sorted-or-not/
def is_sorted(x):
    for i in range(len(x) - 1):
        if x[i] < x[i + 1]:
            return False
    return True


def plotting_hist(
    start=0,
    duration=30,
    pointer=0,
    sort_by="loss",
    show_plot=True,
    fn="test_",
    ln="_l",
    lr="_lr",
    unsorted=False,
    reverse=False,
):
    if duration > 0:
        end = start + duration
    else:
        end = -1
    print(
        "start=",
        start,
        ", duration=",
        duration,
        ", pointer=",
        pointer,
        ", sort_by=",
        sort_by,
    )
    mypath = "./Models/"
    files = [
        f
        for f in os.listdir(mypath)
        if "history.json" in f and fn in f
        # if "test_" in f
        # if "t_" in f
        # if "model_" in f
        and ln in f
        # if "80" in f
        and lr in f and os.path.isfile(os.path.join(mypath, f))
    ]
    file_list = files.copy()
    hist_accuracy = []
    hist_val_accuracy = []
    hist_loss = []
    hist_val_loss = []
    fig, ax = plt.subplots(4, figsize=(20, 20), sharey="row")
    plt.xlabel("epoch")
    for f in files:
        if os.path.exists(f"./Models/{f}"):
            with open(f"./Models/{f}", "r") as jsonf:
                history = json.load(jsonf)

                history["loss"] = (pd.Series(history["loss"])).tolist()
                history["val_loss"] = (pd.Series(history["val_loss"])).tolist()
                history["loss"] = [-_ if _ < 0 else _ for _ in history["loss"]]
                history["val_loss"] = [-_ if _ < 0 else _ for _ in history["val_loss"]]

                # print(history["loss"].index(min(history["loss"][1:])))
                if len(history["loss"]) > pointer and (
                    is_sorted(history["loss"][start : pointer + 1]) or unsorted
                ):
                    if pointer != 0:
                        hist_accuracy.append(history["accuracy"][pointer])
                        hist_val_accuracy.append(history["val_accuracy"][pointer])
                        hist_loss.append(history["loss"][pointer])
                        hist_val_loss.append(history["val_loss"][pointer])
                    # summarize history for accuracy
                    ax[0].plot(history["accuracy"][start:end], alpha=0.75)
                    ax[1].plot(history["val_accuracy"][start:end], alpha=0.75)
                    ax[2].plot(history["loss"][start:end], alpha=0.75)
                    ax[3].plot(history["val_loss"][start:end], alpha=0.75)
                else:
                    file_list.remove(f)
                    print(f)
        else:
            print("not found: ", f)
    if pointer != 0:
        hist = pd.DataFrame(
            {
                "Name": file_list,
                "accuracy": hist_accuracy,
                "val_accuracy": hist_val_accuracy,
                "loss": hist_loss,
                "val_loss": hist_val_loss,
            }
        )
        hist.sort_values(sort_by, inplace=True)

        if reverse:
            hist = hist[::-1]
        print(hist.shape)
        print(hist[:20])
    ax[0].grid(axis="y")
    ax[1].grid(axis="y")
    ax[2].grid(axis="y")
    ax[3].grid(axis="y")
    if show_plot:
        plt.show()


if __name__ == "__main__":
    # plotting_hist(start=1, duration=3, pointer=1, show_plot=False)
    # plotting_hist(start=1, duration=6, pointer=3, show_plot=False)
    # plotting_hist(start=1, duration=9, pointer=6, show_plot=False)
    # plotting_hist(start=1, duration=15, pointer=9, show_plot=False)
    # plotting_hist(start=1, duration=21, pointer=15, show_plot=True)

    # plotting_hist(start=1, duration=-1, pointer=0, show_plot=False, ln="_l1")
    # plotting_hist(start=1, duration=-1, pointer=0, show_plot=False, ln="_l2")
    # plotting_hist(start=1, duration=-1, pointer=0, show_plot=True, ln="_l3")
    # plotting_hist(
    #     start=6, duration=-1, pointer=0, unsorted=True, show_plot=False, fn="t_"
    # )
    # plotting_hist(
    #     start=6,
    #     duration=-1,
    #     pointer=0,
    #     unsorted=True,
    #     show_plot=False,
    #     fn="t_",
    #     ln="b1000_",
    # )
    # plotting_hist(
    #     start=6,
    #     duration=-1,
    #     pointer=0,
    #     unsorted=True,
    #     show_plot=False,
    #     fn="t_",
    #     ln="b10000_",
    # )
    # plotting_hist(
    #     start=6,
    #     duration=-1,
    #     pointer=0,
    #     unsorted=True,
    #     show_plot=True,
    #     fn="t_",
    #     ln="b100_",
    # )

    # plotting_hist(
    #     start=6,
    #     duration=-1,
    #     pointer=-11,
    #     sort_by="accuracy",
    #     unsorted=True,
    #     show_plot=True,
    #     reverse=True,
    # )
    plotting_hist(start=6, duration=-1, pointer=-1, unsorted=True, show_plot=False)
    plotting_hist(
        start=6, duration=-1, pointer=-1, unsorted=True, show_plot=False, lr="_lr0.01"
    )
    plotting_hist(
        start=6, duration=-1, pointer=-1, unsorted=True, show_plot=True, lr="_lr0.00"
    )
    # plotting_hist(
    #     start=6, duration=-1, pointer=0, unsorted=True, show_plot=False, lr="_lr0.05"
    # )
    # plotting_hist(
    #     start=6, duration=-1, pointer=0, unsorted=True, show_plot=True, lr="_lr0.1"
    # )
