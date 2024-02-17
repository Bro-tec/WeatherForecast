import weather_lstm_pytorch as wl
from datetime import datetime as dt
from datetime import timedelta as td


device = wl.check_cuda()

# flexible code for prediction using the console
inv = True
num1 = 0
num2 = 0
id = "00966"
city = ""
inp = input("Please choose if you wan to enter a City(c) name or an ID(i) to create your prediction plot. (to skip this enter \"Dummy\" or \"d\") \n")
inp = inp.lower()
if inp == "dummy" or inp =="d" or inp =="s" or inp =="skip":
    pass
elif inp == "city" or inp =="c":
    city = input("Please enter a City name from the excel list to create your prediction plot. \n")
    id = ""
elif inp == "id" or inp =="i":
    id = input("Please enter an ID from the excel list to create your prediction plot. \n")
    city = ""
else:
    print("Invalid input code will stop now")
    inv = False
    
if inv:
    m = input("Please choose with which mode your want to predict. You can choose between \"Normal\"(n) and \"Timestep\"(ts): ")
    m.lower()
    if m=="timestep" or m=="ts":
        num1 = ""
        while not num1.isdigit():
            num1 = input("Please choose the hourly model number: ")
        num1 = int(num1)
        num2 = ""
        while not num2.isdigit():
            num2 = input("Please choose the daily model number: ")
        num2 = int(num2)
    else:
        m = "normal"
        print("Normal was chosen")
    
    t = ""
    while not t.isdigit():
        t = input("Please choose the the hour of the day between 0 to 23: ")
        if t != "0" and t != "1" and t != "2" and t != "3" and t != "4" and t != "5" and t != "6" and t != "7" and t != "8" and t != "9" and t != "10" and t != "11" and t != "12" and t != "13" and t != "14" and t != "15" and t != "16" and t != "17" and t != "18" and t != "19" and t != "20" and t != "21" and t != "22" and t != "23":
            t = ""
    t = int(t)
    d = ""
    while not d.isdigit():
        d = input("How many days do you want to go back to predict your data? \n")
        if d == "0" or d == "1":
            d = ""
    d = int(d)

    wl.predictHourly(dt.now()-td(days=d), device, id=id, city=city, mode=m, model_num=num1, time=t)
    wl.predictHourly(dt.now()-td(days=d), device, id=id, city=city, mode=m, model_num=num1, time=t)

    wl.predictDaily(dt.now()-td(days=d), device, id=id, city=city, mode=m, model_num=num2)
    wl.predictDaily(dt.now()-td(days=d), device, id=id, city=city, mode=m, model_num=num2)