import weather_lstm_pytorch as wl
from datetime import datetime as dt
from datetime import timedelta as td


device = wl.check_cuda()


num = 2
id = "00966"
wl.predictHourly(dt.now()-td(days=4), device, id=id, mode="ts", model_num=num, time=14)
wl.predictHourly(dt.now()-td(days=1), device, id=id, mode="ts", model_num=num, time=-1)

wl.predictDaily(dt.now()-td(days=4), device, id=id, mode="ts", model_num=num, time=14)
wl.predictDaily(dt.now()-td(days=1), device, id=id, mode="ts", model_num=num, time=-1)