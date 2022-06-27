#%%
from tqsdk import TqApi, TqAuth, TargetPosTask, TqSim,TqAccount,TqReplay,TqBacktest
from datetime import date
import pandas as pd
from datetime import datetime
import numpy as np
from tqsdk.tafunc import time_to_datetime

symbol1="SHFE.ru2205"
slip=1   #滑点
lots=1   #交易手数

#全局变量区
global mypos
mypos=0

time2 = datetime.strptime("2021-01-01 21:00:00.00", "%Y-%m-%d %H:%M:%S.%f").time()
time3 = datetime.strptime("2021-01-01 20:00:00.00", "%Y-%m-%d %H:%M:%S.%f").time()
time4 = datetime.strptime("2021-01-01 09:00:00.00", "%Y-%m-%d %H:%M:%S.%f").time()
time5 = datetime.strptime("2021-01-01 14:57:00.00", "%Y-%m-%d %H:%M:%S.%f").time()
time6 = datetime.strptime("2021-01-01 23:00:00.00", "%Y-%m-%d %H:%M:%S.%f").time()
time7 = datetime.strptime("2021-01-01 00:58:00.00", "%Y-%m-%d %H:%M:%S.%f").time()


sim = TqSim()
sim.set_commission(symbol1, 3)
api = TqApi(account=sim,backtest=TqBacktest(start_dt=date(2022, 3, 31), end_dt=date(2022, 4, 5)), auth=TqAuth("13575469533", "HZLJDcl123456"),web_gui=True)
# api=TqApi(auth=TqAuth("13575469533", "HZLJDcl123456"))
# position = sim.get_position(symbol1)
# quote1 = api.get_quote(symbol1)  # 行情数据,流动性差的品种
# tick1 = api.get_tick_serial(symbol1,1200)
min1 = api.get_kline_serial("SHFE.ru2205",duration_seconds=120)
# target_pos1 = TargetPosTask(api, symbol1)

#%%


def day_close():
    target_pos1.set_target_volume(0)


while True:
    api.wait_update() #信息更新
    # time1 = datetime.strptime(quote1.datetime, "%Y-%m-%d %H:%M:%S.%f").time()

    print(time_to_datetime(min1.iloc[-1]['datetime']))

    # if ((time1 >= time7 and time1 < time4) & (time1 >= time2 and time1 < time6)):
        # 收盘平仓
        # day_close()
api.close()
