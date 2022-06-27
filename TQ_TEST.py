#%%
from tqsdk import TqApi, TqAuth, TargetPosTask, TqSim,TqAccount,TqReplay,TqBacktest
from tqsdk import TqReplay
from datetime import date
import pandas as pd
from datetime import datetime
import time
import joblib
import numpy as np
from ta.volatility import BollingerBands, KeltnerChannel, DonchianChannel
from ta.trend import MACD, macd_diff, macd_signal, SMAIndicator
from ta.momentum import stochrsi, stochrsi_k, stochrsi_d
from tqsdk.tafunc import time_to_datetime
import lightgbm as lgb
symbol1="SHFE.ru2205"
slip=5   #滑点
lots=1   #交易手数
#全局变量区
global mypos
mypos=0
time2 = datetime.strptime("2021-01-01 21:00:00.00", "%Y-%m-%d %H:%M:%S.%f").time()
time3 = datetime.strptime("2021-01-01 20:00:00.00", "%Y-%m-%d %H:%M:%S.%f").time()
time4 = datetime.strptime("2021-01-01 09:00:00.00", "%Y-%m-%d %H:%M:%S.%f").time()
time5 = datetime.strptime("2021-01-01 15:00:00.00", "%Y-%m-%d %H:%M:%S.%f").time()
time6 = datetime.strptime("2021-01-01 23:00:00.00", "%Y-%m-%d %H:%M:%S.%f").time()
time7 = datetime.strptime("2021-01-01 00:58:00.00", "%Y-%m-%d %H:%M:%S.%f").time()
sim = TqSim()
sim.set_commission(symbol1, 3)
api = TqApi(account=sim,backtest=TqBacktest(start_dt=date(2022, 3, 14), end_dt=date(2022, 3, 15)), auth=TqAuth("13575469533", "HZLJDcl123456"),web_gui=True)
# api=TqApi(auth=TqAuth("13575469533", "HZLJDcl123456"))
position = sim.get_position(symbol1)
quote1 = api.get_quote(symbol1)  # 行情数据,流动性差的品种
tick1 = api.get_tick_serial(symbol1,1200)
min1 = api.get_kline_serial(symbol1,duration_seconds=120)
target_pos1 = TargetPosTask(api, symbol1)

gbm = lgb.Booster(model_file='lightgbm.txt')
data = pd.read_csv('2min_feature_columns_ru_nonminfeature.csv')
data = data.iloc[:,2:27]
col = data.columns

def calc_wap1(tick_data):
    wap1 = (tick_data['bid_price1'] * tick_data['ask_volume1'] + tick_data['ask_price1'] * tick_data[
        'bid_volume1']) / (tick_data['bid_volume1'] + tick_data['ask_volume1'])
    return wap1

def calc_wap3(tick_data):
    wap3 = (tick_data['bid_price1'] * tick_data['bid_volume1'] + tick_data['ask_price1'] * tick_data[
        'ask_volume1']) / (
                   tick_data['bid_volume1'] + tick_data['ask_volume1'])
    return wap3

def calc_wap5(tick_data):
    rolling = 120
    tick_data['size'] = tick_data['volume'] - tick_data['volume'].shift(1)
    tick_data['pv'] = tick_data['last_price'] * tick_data['size']
    last_price_wap = tick_data['pv'].rolling(rolling).sum() / tick_data['size'].rolling(rolling).sum()
    return last_price_wap

def tick_factor_preprocesser(tick_data):
    df = pd.DataFrame()
    rolling = 120
    tick_data['wap1'] = calc_wap1(tick_data)
    tick_data['wap3'] = calc_wap3(tick_data)
    tick_data['wap5'] = calc_wap5(tick_data)
    df['wap1_shift2'] = tick_data['wap1'].shift(1) - tick_data['wap1'].shift(2)
    df['wap1_shift4'] = tick_data['wap1'].shift(1) - tick_data['wap1'].shift(4)
    df['wap1_shift10'] = tick_data['wap1'].shift(1) - tick_data['wap1'].shift(10)
    df['wap1_shift25'] = tick_data['wap1'].shift(1) - tick_data['wap1'].shift(25)
    df['wap3_shift2'] = tick_data['wap3'].shift(1) - tick_data['wap3'].shift(2)
    df['wap3_shift4'] = tick_data['wap3'].shift(1) - tick_data['wap3'].shift(4)
    df['wap3_shift10'] = tick_data['wap3'].shift(1) - tick_data['wap3'].shift(10)
    df['wap3_shift25'] = tick_data['wap3'].shift(1) - tick_data['wap3'].shift(25)
    df['wap5_shift2'] = tick_data['wap5'].shift(1) - tick_data['wap5'].shift(2)
    df['wap5_shift3'] = tick_data['wap5'].shift(1) - tick_data['wap5'].shift(4)
    df['wap5_shift4'] = tick_data['wap5'].shift(1) - tick_data['wap5'].shift(10)
    df['wap5_shift5'] = tick_data['wap5'].shift(1) - tick_data['wap5'].shift(25)
    df['last_price_shift2'] = tick_data['last_price'].shift(1) - tick_data['last_price'].shift(2)
    df['last_price_shift10'] = tick_data['last_price'].shift(1) - tick_data['last_price'].shift(10)
    df['last_price_shift30'] = tick_data['last_price'].shift(1) - tick_data['last_price'].shift(30)
    df['last_price_shift100'] = tick_data['last_price'].shift(1) - tick_data['last_price'].shift(100)
    df['pre_vtB'] = np.where(tick_data.bid_price1 == tick_data.bid_price1.shift(1),
                             tick_data.bid_volume1 - tick_data.bid_volume1.shift(1), 0)
    df['vtB'] = np.where(tick_data.bid_price1 > tick_data.bid_price1.shift(1), tick_data.bid_volume1, df.pre_vtB)
    df['log_return1'] = np.log(tick_data['wap1'].shift(1) / tick_data['wap1'].shift(2)) * 100
    df['log_return3'] = np.log(tick_data['wap3'].shift(1) / tick_data['wap3'].shift(2)) * 100
    df['log_return_wap1_shift5'] = np.log(tick_data['wap1'].shift(1) / tick_data['wap1'].shift(5)) * 100
    df['log_return_wap1_shift15'] = np.log(tick_data['wap1'].shift(1) / tick_data['wap1'].shift(15)) * 100
    df['log_return_last_price_shift5'] = np.log(
        tick_data['last_price'].shift(1) / tick_data['last_price'].shift(5)) * 100
    df['log_return_last_price_shift15'] = np.log(
        tick_data['last_price'].shift(1) / tick_data['last_price'].shift(15)) * 100
    # df['rolling_skew5'] = tick_data['wap5'].rolling(rolling).skew()
    df['pre_vtA'] = np.where(tick_data.ask_price1 == tick_data.ask_price1.shift(1),
                             tick_data.ask_volume1 - tick_data.ask_volume1.shift(1), 0)
    df['datetime'] = tick_data['datetime'].apply(lambda x: datetime.fromtimestamp(x // 1000000000))
    df = df.fillna(method='bfill')
    df = df.replace(np.inf, 1)
    df = df.replace(-np.inf, -1)
    # df = df.set_index('datetime').groupby(pd.Grouper(freq='2min')).agg(np.mean)
    # df = df.iloc[-1:,:]
    return df

def min_factor_preprocesser(min_data):
    # print(min_data)
    data = pd.DataFrame()
    smafast = SMAIndicator(close=min_data['close'], window=16)
    data['smafast'] = smafast.sma_indicator()
    smaslow = SMAIndicator(close=min_data['close'], window=32)
    data['smaslow'] = smaslow.sma_indicator()
    bollingband = BollingerBands(close=min_data['close'])
    data['bollingermband'] = bollingband.bollinger_mavg()
    keltnerchannel = KeltnerChannel(high=min_data['high'], low=min_data['low'], close=min_data['close'])
    data['keltnerhband'] = keltnerchannel.keltner_channel_hband()
    data['keltnerlband'] = keltnerchannel.keltner_channel_lband()
    donchichannel = DonchianChannel(high=min_data['high'], low=min_data['low'], close=min_data['close'])
    data['donchimband'] = donchichannel.donchian_channel_mband()
    data['donchilband'] = donchichannel.donchian_channel_lband()
    data['high'] = min_data['high']
    data['low'] = min_data['low']
    data['close'] = min_data['close']
    data = data.fillna(method='bfill')
    data['datetime'] = min_data['datetime'].apply(lambda x: datetime.fromtimestamp(x // 1000000000))
    data = data.set_index('datetime')
    # data = data.iloc[-1:,:]
    return data

def main_pos():
    #生成交易信号
    # factor_list = tick_factor_preprocesser(tick1)
    # factor_list = factor_list.reindex(columns=col)
    # final_list = factor_list.iloc[-1:, :]
    predict = gbm.predict(final_list)
    # print(predict)
    y_pred = [1 if y > 0.5 else 0 for y in predict]
    # print(y_pred)
    if y_pred[0] == 1: #多头信号
        pos = 1
    else:           #空头信号
        pos = -1
    return pos

def strategy(pos):

    close = min1.close.iloc[-1]
    if  pos == 1:  #做多
        if position.pos_long == 0 and position.pos_short==0:
            order = api.insert_order(symbol=symbol1, direction="BUY", offset="OPEN", volume=lots,
                                     limit_price=close + slip, advanced="FAK")
        elif position.pos_short!=0 and position.pos_long == 0:
            order1 = api.insert_order(symbol=symbol1, direction="BUY", offset="CLOSETODAY", volume=lots,
                                      limit_price=close + slip, advanced="FAK")
            order2= api.insert_order(symbol=symbol1, direction="BUY", offset="OPEN", volume=lots,
                                     limit_price=close + slip, advanced="FAK")
    if pos == -1:
        if position.pos_short==0 and position.pos_long==0:
            order = api.insert_order(symbol=symbol1, direction="SELL", offset="OPEN", volume=lots,
                                     limit_price=close - slip, advanced="FAK")
        elif position.pos_long!=0 and position.pos_short==0:
            order1 = api.insert_order(symbol=symbol1, direction="SELL", offset="CLOSETODAY", volume=lots,
                                      limit_price=close - slip, advanced="FAK")
            order2 = api.insert_order(symbol=symbol1, direction="SELL", offset="OPEN", volume=lots,
                                      limit_price=close - slip, advanced="FAK")
def day_close():
    target_pos1.set_target_volume(0)
while True:
    api.wait_update() #信息更新
    time1 = datetime.strptime(quote1.datetime, "%Y-%m-%d %H:%M:%S.%f").time()
    # if time1 <= time2 and time1 > time3:
    #     #每天开盘需要重置的变量
    #     mypos=0
    if (time1 >= time4 and time1 < time5):  #交易时间段设置
        tick1['time'] = tick1['datetime'].apply(lambda x: datetime.fromtimestamp(x // 1000000000))
        tick1['min'] = tick1['time'].dt.minute
        if tick1['min'].iloc[-1] % 2 != 0 and tick1['min'].iloc[-1] != tick1['min'].iloc[-2]:
            factor_list = tick_factor_preprocesser(tick1.iloc[:-2,:])
            factor_list = factor_list.set_index('datetime').groupby(pd.Grouper(freq='2min')).agg(np.mean)
            factor_list = factor_list.reindex(columns=col)
            factor_list = factor_list.iloc[-2]
            final_list = np.array(factor_list).reshape(1,-1)
            # print(final_list)
            # print(tick1['time'].iloc[-1])
            # print(min1.open.iloc[-1])
            pos = main_pos()
            strategy(pos)
    # if ((time1 >= time6 and time1 < time4) & (time1 >= time5 and time1 < time2)):
    #     #收盘平仓
    #     day_close()
api.close()