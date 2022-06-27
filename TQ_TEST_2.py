#%%
from functools import reduce
from tqsdk import TqApi, TqAuth, TargetPosTask, TqSim,TqAccount,TqReplay,TqBacktest
from tqsdk import TqReplay
from datetime import date
import pandas as pd
from datetime import datetime
import time
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

symbol1="SHFE.ni2202"
slip=1   #滑点
lots=1   #交易手数

#全局变量区
global mypos
mypos=0

time2 = datetime.strptime("2021-01-01 21:00:00.00", "%Y-%m-%d %H:%M:%S.%f").time()
time3 = datetime.strptime("2021-01-01 20:00:00.00", "%Y-%m-%d %H:%M:%S.%f").time()
time4 = datetime.strptime("2021-01-01 09:00:00.00", "%Y-%m-%d %H:%M:%S.%f").time()
time5 = datetime.strptime("2021-01-01 14:57:00.00", "%Y-%m-%d %H:%M:%S.%f").time()
time6 = datetime.strptime("2021-01-01 23:59:00.00", "%Y-%m-%d %H:%M:%S.%f").time()
time7 = datetime.strptime("2021-01-01 00:58:00.00", "%Y-%m-%d %H:%M:%S.%f").time()


sim = TqSim()
sim.set_commission(symbol1, 3)
api = TqApi(account=sim,backtest=TqBacktest(start_dt=date(2022, 1, 10), end_dt=date(2022, 1, 12)), auth=TqAuth("13575469533", "HZLJDcl123456"),web_gui=True)
# api=TqApi(auth=TqAuth("13575469533", "HZLJDcl123456"))

quote1 = api.get_quote(symbol1)  # 行情数据,流动性差的品种
tick1 = api.get_tick_serial(symbol1,200)  #最大200个数据

target_pos1 = TargetPosTask(api, symbol1)

# model_1 = joblib.load('RandomForest_-1.pkl')
model_2 = joblib.load('RandomForest_-1.pkl')
#%%
def factor_generating(data):

    def reciprocal_transformation(series):
        return np.sqrt(np.abs(1 / series)) * 100000

    def square_root_translation(series):
        return series ** (1 / 2)

    # Function to calculate second WAP
    def calc_wap1(df):
        wap = (df['bid_price1'] * df['ask_volume1'] + df['ask_price1'] * df['bid_volume1']) / (
                    df['bid_volume1'] + df['ask_volume1'])
        return wap
    def calc_wap2(df):
        wap = (df['bid_price2'] * df['ask_volume2'] + df['ask_price2'] * df['bid_volume2']) / (
                    df['bid_volume2'] + df['ask_volume2'])
        return wap

    def calc_wap3(df):
        wap = (df['bid_price1'] * df['bid_volume1'] + df['ask_price1'] * df['ask_volume1']) / (
                    df['bid_volume1'] + df['ask_volume1'])
        return wap


    def factor_preprocesser(data):

        df = data

        df['volume_size'] = df['volume'] - df['volume'].shift(1)
        df['wap1'] = calc_wap1(df)
        df['wap3'] = calc_wap3(df)

        df['wap3_shift3'] = df['wap3'].shift(1) - df['wap3'].shift(3)
        df['wap3_shift4'] = df['wap3'].shift(1) - df['wap3'].shift(4)
        df['wap3_shift5'] = df['wap3'].shift(1) - df['wap3'].shift(5)

        df['ask_size1_2_minus'] = df['ask_volume1'] - df['ask_volume2']
        df['bid_ask_size1_minus'] = df['bid_volume1'] - df['ask_volume1']
        df['bid_ask_size1_plus'] = df['bid_volume1'] + df['ask_volume1']
        df['bid_ask_size2_minus'] = df['bid_volume2'] - df['ask_volume2']
        df['bid_ask_size2_plus'] = df['bid_volume2'] + df['ask_volume2']
        df['bid_size1_shift'] = df['bid_volume1'] - df['bid_volume1'].shift()
        df['bid_size2_shift'] = df['bid_volume2'] - df['bid_volume2'].shift()
        df['ask_size1_shift'] = df['ask_volume1'] - df['ask_volume1'].shift()
        df['ask_size2_shift'] = df['ask_volume2'] - df['ask_volume2'].shift()
        df['bid_ask_size1_spread'] = df['bid_ask_size1_minus'] / df['bid_ask_size1_plus']
        df['bid_ask_size2_spread'] = df['bid_ask_size2_minus'] / df['bid_ask_size2_plus']
        df['mid_price1'] = (df['ask_price1'] + df['bid_price1']) / 2
        df['mid_price2'] = (df['ask_price2'] + df['bid_price2']) / 2
        df['mid_price3'] = (df['ask_price3'] + df['bid_price3']) / 2

        df['bid_amount_abs1'] = abs(
            (df['amount'] - df['amount'].shift(1)) / df['volume_size'] - df['bid_price1']) - abs(
            (df['amount'] - df['amount'].shift(1)) / df['volume_size'] - df['ask_price1'])

        df['press_buy1'] = ((df['mid_price1'] / (df['mid_price1'] - df['bid_price1'])) / (
                    (df['mid_price1'] / (df['mid_price1'] - df['bid_price1'])) + (
                        df['mid_price2'] / (df['mid_price2'] - df['bid_price2'])) + (
                                df['mid_price3'] / (df['mid_price3'] - df['bid_price3'])))) * (
                                       df['bid_volume1'] + df['bid_volume2'] + df['bid_volume3'])
        df['press_sell1'] = ((df['mid_price1'] / (df['ask_price1'] - df['mid_price1'])) / (
                    (df['mid_price1'] / (df['ask_price1'] - df['mid_price1'])) + (
                        df['mid_price2'] / (df['ask_price2'] - df['mid_price2'])) + (
                                df['mid_price3'] / (df['ask_price3'] - df['mid_price3'])))) * (
                                        df['ask_volume1'] + df['ask_volume2'] + df['ask_volume3'])
        df['order_BS1'] = np.log(df['press_buy1']) - np.log(df['press_sell1'])

        df['pre_vtA'] = np.where(df.ask_price1 == df.ask_price1.shift(1), df.ask_volume1 - df.ask_volume1.shift(1), 0)
        df['vtA'] = np.where(df.ask_price1 > df.ask_price1.shift(1), df.ask_volume1, df.pre_vtA)
        df['pre_vtB'] = np.where(df.bid_price1 == df.bid_price1.shift(1), df.bid_volume1 - df.bid_volume1.shift(1), 0)
        df['vtB'] = np.where(df.bid_price1 > df.bid_price1.shift(1), df.bid_volume1, df.pre_vtB)
        df['Oiab'] = df['vtB'] - df['vtA']

        ask_quantity_1 = df['ask_volume1']
        ask_quantity_2 = df['ask_volume2']
        ask_quantity_3 = df['ask_volume3']
        bid_quantity_1 = df['bid_volume1']
        bid_quantity_2 = df['bid_volume2']
        bid_quantity_3 = df['bid_volume3']

        def weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3, bid_quantity_1, bid_quantity_2,
                             bid_quantity_3):
            Weight_Ask = (w1 * ask_quantity_1 + w2 * ask_quantity_2 + w3 * ask_quantity_3)
            Weight_Bid = (w1 * bid_quantity_1 + w2 * bid_quantity_2 + w3 * bid_quantity_3)
            W_AB = Weight_Ask / Weight_Bid
            W_A_B = (Weight_Ask - Weight_Bid) / (Weight_Ask + Weight_Bid)

            return W_AB, W_A_B

        w1, w2, w3 = [100.0, 0.0, 0.0]
        df['W_A_B_100'],df['W_AB_100'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3,
                                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
        w1, w2, w3 = [0.0, 100.0, 0.0]
        df['W_AB_010'], df['W_A_B_010'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3,
                                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
        w1, w2, w3 = [60.0, 40.0, 0.0]
        df['W_AB_640'], df['W_A_B_640'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3,
                                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
        w1, w2, w3 = [50.0, 50.0, 0.0]
        df['W_AB_550'], df['W_A_B_550'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3,
                                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
        w1, w2, w3 = [50.0, 30.0, 20.0]
        df['W_AB_532'], df['W_A_B_532'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3,
                                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
        w1, w2, w3 = [1.0, 1.0, 1.0]
        df['W_AB_111'], df['W_A_B_111'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3,
                                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
        w1, w2, w3 = [10.0, 90.0, 1.0]
        df['W_AB_190'], df['W_A_B_190'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3,
                                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
        w1, w2, w3 = [20.0, 80.0, 0.0]
        df['W_AB_280'], df['W_A_B_280'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3,
                                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
        w1, w2, w3 = [30.0, 70.0, 0.0]
        df['W_AB_370'], df['W_A_B_370'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3,
                                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
        w1, w2, w3 = [40.0, 60.0, 0.0]
        df['W_AB_460'], df['W_A_B_460'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3,
                                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
        w1, w2, w3 = [10.0, 20.0, 70.0]
        df['W_AB_127'], df['W_A_B_127'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3,
                                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)
        w1, w2, w3 = [20.0, 30.0, 50.0]
        df['W_AB_235'], df['W_A_B_235'] = weight_pecentage(w1, w2, w3, ask_quantity_1, ask_quantity_2, ask_quantity_3,
                                                           bid_quantity_1, bid_quantity_2, bid_quantity_3)

        create_feature_dict = {
            'wap1': [np.mean],
            'W_A_B_100': [np.mean, square_root_translation],
            'W_AB_010': [np.mean, reciprocal_transformation, square_root_translation],
            'W_A_B_010': [np.mean, square_root_translation],
            'W_AB_640': [np.mean, reciprocal_transformation, square_root_translation],
            'W_A_B_640': [np.mean],
            'W_AB_550': [np.mean, reciprocal_transformation, square_root_translation],
            'W_A_B_550': [np.mean, square_root_translation],
            'W_AB_532': [np.mean, reciprocal_transformation, square_root_translation],
            'W_A_B_532': [np.mean, square_root_translation],
            'W_AB_111': [np.mean, reciprocal_transformation, square_root_translation],
            'W_A_B_111': [np.mean, square_root_translation],
            'W_AB_190': [np.mean, reciprocal_transformation, square_root_translation],
            'W_A_B_190': [np.mean, square_root_translation],
            'W_AB_280': [np.mean, reciprocal_transformation, square_root_translation],
            'W_A_B_280': [np.mean, square_root_translation],
            'W_AB_370': [np.mean, reciprocal_transformation, square_root_translation],
            'W_A_B_370': [np.mean, square_root_translation],
            'W_AB_460': [np.mean, reciprocal_transformation, square_root_translation],
            'W_A_B_460': [np.mean, square_root_translation],
            'W_AB_127': [np.mean, reciprocal_transformation, square_root_translation],
            'W_A_B_127': [np.mean, square_root_translation],
            'W_AB_235': [np.mean, reciprocal_transformation, square_root_translation],
            'W_A_B_235': [np.mean, square_root_translation],
            'press_buy1': [reciprocal_transformation, square_root_translation],
            'press_sell1':[square_root_translation],
            'ask_size2_shift':[np.mean, square_root_translation],
            'bid_ask_size1_spread':[np.mean, square_root_translation],
            'wap3_shift3':[np.mean],
            'wap3_shift4':[np.mean, square_root_translation],
            'wap3_shift5':[np.mean],
            'order_BS1':[np.mean, square_root_translation],
            'vtB':[np.mean],
            'Oiab':[square_root_translation],
            'bid_ask_size2_minus': [np.mean, square_root_translation],
            'bid_size2_shift': [np.mean, square_root_translation],
            'bid_amount_abs1': [np.mean, square_root_translation],
        }

        def get_stats_window(fe_dict, index, add_suffix=False):
            df_feature = df.groupby(['datetime']).agg(fe_dict).reset_index()
            df_feature.columns = ['_'.join(col) for col in df_feature.columns]
            return df_feature

        df_feature = get_stats_window(create_feature_dict, index=0, add_suffix=False)

        return df_feature

    test_data = factor_preprocesser(data)
    factor = test_data.fillna(0)
    factor = factor.replace(np.inf, 1)
    factor = factor.replace(-np.inf, -1)

    factor = factor.drop(['datetime_'], axis=1)
    sc = MinMaxScaler(feature_range=(0, 1))
    data = sc.fit_transform(factor)
    tick = data[-1].reshape(1, -1)
    return tick


def main_pos_2():  #多头交易
    #生成交易信号
    tick_list = factor_generating(tick1)
    y_pred = model_2.predict(tick_list)
    print(y_pred)
    if y_pred == -1:
        pos = -1
    else:
        pos = 0

    return pos

def strategy(pos):
    global mypos

    tick = tick1.iloc[-1:,:]
    wap_price = int((tick['bid_price1'] * tick['ask_volume1'] + tick['ask_price1'] * tick['bid_volume1']) / (tick['bid_volume1'] + tick['ask_volume1']))

    if mypos == 0 and pos == -1:  #做空
        openorder = api.insert_order(symbol=symbol1, direction="SELL", offset="OPEN", volume=lots,limit_price=wap_price, advanced="FAK")
        mypos = -1
    if mypos < 0 and pos == 0:  #平空
        closeorder = api.insert_order(symbol=symbol1, direction="BUY", offset="CLOSETODAY", volume=lots,limit_price=wap_price, advanced="FAK")  # , advanced="FAK"
        mypos = 0


def day_close():
    target_pos1.set_target_volume(0)


while True:
    api.wait_update() #信息更新
    time1 = datetime.strptime(quote1.datetime, "%Y-%m-%d %H:%M:%S.%f").time()
    # if time1 <= time2 and time1 > time3:
    #     #每天开盘需要重置的变量
    #     mypos=0

    if (time1 >= time4 and time1 < time5):    #交易时间段设置
        pos = main_pos_2()
        # print(pos)
        strategy(pos)

    if (time1 >= time7 and time1 < time4):
        #收盘平仓
        day_close()
api.close()