#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import array
import xgboost as xgb
import tensorflow as tf
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

import math
import sklearn.metrics as skm
import datetime
#%%
data = pd.read_csv('test2005_2205_2min_data_ru_last_price_new.csv')
#%%
data = data.drop(['volume_size_x'], axis=1)
#%%
from ta.volume import ForceIndexIndicator, EaseOfMovementIndicator
from ta.volatility import BollingerBands, KeltnerChannel, DonchianChannel
from ta.trend import MACD, macd_diff, macd_signal, SMAIndicator
from ta.momentum import stochrsi, stochrsi_k, stochrsi_d

forceindex = ForceIndexIndicator(close=data['last_price'], volume=data['volume'])
data['forceindex'] = forceindex.force_index()
easyofmove = EaseOfMovementIndicator(high=data['last_price_amax'], low=data['last_price_amin'], volume=data['volume_size_y'])
data['easyofmove'] = easyofmove.ease_of_movement()
bollingband = BollingerBands(close=data['last_price'])
data['bollingerhband'] = bollingband.bollinger_hband()
data['bollingerlband'] = bollingband.bollinger_lband()
data['bollingermband'] = bollingband.bollinger_mavg()
data['bollingerpband'] = bollingband.bollinger_pband()
data['bollingerwband'] = bollingband.bollinger_wband()
keltnerchannel = KeltnerChannel(high=data['last_price_amax'], low=data['last_price_amin'], close=data['last_price'])
data['keltnerhband'] = keltnerchannel.keltner_channel_hband()
data['keltnerlband'] = keltnerchannel.keltner_channel_lband()
data['keltnerwband'] = keltnerchannel.keltner_channel_wband()
data['keltnerpband'] = keltnerchannel.keltner_channel_pband()
donchichannel = DonchianChannel(high=data['last_price_amax'], low=data['last_price_amin'], close=data['last_price'])
data['donchimband'] = donchichannel.donchian_channel_mband()
data['donchilband'] = donchichannel.donchian_channel_lband()
data['donchipband'] = donchichannel.donchian_channel_pband()
data['donchiwband'] = donchichannel.donchian_channel_wband()
macd = MACD(close=data['last_price'])
data['macd'] = macd.macd()
data['macdsignal'] = macd_signal(close=data['last_price'])
data['macddiff'] = macd_diff(close=data['last_price'])
smafast = SMAIndicator(close=data['last_price'],window=16)
data['smafast'] = smafast.sma_indicator()
smaslow = SMAIndicator(close=data['last_price'],window=32)
data['smaslow'] = smaslow.sma_indicator()
data['stochrsi'] = stochrsi(close=data['last_price'],window=9, smooth1=26, smooth2=12)
data['stochrsi_k'] = stochrsi_k(close=data['last_price'],window=9, smooth1=26, smooth2=12)
data['stochrsi_d'] = stochrsi_d(close=data['last_price'],window=9, smooth1=26, smooth2=12)
data = data.fillna(method='bfill')
data = data.replace(np.inf, 1)
data = data.replace(-np.inf, -1)
#%%
data['target'] = np.log(data['last_price'] / data['last_price'].shift(1)) * 100
data['target'] = data['target'].shift(-1)
data = data.dropna(axis=0, how='any')
#%%
data = data.iloc[:,1:]
data = data.set_index('datetime')
#%%
def calcSpearman(data):

    ic_list = []
    data = data.copy()
    # target = data['target']
    for column in list(data.columns[0:53]):

        ic = data[column].rolling(200).corr(data['target'])
        ic_mean = np.mean(ic)
        print(ic_mean)
        ic_list.append(ic_mean)

        # print(ic_list)

    return ic_list

IC = calcSpearman(data)

IC = pd.DataFrame(IC)
columns = pd.DataFrame(data.columns)

IC_columns = pd.concat([IC, columns], axis=1)
col = ['value', 'variable']
IC_columns.columns = col

filter_value = 0.09
filter_value2 = -0.09
x_column = IC_columns.variable[IC_columns.value > filter_value]
y_column = IC_columns.variable[IC_columns.value < filter_value2]

x_column = x_column.tolist()
y_column = y_column.tolist()
final_col = x_column+y_column
#%%
data = data.reindex(columns=final_col)
#%%
def classify(y):

    if y < 0:
        return 0
    if y > 0:
        return 1
    else:
        return -1
data['target'] = data['target'].apply(lambda x:classify(x))
print(data['target'].value_counts())
#%%
data = data[~data['target'].isin([-1])]
#%%
cols = data.columns #所有列
train_col = [] # 选择测试集的列
for i in cols:
    if i != "target":
        train_col.append(i)

train = data[train_col]
target = data["target"] # 取前26列为训练数据，最后一列为target
#%%
train_set = train[:70000]
test_set = train[70000:]
train_target = target[:70000]
test_target = target[70000:]
#%%
train_target = np.array(train_target)
test_target = np.array(test_target)
X_train=np.array(train_set)
X_train_target=train_target
X_test=np.array(test_set)
X_test_target=test_target

#%%
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=2202,sampling_strategy=1.0)
X_train,X_train_target = sm.fit_resample(X_train,X_train_target)
#%% binary classification
model = xgb.XGBClassifier(learning_rate=0.7,
                          n_estimators=100,         # 树的个数--1000棵树建立xgboost
                          max_depth=10,               # 树的深度
                          min_child_weight = 10,      # 叶子节点最小权重
                          gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                          # objective='multi:softmax',
                          # num_class='3',
                          random_state=2202            # 随机数
                          )
#%%
print(X_train.shape)
print(X_train_target.shape)
model.fit(X_train,X_train_target)
y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print(accuracy_score(y_pred,X_test_target))
print("测试集表现：")
print(classification_report(y_pred,X_test_target))
#%%
test_data = data[200000:]
test_data = test_data.reset_index(drop=True)
test_data['datetime'] = test_data['datetime'].str.extract('\s(.{0,15})')
predict = pd.DataFrame(y_pred,columns=list('P'))
predict['datetime'] = test_data['datetime']
predict['wap1_mean'] = test_data['wap1_mean']
#%%
predict.to_csv('predict_-1_0.001.csv')
#%%
train_target_1 = model.predict(X_train)
print(accuracy_score(X_train_target,train_target_1))
print(classification_report(X_train_target,train_target_1))