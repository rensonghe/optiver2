#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import array
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

import joblib
import math
import sklearn.metrics as skm
import datetime
#%%
data = pd.read_csv('test2202_01_04_01_31.csv')
data = data.iloc[:,1:]

data['datetime'] = pd.to_datetime(data['datetime'])
data['hour'] = data['datetime'].dt.hour
data['min'] = data['datetime'].dt.minute
data['sec'] = data['datetime'].dt.second
# data['day'] = data['datetime'].dt.day
# data['month'] = data['datetime'].dt.month

from sklearn.preprocessing import FunctionTransformer


def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

data['sin_sec'] = sin_transformer(60).fit_transform(data['sec'])
data['cos_sec'] = cos_transformer(60).fit_transform(data['sec'])
data['sin_min'] = sin_transformer(60).fit_transform(data['min'])
data['cos_min'] = cos_transformer(60).fit_transform(data['min'])
data['sin_hour'] = sin_transformer(24).fit_transform(data['hour'])
data['cos_hour'] = cos_transformer(24).fit_transform(data['hour'])
#%%
data = data[(data.datetime>='2022-01-11 09:00:00')& (data.datetime<='2022-01-13 08:59:59')]
data = data.reset_index(drop=True)

data['target'] = np.log(data['wap1_mean'] / data['wap1_mean'].shift(1)) * 100
# data['log_return'][np.isinf(data['log_return'])] = 0
data['target'] = data['target'].shift(-1)
data = data.dropna(axis=0, how='any')
#%%
data1 = data.drop(['datetime'],axis=1)
#%%
def classify(y):
    if y > 0.001:
        return 1
    # if y < -0.001:
    #     return -1
    else:
        return 0
data1['target'] = data1['target'].apply(lambda x:classify(x))

cols = data1.columns #所有列
train_col = [] # 选择测试集的列
for i in cols:
    if i != "target":
        train_col.append(i)

train = data1[train_col]
target = data1["target"] # 取前26列为训练数据，最后一列为target
#%%
train_set = train[:52647]
test_set = train[52647:]
train_target = target[:52647]
test_target = target[52647:]
#%%
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))  # 创建归一化模板
train_set_scaled = sc.fit_transform(train_set)# 数据归一
test_set_scaled = sc.transform(test_set)
train_target = np.array(train_target)
test_target = np.array(test_target)

#- from numpy import array
X_train=train_set_scaled
X_train_target=train_target
X_test=test_set_scaled
X_test_target=test_target
#%%
model = joblib.load('RandomForest_-1.pkl')
#%%
model_2 = joblib.load('RandomForest_1.pkl')
#%%
model.fit(X_train,X_train_target)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
print(accuracy_score(y_pred,X_test_target))
print("测试集表现：")
print(classification_report(y_pred,X_test_target))
print("评价指标-混淆矩阵：")
print(metrics.confusion_matrix(y_pred,X_test_target))
#%%
model_2.fit(X_train,X_train_target)
y_pred = model_2.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
print(accuracy_score(y_pred,X_test_target))
print("测试集表现：")
print(classification_report(y_pred,X_test_target))
print("评价指标-混淆矩阵：")
print(metrics.confusion_matrix(y_pred,X_test_target))
#%%
test_data = data[52647:]
test_data = test_data.reset_index(drop=True)
# test_data['datetime'] = test_data['datetime'].str.extract('\s(.{0,15})')
predict = pd.DataFrame(y_pred,columns=list('P'))
predict['datetime'] = test_data['datetime']
predict['wap1_mean'] = test_data['wap1_mean']
#%%
predict.to_csv('predict_1_0.001_RF_2022_01_12.csv')
#%%
joblib.dump(model,'RandomForest_-1.pkl')
#%%
joblib.dump(model_2,'RandomForest_1.pkl')