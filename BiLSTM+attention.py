#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import array
import tensorflow as tf
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

import math
import sklearn.metrics as skm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Bidirectional,Attention,Input
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

#%%
data = pd.read_csv('test2005_2205_2min_data_ru_last_price_new_1.csv')
#%%
data = data.drop(['volume_size_x'], axis=1)
#%%
data['volume_size_y'] = data.iloc[:,-1:]
data = data.drop(data.iloc[:,-2:-1], axis=1)
data = data.drop(['volume_size'], axis=1)
#%%
from ta.volume import ForceIndexIndicator, EaseOfMovementIndicator
from ta.volatility import BollingerBands, KeltnerChannel, DonchianChannel
from ta.trend import MACD, macd_diff, macd_signal, SMAIndicator
from ta.momentum import stochrsi, stochrsi_k, stochrsi_d

forceindex = ForceIndexIndicator(close=data['last_price'], volume=data['volume_size_y'])
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
    for column in list(data.columns[0:66]):

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

filter_value = 0.1
filter_value2 = -0.1
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

# 将数据归一化，范围是0到1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))  # 创建归一化模板
train_set_scaled = sc.fit_transform(train_set)# 数据归一
test_set_scaled = sc.transform(test_set)
train_target = np.array(train_target)
test_target = np.array(test_target)


from keras.utils.np_utils import to_categorical
train_target = to_categorical(train_target, num_classes=2)
test_target = to_categorical(test_target, num_classes=2)

# from numpy import array
# 取前 n_timestamp 天的数据为 X；n_timestamp+1天数据为 Y。
def data_split(sequence,target ,n_timestamp):
    X = []
    y = []
    X_target = []
    y_target = []
    for i in range(len(sequence)):
        end_ix = i + n_timestamp

        if end_ix > len(sequence) - 1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        seq_target_x, seq_target_y = target[i:end_ix], target[end_ix]
        X.append(seq_x)
        y.append(seq_y)
        X_target.append(seq_target_x)
        y_target.append(seq_target_y)

    return array(X), array(y), array(X_target), array(y_target)

n_timestamp = 20

X_train, y_train, X_train_target, y_train_target = data_split(train_set_scaled,train_target, n_timestamp)
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
X_test, y_test, X_test_target, y_test_target = data_split(test_set_scaled,test_target, n_timestamp)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
#%%
cols = data.columns #所有列
train_col = [] # 选择测试集的列
for i in cols:
    if i != "target":
        train_col.append(i)

train = data[train_col]
target = data["target"] # 取前26列为训练数据，最后一列为target

# 将数据归一化，范围是0到1
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

sc = MinMaxScaler(feature_range=(0, 1))  # 创建归一化模板
train = sc.fit_transform(train)  # 数据归一
target = np.array(target)
target = to_categorical(target, num_classes=3)

train_X,test_X, train_y, test_y = train_test_split(train, target, test_size = 0.2, random_state = 3)
train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)
#%%
from keras.layers import Dropout
from keras import regularizers

import numpy
import keras
from keras.layers import Layer
from keras import backend as K
from keras import activations
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, Bidirectional
K.clear_session()

class AttentionLayer(Layer):
    def __init__(self, attention_size=None, **kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config
        
    def build(self, input_shape):
        assert len(input_shape) == 3
        
        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size
            
        self.W = self.add_weight(name='att_weight', shape=(hidden_size, self.attention_size),
                                initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                initializer='uniform', trainable=True)
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        self.V = K.reshape(self.V, (-1, 1))
        H = K.tanh(K.dot(inputs, self.W) + self.b)
        score = K.softmax(K.dot(H, self.V), axis=1)
        outputs = K.sum(score * inputs, axis=1)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]
        
        
def my_model():
    
    inputs = Input(shape=(245, 1))
    
    # BiLSTM层
    x = Bidirectional(LSTM(units=50,return_sequences=True),input_shape=(245, 1))(inputs)
    x = Dropout(0.5)(x)
    
    # Attention层
    x = AttentionLayer(attention_size=100)(x)
    
    # BiLSTM层
    x = Bidirectional(LSTM(units=50),input_shape=(49, 1))(inputs)
    x = Dropout(0.5)(x)
    
    # 输出层
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    model.summary() #输出模型结构和参数数量
    return model
    
model = my_model()
#%%
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
#%%
n_epochs = 100
history = model.fit(train_X, train_y,  # revise
                    batch_size=128,
                    epochs=n_epochs,
                    validation_data=(test_X, test_y),  # revise
                    validation_freq=1)
#%%
n_epochs = 100
history = model.fit(y_train, y_train_target,  # revise
                    batch_size=128,
                    epochs=n_epochs,
                    validation_data=(y_test, y_test_target),  # revise
                    validation_freq=1)  # 测试的epoch间隔数
#%%
predicted_contract = model.predict(y_test)
#%%
yhat = predicted_contract[:, 1]
#%%
y = y_test_target[:,1]
#%%
from sklearn.metrics import precision_recall_curve
from numpy import argmax
precision, recall, thresholds = precision_recall_curve(y, yhat)
fscore = (2 * precision * recall) / (precision + recall)
ix = argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
#%%
from sklearn.metrics import classification_report
y_1 = [1 if y > 0.385720 else 0 for y in yhat]
print(classification_report(y_1,y))
