#%%
import sqlite3
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, GlobalAveragePooling1D
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

import tensorflow as tf

#early stop class loded from stackoverflow link: https://stackoverflow.com/questions/53500047/stop-training-in-keras-when-accuracy-is-already-1-0
class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, monitor='val_loss', baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get(self.monitor)
        if val_loss is not None:
            if val_loss <= self.baseline:
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True
# The following lines are to setup my GPU for the learning
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
#%%
data = pd.read_csv('test2202_01_04_01_31_ask_bid_price1.csv')
#%%
data = data.iloc[:,1:]
#%%
data = data[(data.datetime>='2022-01-04 09:00:00')& (data.datetime<='2022-01-08 08:59:59')]
#%%
data['target'] = np.log(data['ask_price1'] / data['bid_price1'].shift(-1)) * 100
data['target'] = data['target'].shift(-1)
data = data.dropna(axis=0, how='any')
#%%
data = data.drop(['datetime','ask_price1','bid_price1'],axis=1)
#%%
def classify(y):
    if y < 0:
        return -1
    elif y > 0:
        return 1
    else:
        return 0
data['target'] = data['target'].apply(lambda x:classify(x))
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

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size = 0.3, random_state = 1)
# train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
# test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)
#%%
#setting up the model of tensorflow
input_layer = Input(shape=(X_train.shape[1],1))
x = input_layer
for _ in range(5): # five layers
       x = Dropout(0.2)(x) # Dropout to avoid overfitting
       x = CuDNNLSTM(X_train.shape[1], return_sequences = True)(x) # using LSTM with return sequences to adopt to time sequences
x = GlobalAveragePooling1D()(x) # Global averaging to one layer shape to feed to a dense categorigal classification
output = Dense(target.shape[1], activation='softmax')(x)
model = Model(inputs=input_layer, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['acc'])
#
#creating an early stop based on minmizing val_loss
# early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1000,restore_best_weights=True)
early_stop = [TerminateOnBaseline(monitor='val_loss', baseline=0.05)]

#fit the model
r = model.fit(X_train, y_train, epochs = 100, batch_size=16400,
             validation_data = (X_test, y_test), callbacks=[early_stop], shuffle=False) #fit the data without shuffling
#plot the results.
pd.DataFrame(r.history).plot()

