#%%
# optimal threshold for precision-recall curve with logistic regression model
from numpy import argmax
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot
#%%
# generate dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# fit a model
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
# predict probabilities
yhat = model.predict_proba(testX)
# keep probabilities for the positive outcome only
yhat = yhat[:, 1]
# calculate roc curves
precision, recall, thresholds = precision_recall_curve(testy, yhat)
# convert to f score
fscore = (2 * precision * recall) / (precision + recall)
# locate the index of the largest f score
ix = argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
# plot the roc curve for the model
no_skill = len(testy[testy==1]) / len(testy)
pyplot.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
pyplot.plot(recall, precision, marker='.', label='Logistic')
pyplot.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.legend()
# show the plot
pyplot.show()
#%%
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import pandas as pd
min_max_scaler = MinMaxScaler()
iris = load_iris()
iris_scaler = min_max_scaler.fit_transform(iris.data)
iris_scaler = pd.DataFrame(iris_scaler)
#%%
iris_scaler['target'] = iris.target
#%%
X = np.matrix(iris_scaler)
VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
print(VIF_list)
#%%
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
plt.style.use('ggplot')
epochs=30

data = pd.read_csv('test2005_2205_2min_data_ru_last_price_new.csv')

data = data.drop(['volume_size_x'], axis=1)

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

data = data.iloc[:,1:]
data = data.set_index('datetime')

#%%
def classify(y):

    if y < 0:
        return 0.05
    if y > 0:
        return 0.95
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
target = data["target"]
#%%
from tensorflow.keras.utils import to_categorical
target_label = to_categorical(target,num_classes=2)
#%%
# target_label = target
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train,target_label, test_size=0.2,stratify=target_label)
#%%
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten,Reshape, BatchNormalization, Dense
from keras.optimizers import Adam

def build_cnn_1D(input_size, num_classes, lr):
    """1D卷积网络"""
    model = Sequential()
    model.add(Reshape((input_size, 1), input_shape=(input_size,)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='tanh', input_shape=(input_size, 1))) #输入维度设置
    model.add(BatchNormalization(name="batch_norm_1"))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(filters=32, kernel_size=3, activation='tanh'))
    model.add(BatchNormalization(name="batch_norm_2"))
    model.add(MaxPooling1D(2))

    model.add(Flatten())    # 展平
    model.add(Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l1(1e-4)))
    model.add(Dense(num_classes, activation='sigmoid')) # output size: (batch_size, num_classes)
    # opt = Adam(lr)

    from tensorflow.keras.losses import CategoricalCrossentropy
    loss = CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(loss=loss, optimizer='sgd', metrics=["accuracy"])
    return(model)
model = build_cnn_1D(X_train.shape[1], 2 , 0.001)
#%%
n_epochs = 50
history = model.fit(X_train, y_train,  # revise
                    batch_size=128,
                    epochs=n_epochs,
                    validation_data=(X_test, y_test),  # revise
                    validation_freq=1)
#%%
predict = model.predict(X_test)
#%%
yhat = predict[:, 1]
y_true = y_test[:,1]
# calculate roc curves
precision, recall, thresholds = precision_recall_curve(y_true, yhat)
# convert to f score
fscore = (2 * precision * recall) / (precision + recall)
# locate the index of the largest f score
ix = argmax(fscore)
print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))