import pandas as pd
import numpy as np
import torch.optim
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import KFold
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
train_set = train[:75000]
test_set = train[75000:]
train_target = target[:75000]
test_target = target[75000:]
#%%
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))  # 创建归一化模板
train_set_scaled = sc.fit_transform(train_set)# 数据归一
test_set_scaled = sc.transform(test_set)
train_target = np.array(train_target)
test_target = np.array(test_target)

X_train = train_set_scaled
X_train_target=train_target
X_test = test_set_scaled
X_test_target =test_target
#%%
from imblearn.over_sampling import SMOTE

sm = SMOTE()
X_train,X_train_target = sm.fit_resample(X_train,X_train_target)
#%%
train = data.iloc[:70000,:]
test = data.iloc[70000:,:]
#%%
from sklearn import metrics
from sklearn.model_selection import TimeSeriesSplit
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

def classification_report(y_true, y_pred):
    report = metrics.classification_report(y_pred, y_true)
    return report


# tabnet_params = dict(
#     cat_idxs=cat_idxs,
#     cat_dims=cat_dims,
#     cat_emb_dim=1,
#     n_d=16,
#     n_a=16,
#     n_steps=2,
#     gamma=2,
#     n_independent=2,
#     n_shared=2,
#     lambda_sparse=0,
#     optimizer_fn=Adam,
#     optimizer_params=dict(lr=(2e-2)),
#     mask_type="entmax",
#     scheduler_params=dict(T_0=200, T_mult=1, eta_min=1e-4, last_epoch=-1, verbose=False),
#     scheduler_fn=CosineAnnealingWarmRestarts,
#     seed=42,
#     verbose=10
#
# )

def train_and_test(train, test):

    kf = TimeSeriesSplit(n_splits=10)
    features = [col for col in train.columns if col not in {'target'}]
    # print(features)
    off_prediction = np.zeros(train.shape[0])
    test_prediction = np.zeros(test.shape[0])
    y = train['target']
    for fold, (train_index, test_index) in enumerate(kf.split(train)):
        print(f'Training fold {fold + 1}')
        x_train, x_test = train[features].iloc[train_index], train[features].iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(x_train)

        tb_cls = TabNetClassifier(optimizer_fn=torch.optim.Adam,
                                  optimizer_params=dict(lr=1e-3),
                                  scheduler_params=dict(T_0=200, T_mult=1, eta_min=1e-4, last_epoch=-1, verbose=False),
                                  scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                                  mask_type='entmax'  # "sparsemax"
                                  )
        tb_cls.fit(x_train, y_train,
                   eval_set=[(x_train, y_train), (x_test, y_test)],
                   eval_name=['train', 'test'],
                   eval_metric=['accuracy'],
                   max_epochs=1000, patience=100,
                   batch_size=128, virtual_batch_size=64 ,drop_last=False)

        off_prediction[test_index] = tb_cls.predict(x_test[features])
        test_prediction += tb_cls.predict(test[features]) / kf.n_splits
    # rmspe_score = rmspe(y, off_prediction)

    off_prediction = [1 if y > 0.5 else 0 for y in off_prediction]
    report = classification_report(y, off_prediction)
    # print(f'Our out of folds RMSPE is {rmspe_score}')
    print(report)
    # print('Full AUC score %.6f' % roc_auc_score(y, off_prediction))
    # lgb.plot_importance(model, max_num_features=20)
    # Return test predictions
    return test_prediction

prediction_tabnet = train_and_test(train,test)
#%%
