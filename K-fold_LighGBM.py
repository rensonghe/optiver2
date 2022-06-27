#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.metrics import classification_report
from sklearn import metrics
from statsmodels.tsa.holtwinters import ExponentialSmoothing,Holt
#%%
data = pd.read_csv('test2005_2205_2min_data_ru_last_price_new.csv')
#%%
data = data.drop(['volume_size_x'], axis=1)
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
target = data["target"]
#%%
train = data[:70000]
test = data[70000:]
#%%
from sklearn.model_selection import TimeSeriesSplit
seed0 = 2202
# params0 = {
#     'objective':'rmse',
#     'boosting_type': 'gbdt',
#     'max_depth': 10,
#     'max_bin': 100,
#     'min_data_in_leaf': 500,
#     'learning_rate': 0.05,
#     'subsample': 0.72,
#     'subsample_freq': 4,
#     'feature_fraction': 0.5,
#     'lambda_l1': 0.5,
#     'lambda_l2': 1.0,
#     # 'categorical_column': [0],
#     'seed': seed0,
#     'feature_fraction_seed': seed0,
#     'bagging_seed': seed0,
#     'drop_seed': seed0,
#     'data_random_seed': seed0,
#     'n_jobs': -1,
#     'verbose': -1}

# self-defined objective function
def custom_smooth_l1_loss_train(y_true, y_pred):
    """Calculate gradient and hessien of loss function
    Args:
        y_true : array-like of shape = [n_samples]  The target values.
        y_pred : array-like of shape = [n_samples * n_classes] (for multi-class task)
    Returns:
        grad: gradient, should be list, numpy 1-D array or pandas Series
        hess: matrix hessien
    """
    import torch
    y_pred = y_pred.reshape(len(y_true), len(y_pred) // len(y_true))
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = torch.from_numpy(y_pred).float()
    y_pred.requires_grad = True

    y_true = torch.from_numpy(y_true).float()
    y_true.requires_grad = False

    loss = torch.nn.SmoothL1Loss()(y_true, y_pred)
    grad = torch.autograd.grad(loss, y_pred, create_graph=True, retain_graph=True)[0]
    hess = torch.autograd.grad(grad, y_pred,
                                 grad_outputs=torch.ones(y_pred.shape),
                                 create_graph=False)[0]
    return grad.detach().numpy(), hess.detach().numpy()


# self-defined eval metric function
def custom_smooth_l1_loss_valid(y_true, y_pred):
    """Calculate smooth_l1_loss
    Args:
        y_true : array-like of shape = [n_samples]  The target values.
        y_pred : array-like of shape = [n_samples * n_classes] (for multi-class task)
    Returns:
        loss: loss function value in evaluation
    """
    import torch
    y_pred = y_pred.reshape(len(y_true), len(y_pred) // len(y_true))
    y_pred = y_pred.max(axis=1)
    y_pred = torch.from_numpy(y_pred).float()
    y_pred.requires_grad = True

    y_true = torch.from_numpy(y_true).float()
    y_true.requires_grad = False

    loss = torch.nn.SmoothL1Loss()(y_pred, y_true).detach().numpy()
    return "custom_smooth_l1_loss_eval", np.mean(loss), False

def classification_report(y_true, y_pred):
    report = metrics.classification_report(y_pred, y_true)
    return report

def roc_auc_score(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred)
#%%
from sklearn import preprocessing
from sklearn import utils
lab_enc = preprocessing.LabelEncoder()
label = lab_enc.fit_transform(data['target'])
#%%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train,label, test_size=0.2)
#%%
oof_preds = np.zeros(X_train.shape[0])
test_preds = np.zeros(X_test.shape[0])
folds = TimeSeriesSplit(n_splits=10)
clf = LGBMClassifier()
clf.set_params(**{'objective': custom_smooth_l1_loss_train})  # 自定义损失函数
######## 模型训练 ##########
for fold_, (trn_, val_) in enumerate(folds.split(X_train, y_train)):
    trn_x, trn_y = X_train.iloc[trn_], y_train[trn_]
    val_x, val_y = X_train.iloc[val_], y_train[val_]

    clf.fit(trn_x, trn_y.astype('int'),
           eval_set=[(trn_x, trn_y), (val_x, val_y)],
           eval_metric=custom_smooth_l1_loss_valid,
           verbose=0)

    oof_preds[val_] = clf.predict(val_x, num_iteration=clf.best_iteration_)
    test_preds += clf.predict(X_test, num_iteration=clf.best_iteration_)/folds.n_splits
  # imp_df = pd.DataFrame()
  # imp_df['feature'] = self.feature_cols
  # imp_df['gain'] = clf.feature_importances_
  # imp_df['fold'] = fold_ + 1
  # importances = pd.concat([importances, imp_df], axis=0, sort=False)
  # clfs.append(clf)
    try:
        fold_accuracy = accuracy_score(val_y, oof_preds[val_])
     # fold_logloss = log_loss(val_y, clf.predict_proba(val_x))
     # print(LGBM Model no {}-fold accuracy_score is {}, LogLoss score is {},".format(fold_ + 1, fold_accuracy,
     #                                                                                      fold_logloss))
    except:
        print("accuracy_score or logloss calcul error")
#%%
def train_and_test_lgb(train, test, params0):

    kf = TimeSeriesSplit(n_splits=10)
    features = [col for col in train.columns if col not in {'target'}]
    # print(features)
    off_prediction = np.zeros(train.shape[0])
    test_prediction = np.zeros(test.shape[0])
    y = train['target']
    y_target = test['target']
    for fold, (train_index, test_index) in enumerate(kf.split(train)):
        print(f'Training fold {fold + 1}')
        x_train, x_test = train.iloc[train_index], train.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # print(y_train)
        # train_weight = 1 / np.square(y_train)
        # print(train_weight)
        # test_weight = 1 / np.square(y_test)
        train_dataset = lgb.Dataset(x_train[features], y_train)#, weight=train_weight)
        test_dataset = lgb.Dataset(x_test[features], y_test)#, weight=test_weight)

        model = lgb.train(params=params0,
                          num_boost_round=1000,
                          train_set=train_dataset,
                          valid_sets=[train_dataset,test_dataset],
                          verbose_eval= 250,
                          early_stopping_rounds=50,
                          # feval=custom_smooth_l1_loss_eval
                          )
        # lgb.fit(x_train, y_train.astype('int'), eval_set=[(x_train,y_train),(x_test,y_test)],
        #         eval_metric=custom_smooth_l1_loss_eval,verbose=0)

        off_prediction[test_index] = model.predict(x_test[features])
        test_prediction += model.predict(test[features]) / kf.n_splits
        print(test_prediction)
    # rmspe_score = rmspe(y, off_prediction)

    y_hat = [1 if y > 0.5 else 0 for y in test_prediction]
    # print(y_hat)
    report = classification_report(y_hat, y_target)
    # print(f'Our out of folds RMSPE is {rmspe_score}')
    print(report)
    # print('Full AUC score %.6f' % roc_auc_score(y, off_prediction))
    # lgb.plot_importance(model, max_num_features=20)
    # Return test predictions
    return test_prediction


prediction_lgb = train_and_test_lgb(train,test,params0)
