#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm.sklearn import LGBMClassifier
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.metrics import classification_report
from sklearn import metrics
from statsmodels.tsa.holtwinters import ExponentialSmoothing,Holt
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%
data = pd.read_csv('test1912_2206_2min_data_ag_last_price.csv')
#%%
data = data.drop(['volume_size_x'], axis=1)
#%%
import ta

data = ta.add_all_ta_features(data, open='highest',close='last_price', high='last_price_amax', low='last_price_amin',volume='volume_size_y', fillna=True)
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
def check_VIF(df):

    from sklearn.preprocessing import MinMaxScaler
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools import add_constant
    df['c'] = 1
    name = df.columns
    # scaler = MinMaxScaler()
    # df = scaler.fit_transform(df)
    # df = pd.DataFrame(df)
    # X = add_constant(df)
    X = np.matrix(df)
    VIF_list = [variance_inflation_factor(X, i) for i in tqdm(range(X.shape[1]))]
    VIF = pd.DataFrame({'feature':name, 'VIF':VIF_list})
    # VIF = VIF.drop(['c'], axis=0)
    max_VIF = max(VIF)
    print(max_VIF)
    return VIF

VIF_list = check_VIF(data)

name = data.columns
VIF = pd.DataFrame({'feature': name, 'VIF': VIF_list})
#%%
VIF_list = pd.read_csv('LightGBM_2min_ru_VIF_list_nontarget.csv')
VIF_list = VIF_list.iloc[:,1:]

col_save = VIF_list.feature[VIF_list.VIF < 0.9]
col_save = col_save.tolist()

data = data.reindex(columns=col_save)
data = data.drop(['c'], axis=1)
#%%
def calcSpearman(data):

    ic_list = []
    data = data.copy()
    # target = data['target']
    for column in list(data.columns[0:257]):

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
train_set = train[:72028]
test_set = train[72028:72707]
train_target = target[:72028]
test_target = target[72028:72707]
#%%
train_set = train[:72708]
test_set = train[72708:73427]
train_target = target[:72708]
test_target = target[72708:73427]
#%%
train_set = train[:73428]
test_set = train[73428:74135]
train_target = target[:73428]
test_target = target[73428:74135]
#%%
train_set = train[:144718]
test_set = train[144718:146452]
train_target = target[:144718]
test_target = target[144718:146452]
#%%
train_set = train[:139183]
test_set = train[139183:]
train_target = target[:139183]
test_target = target[139183:]
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
# train_target = np.array(train_target)
# test_target = np.array(test_target)
# X_train=np.array(train_set)
# X_train_target=train_target
# X_test=np.array(test_set)
# X_test_target=test_target
#%%
from imblearn.over_sampling import SMOTE

sm = SMOTE()
X_train, X_train_target = sm.fit_resample(X_train, X_train_target)
#%% GridSearchCV
# from sklearn.model_selection import GridSearchCV
# ## 定义参数取值范围
# learning_rate = [0.1, 0.3, 0.5, 0.7]
# feature_fraction = [0.3, 0.5, 0.8, 1]
# num_leaves = [16, 32, 64, 128]
# max_depth = [-1, 1, 3, 7, 10]
#
# parameters = {'learning_rate': learning_rate,
#               'feature_fraction':feature_fraction,
#               'num_leaves': num_leaves,
#               'max_depth': max_depth}
#
# model = LGBMClassifier(n_estimators=100)
# ## 进行网格搜索
# clf = GridSearchCV(model, parameters, cv=3, scoring='accuracy',verbose=3)
# clf = clf.fit(X_train,X_train_target)
# clf.best_params_## 网格搜索后的最优参数

#%%
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from bayes_opt import BayesianOptimization

# kf = TimeSeriesSplit(n_splits=10)
def lgb_cv(colsample_bytree, learning_rate, min_child_samples, min_child_weight, n_estimators, num_leaves, subsample, max_depth, min_split_gain):
    model = LGBMClassifier(boosting_type='gbdt',objective='binary',
           colsample_bytree=float(colsample_bytree), learning_rate=float(learning_rate),
           min_child_samples=int(min_child_samples), min_child_weight=float(min_child_weight),
           n_estimators=int(n_estimators), n_jobs=10, num_leaves=int(num_leaves),
           random_state=None, reg_alpha=0.0, reg_lambda=0.0, max_depth=int(max_depth),
           subsample=float(subsample), min_split_gain=float(min_split_gain), )
    cv_score = cross_val_score(model, X_train, X_train_target, scoring="accuracy", cv=5).mean()
    return cv_score
# 使用贝叶斯优化
lgb_bo = BayesianOptimization(
        lgb_cv,
        {'colsample_bytree': (0.7, 1),
         'learning_rate': (0.0001, 0.1),
         'min_child_samples': (2, 100),
         'min_child_weight':(0.0001, 0.1),
         'n_estimators': (500, 10000),
         'num_leaves': (5, 250),
         'subsample': (0.7, 1),
         'max_depth': (2, 100),
         'min_split_gain': (0.1, 1)
         }
    )
lgb_bo.maximize()
#%%
lgb_bo.max
#%%
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
import lightgbm as lgb


# kf = StratifiedKFold(n_splits=10,random_state=20,shuffle=True)
kf = TimeSeriesSplit(n_splits=10)
# kf = GapLeavePOut(p=35000, gap_before=11000, gap_after=24000)
y_pred = np.zeros(len(X_test_target))

for fold, (train_index, val_index) in enumerate(kf.split(X_train, X_train_target)):
    x_train, x_val = X_train[train_index], X_train[val_index]
    y_train, y_val = X_train_target[train_index], X_train_target[val_index]
    train_set = lgb.Dataset(x_train, y_train)
    val_set = lgb.Dataset(x_val, y_val)

    params = {
        'boosting_type': 'gbdt',
        'metric': {'cross_entropy','auc','average_precision',},
        'objective': 'cross_entropy',  # regression,binary,multiclass
        # 'num_class': 3,
        'seed': 666,
        'num_leaves': 164,
        'learning_rate': 0.02,
        'max_depth': 3,
        'n_estimators': 973,
        # 'lambda_l1': 1,
        # 'lambda_l2': 1,
        # 'bagging_fraction': 1,
        # 'bagging_freq': 1,
        'colsample_bytree': 0.76,
        'subsample': 0.78,
        'min_child_samples': 94,
        'min_child_weight': 0.0073,
        'min_split_gain': 0.65,
        'verbose': -1,
        # 'cross_entropy':'xentropy'
    }

    model = lgb.train(params, train_set, num_boost_round=5000, early_stopping_rounds=50,
                      valid_sets=[val_set], verbose_eval=100)

    y_pred += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
lgb.plot_importance(model, max_num_features=20)
    # a = y_pred[:,1]
# plt.show()
#%% roccurve
from sklearn.metrics import roc_curve
from numpy import sqrt,argmax
fpr, tpr, thresholds = roc_curve(X_test_target, y_pred)
gmeans = sqrt(tpr * (1-fpr))
ix = argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
#%% train set score
train_target_1 = model.predict(X_train)
train_target_1 = [1 if y > 0.489394 else 0 for y in train_target_1]
print("训练集表现：")
print(accuracy_score(X_train_target,train_target_1))
print(classification_report(X_train_target,train_target_1))
#%% 2min
# thresholds_point = 0.4759
thresholds_point = 0.499305
yhat = [1 if y > thresholds_point else 0 for y in y_pred]
print("测试集表现：")
print(classification_report(yhat,X_test_target))
print(metrics.confusion_matrix(yhat, X_test_target))
#%%
import graphviz
from sklearn.tree import export_graphviz
dot_data = export_graphviz(lgb.estimators_[0, 0], out_file=None, feature_names=X_train.columns,
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('tree1.gv')
#%%
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from bayes_opt import BayesianOptimization

# kf = TimeSeriesSplit(n_splits=10)
def lgb_cv(colsample_bytree, learning_rate, min_child_samples, min_child_weight, n_estimators, num_leaves, subsample, max_depth, min_split_gain):
    model = LGBMClassifier(boosting_type='gbdt',objective='binary',
           colsample_bytree=float(colsample_bytree), learning_rate=float(learning_rate),
           min_child_samples=int(min_child_samples), min_child_weight=float(min_child_weight),
           n_estimators=int(n_estimators), n_jobs=10, num_leaves=int(num_leaves),
           random_state=None, reg_alpha=0.0, reg_lambda=0.0, max_depth=int(max_depth),
           subsample=float(subsample), min_split_gain=float(min_split_gain), )
    cv_score = cross_val_score(model, X_train, X_train_target, scoring="f1", cv=5).mean()
    return cv_score
# 使用贝叶斯优化
lgb_bo = BayesianOptimization(
        lgb_cv,
        {'colsample_bytree': (0.7, 1),
         'learning_rate': (0.0001, 0.1),
         'min_child_samples': (2, 100),
         'min_child_weight':(0.0001, 0.1),
         'n_estimators': (500, 10000),
         'num_leaves': (5, 250),
         'subsample': (0.7, 1),
         'max_depth': (2, 100),
         'min_split_gain': (0.1, 1)
         }
    )
lgb_bo.maximize()
#%%
lgb_bo.max
#%%
import time
start = time.time()

gbm = LGBMClassifier(boosting_type='gbdt', objective='binary',
                     colsample_bytree=0.7457536578281572,
                     min_child_samples=2, min_child_weight=0.05323003673865087, min_split_gain=0.9096276267623588,
                     n_estimators=8421,
                     # feature_fraction=1,
                     subsample=0.8455862164744741,
                     learning_rate=0.0333414613402326, max_depth=7, num_leaves=145,
                     reg_alpha=0.0, reg_lambda=0.0
                     )
# cv_score = cross_val_score(gbm, X_train, X_train_target, scoring="roc_auc", cv=5).mean()
# y_pred_gbm = cv_score
gbm.fit(X_train,X_train_target)
y_pred_gbm = gbm.predict(X_test)
end = time.time()
print('Total Time = %s'%(end-start))

print(accuracy_score(y_pred_gbm,X_test_target))
print("测试集表现：")
print(classification_report(y_pred_gbm,X_test_target))
print("评价指标-混淆矩阵：")
print(metrics.confusion_matrix(y_pred_gbm,X_test_target))
#%%
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(X_test_target, y_pred)
plt.figure()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(recall, precision)
plt.show()
#%%
p = [1 if y > 0.6 else 0 for y in p]
print("测试集表现：")
print(classification_report(p,X_test_target))
print(metrics.confusion_matrix(p, X_test_target))
#%%
import joblib
joblib.dump(model,'lightGBM_ru.pkl')
#%%
features = data.columns
features = pd.DataFrame(features)
features.to_csv('features.csv')
#%%
data = data.reset_index()
#%%
test_data = data[144718:146452]
test_data = test_data.reset_index(drop=True)
predict = pd.DataFrame(yhat,columns=list('P'))
predict['datetime'] = test_data['datetime']
predict['last_price'] = test_data['last_price']
predict['target'] = test_data['target']
#%%
model.save_model('lightGBM_ru.txt')
#%%
predict.to_csv('predict_1_0_GBDT_2min_4.6_4.16_ag_last_price.csv')
