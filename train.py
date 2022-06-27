#%%
from IPython.core.display import display, HTML

import pandas as pd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
import gc

from joblib import Parallel, delayed

from sklearn import preprocessing, model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns
import numpy.matlib


path_submissions = '/'

target_name = 'target'
scores_folds = {}
#%%
# data directory
data_dir = '../input/optiver-realized-volatility-prediction/'

# Function to calculate first WAP
def calc_wap1(df):
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

# Function to calculate second WAP
def calc_wap2(df):
    wap = (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

def calc_wap3(df):
    wap = (df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

def calc_wap4(df):
    wap = (df['bid_price2'] * df['bid_size2'] + df['ask_price2'] * df['ask_size2']) / (df['bid_size2'] + df['ask_size2'])
    return wap

# Function to calculate the log of the return
# Remember that logb(x / y) = logb(x) - logb(y)
def log_return(series):
    return np.log(series).diff()

# Calculate the realized volatility
def realized_volatility(series):
    return np.sqrt(np.sum(series**2))

# Function to count unique elements of a series
def count_unique(series):
    return len(np.unique(series))

# Function to read our base train and test set
def read_train_test():
    train = pd.read_csv('../input/optiver-realized-volatility-prediction/train.csv')
    test = pd.read_csv('../input/optiver-realized-volatility-prediction/test.csv')
    # Create a key to merge with book and trade data
    train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)
    test['row_id'] = test['stock_id'].astype(str) + '-' + test['time_id'].astype(str)
    print(f'Our training set has {train.shape[0]} rows')
    return train, test


def book_preprocessor(file_path):
    df = pd.read_parquet(file_path)
    # Calculate Wap
    df['wap1'] = calc_wap1(df)
    df['wap2'] = calc_wap2(df)
    df['wap3'] = calc_wap3(df)
    df['wap4'] = calc_wap4(df)
    # Calculate log returns
    df['log_return1'] = df.groupby(['time_id'])['wap1'].apply(log_return)
    df['log_return2'] = df.groupby(['time_id'])['wap2'].apply(log_return)
    df['log_return3'] = df.groupby(['time_id'])['wap3'].apply(log_return)
    df['log_return4'] = df.groupby(['time_id'])['wap4'].apply(log_return)
    # Calculate wap balance
    df['wap_balance'] = abs(df['wap1'] - df['wap2'])
    # Calculate spread
    df['price_spread'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1']) / 2)
    df['price_spread2'] = (df['ask_price2'] - df['bid_price2']) / ((df['ask_price2'] + df['bid_price2']) / 2)
    df['bid_spread'] = df['bid_price1'] - df['bid_price2']
    df['ask_spread'] = df['ask_price1'] - df['ask_price2']
    df["bid_ask_spread"] = abs(df['bid_spread'] - df['ask_spread'])
    df['total_volume'] = (df['ask_size1'] + df['ask_size2']) + (df['bid_size1'] + df['bid_size2'])
    df['volume_imbalance'] = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))

    # Dict for aggregations
    create_feature_dict = {
        'wap1': [np.sum, np.std],
        'wap2': [np.sum, np.std],
        'wap3': [np.sum, np.std],
        'wap4': [np.sum, np.std],
        'log_return1': [realized_volatility],
        'log_return2': [realized_volatility],
        'log_return3': [realized_volatility],
        'log_return4': [realized_volatility],
        'wap_balance': [np.sum, np.max],
        'price_spread': [np.sum, np.max],
        'price_spread2': [np.sum, np.max],
        'bid_spread': [np.sum, np.max],
        'ask_spread': [np.sum, np.max],
        'total_volume': [np.sum, np.max],
        'volume_imbalance': [np.sum, np.max],
        "bid_ask_spread": [np.sum, np.max],
    }
    create_feature_dict_time = {
        'log_return1': [realized_volatility],
        'log_return2': [realized_volatility],
        'log_return3': [realized_volatility],
        'log_return4': [realized_volatility],
    }

    def get_stats_window(fe_dict, seconds_in_bucket, add_suffix=False):
        # Group by the window
        df_feature = df[df['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(fe_dict).reset_index()
        # Rename columns joining suffix
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
        return df_feature

    # Get the stats for different windows
    df_feature = get_stats_window(create_feature_dict, seconds_in_bucket=0, add_suffix=False)
    df_feature_500 = get_stats_window(create_feature_dict_time, seconds_in_bucket=500, add_suffix=True)
    df_feature_400 = get_stats_window(create_feature_dict_time, seconds_in_bucket=400, add_suffix=True)
    df_feature_300 = get_stats_window(create_feature_dict_time, seconds_in_bucket=300, add_suffix=True)
    df_feature_200 = get_stats_window(create_feature_dict_time, seconds_in_bucket=200, add_suffix=True)
    df_feature_100 = get_stats_window(create_feature_dict_time, seconds_in_bucket=100, add_suffix=True)

    df_feature = df_feature.merge(df_feature_500, how='left', left_on='time_id_', right_on='time_id__500')
    df_feature = df_feature.merge(df_feature_400, how='left', left_on='time_id_', right_on='time_id__400')
    df_feature = df_feature.merge(df_feature_300, how='left', left_on='time_id_', right_on='time_id__300')
    df_feature = df_feature.merge(df_feature_200, how='left', left_on='time_id_', right_on='time_id__200')
    df_feature = df_feature.merge(df_feature_100, how='left', left_on='time_id_', right_on='time_id__100')
    # Drop unnecesary time_ids
    df_feature.drop(['time_id__500', 'time_id__400', 'time_id__300', 'time_id__200', 'time_id__100'], axis=1,
                    inplace=True)

    # Create row_id so we can merge
    stock_id = file_path.split('=')[1]
    df_feature['row_id'] = df_feature['time_id_'].apply(lambda x: f'{stock_id}-{x}')
    df_feature.drop(['time_id_'], axis=1, inplace=True)
    return df_feature

def trade_preprocessor(file_path):
    df = pd.read_parquet(file_path)
    df['log_return'] = df.groupby('time_id')['price'].apply(log_return)
    df['amount'] = df['price'] * df['size']
    # Dict for aggregations
    create_feature_dict = {
        'log_return': [realized_volatility],
        'seconds_in_bucket': [count_unique],
        'size': [np.sum, np.max, np.min],
        'order_count': [np.sum, np.max],
        'amount': [np.sum, np.max, np.min],
    }
    create_feature_dict_time = {
        'log_return': [realized_volatility],
        'seconds_in_bucket': [count_unique],
        'size': [np.sum],
        'order_count': [np.sum],
    }

    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(fe_dict, seconds_in_bucket, add_suffix=False):
        # Group by the window
        df_feature = df[df['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(
            fe_dict).reset_index()
        # Rename columns joining suffix
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
        return df_feature



#%%
from sklearn.cluster import KMeans
import pandas as pd

# making agg features

train_p = pd.read_csv('train.csv')
train_p = train_p.pivot(index='time_id', columns='stock_id', values='target')

corr = train_p.corr()
ids = corr.index
#%%
kmeans = KMeans(n_clusters=7, random_state=0).fit(corr.values)
print(kmeans.labels_)
#%%
l = []
for n in range(7):
    l.append ( [ (x-1) for x in ( (ids+1)*(kmeans.labels_ == n)) if x > 0] )
#%%
mat = []
matTest = []

n = 0
for ind in l:
    print(ind)
    newDf = train.loc[train['stock_id'].isin(ind)]
    newDf = newDf.groupby(['time_id']).agg(np.nanmean)
    newDf.loc[:, 'stock_id'] = str(n) + 'c1'
    mat.append(newDf)

    newDf = test.loc[test['stock_id'].isin(ind)]
    newDf = newDf.groupby(['time_id']).agg(np.nanmean)
    newDf.loc[:, 'stock_id'] = str(n) + 'c1'
    matTest.append(newDf)

    n += 1

mat1 = pd.concat(mat).reset_index()
mat1.drop(columns=['target'], inplace=True)

mat2 = pd.concat(matTest).reset_index()