from IPython.core.display import display, HTML
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#%%
data_2203 = pd.read_csv('Z:/Tick_TQ/p/p2001.csv')

col =['datetime','timestamp','last_price','highest','lowest','volume','amount','interest',
      'bid_price1','bid_size1','ask_price1','ask_size1']

data_2203.columns = col

#%%
data = data_2203[(data_2203.datetime>='2019-08-02 09:00:00')& (data_2203.datetime<='2019-11-25 14:59:59')]
#%%
data = data.dropna(axis=0,how='any')
#%%
data = data.iloc[1:,:]
#%%

#%%
data1 = data.iloc[:,1:].astype(float)
#%%
data1['datetime'] = data['datetime']
#%%
# data1 = data1.reset_index(drop=True).reset_index()

#%%
def calc_wap1(df):
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

def calc_wap3(df):
    wap = (df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1']) / (df['bid_size1'] + df['ask_size1'])
    return wap

def calc_wap5(df):
    rolling = 120
    df['size'] = df['volume'] - df['volume'].shift(1)
    df['pv'] = df['last_price'] * df['size']
    last_price_wap = df['pv'].rolling(rolling).sum() / df['size'].rolling(rolling).sum()
    return last_price_wap

#%%
# Calculate the realized volatility
def realized_volatility(series):
    return np.sqrt(np.sum(series ** 2))

def realized_quarticity(series):
    return (np.sum(series**4)*series.shape[0]/3)

def reciprocal_transformation(series):
    return np.sqrt(1/series)*100000

def square_root_translation(series):
    return series**(1/2)

#%%
def book_preprocessor(data):

    df = data

    rolling = 120
    df['volume_size'] = df['volume'] - df['volume'].shift(1)

    # Calculate Wap
    df['wap1'] = calc_wap1(df)
    df['wap3'] = calc_wap3(df)
    df['wap5'] = calc_wap5(df)
    df['wap1_quarticity']=realized_quarticity(df['wap1'])
    df['wap1_reciprocal'] = reciprocal_transformation(df['wap1'])
    df['wap1_square_root'] = square_root_translation(df['wap1'])
    df['wap3_quarticity']=realized_quarticity(df['wap3'])
    df['wap3_reciprocal'] = reciprocal_transformation(df['wap3'])
    df['wap3_square_root'] = square_root_translation(df['wap3'])
    df['wap5_quarticity'] = realized_quarticity(df['wap5'])
    df['wap5_reciprocal'] = reciprocal_transformation(df['wap5'])
    df['wap5_square_root'] = square_root_translation(df['wap5'])

    df['wap1_shift2'] = df['wap1'].shift(1) - df['wap1'].shift(2)
    df['wap1_shift4'] = df['wap1'].shift(1) - df['wap1'].shift(4)
    df['wap1_shift10'] = df['wap1'].shift(1) - df['wap1'].shift(10)
    df['wap1_shift25'] = df['wap1'].shift(1) - df['wap1'].shift(25)

    df['wap3_shift2'] = df['wap3'].shift(1) - df['wap3'].shift(2)
    df['wap3_shift4'] = df['wap3'].shift(1) - df['wap3'].shift(4)
    df['wap3_shift10'] = df['wap3'].shift(1) - df['wap3'].shift(10)
    df['wap3_shift25'] = df['wap3'].shift(1) - df['wap3'].shift(25)

    df['wap5_shift2'] = df['wap5'].shift(1) - df['wap5'].shift(2)
    df['wap5_shift3'] = df['wap5'].shift(1) - df['wap5'].shift(4)
    df['wap5_shift4'] = df['wap5'].shift(1) - df['wap5'].shift(10)
    df['wap5_shift5'] = df['wap5'].shift(1) - df['wap5'].shift(25)

    df['last_price_shift2'] = df['last_price'].shift(1) - df['last_price'].shift(2)
    df['last_price_shift10'] = df['last_price'].shift(1) - df['last_price'].shift(10)
    df['last_price_shift30'] = df['last_price'].shift(1) - df['last_price'].shift(30)
    df['last_price_shift100'] = df['last_price'].shift(1) - df['last_price'].shift(100)

    df['mid_price'] = np.where(df.volume_size > 0, (df.amount - df.amount.shift(1)) / df.volume_size, df.last_price)
    df['mid_price1'] = (df['ask_price1']+df['bid_price1'])/2

    df['HR1'] = ((df['bid_price1']-df['bid_price1'].shift(1))-(df['ask_price1']-df['ask_price1'].shift(1)))/((df['bid_price1']-df['bid_price1'].shift(1))+(df['ask_price1']-df['ask_price1'].shift(1)))

    df['pre_vtA'] = np.where(df.ask_price1==df.ask_price1.shift(1),df.ask_size1-df.ask_size1.shift(1),0)
    df['vtA'] = np.where(df.ask_price1>df.ask_price1.shift(1),df.ask_size1,df.pre_vtA)
    df['pre_vtB'] = np.where(df.bid_price1==df.bid_price1.shift(1),df.bid_size1-df.bid_size1.shift(1),0)
    df['vtB'] = np.where(df.bid_price1>df.bid_price1.shift(1),df.bid_size1,df.pre_vtB)

    df['Oiab'] = df['vtB']-df['vtA']

    df['bid_ask_size1_minus'] = df['bid_size1']-df['ask_size1']
    df['bid_ask_size1_plus'] = df['bid_size1']+df['ask_size1']

    df['bid_size1_shift'] = df['bid_size1']-df['bid_size1'].shift()
    df['ask_size1_shift'] = df['ask_size1']-df['ask_size1'].shift()

    df['bid_ask_size1_spread'] = df['bid_ask_size1_minus']/df['bid_ask_size1_plus']
    df['bid_amount_abs1'] = abs((df['amount']-df['amount'].shift(1))/df['volume_size']-df['bid_price1'])-abs((df['amount']-df['amount'].shift(1))/df['volume_size']-df['ask_price1'])

    # Calculate log returns

    df['rolling_mid_price_mean'] = df['mid_price'].rolling(rolling).mean()
    df['rolling_mid_price_std'] = df['mid_price'].rolling(rolling).std()
    df['roliing_mid_price1_mean'] = df['mid_price1'].rolling(rolling).mean()
    df['rolling_mid_price1_std'] = df['mid_price1'].rolling(rolling).std()

    df['rolling_HR1_mean'] = df['HR1'].rolling(rolling).mean()

    df['rolling_bid_ask_size1_minus_mean1'] = df['bid_ask_size1_minus'].rolling(rolling).mean()

    df['rolling_bid_size1_shift_mean1'] = df['bid_size1_shift'].rolling(rolling).mean()
    df['rolling_ask_size1_shift_mean1'] = df['ask_size1_shift'].rolling(rolling).mean()
    df['rolling_bid_ask_size1_spread_mean1'] = df['bid_ask_size1_spread'].rolling(rolling).mean()

    df['log_return1'] = np.log(df['wap1'].shift(1)/df['wap1'].shift(2))*100
    df['log_return3'] = np.log(df['wap3'].shift(1)/df['wap3'].shift(2))*100

    df['log_return_wap1_shift5'] = np.log(df['wap1'].shift(1)/df['wap1'].shift(5))*100
    df['log_return_wap1_shift15'] = np.log(df['wap1'].shift(1)/df['wap1'].shift(15))*100

    df['log_return_last_price_shift5'] = np.log(df['last_price'].shift(1)/df['last_price'].shift(5))*100
    df['log_return_last_price_shift15'] = np.log(df['last_price'].shift(1)/df['last_price'].shift(15))*100

    df['rolling_last_price_mean'] = df['last_price'].rolling(rolling).mean()
    df['rolling_last_price_std'] = df['last_price'].rolling(rolling).std()
    df['rolling_last_price_min'] = df['last_price'].rolling(rolling).min()
    df['rolling_last_price_max'] = df['last_price'].rolling(rolling).max()
    df['rolling_last_price_skew'] = df['last_price'].rolling(rolling).skew()
    df['rolling_last_price_kurt'] = df['last_price'].rolling(rolling).kurt()
    df['last_price_quantile_25'] = df['last_price'].rolling(rolling).quantile(.25)
    df['last_price_quantile_75'] = df['last_price'].rolling(rolling).quantile(.75)
    df['rolling_last_price_quarticity'] = realized_quarticity(df['last_price'])
    df['rolling_last_price_reciprocal'] = reciprocal_transformation(df['last_price'])
    df['rolling_last_price_square_root'] = square_root_translation(df['last_price'])

    df['ewm_last_price_mean'] = pd.DataFrame.ewm(df['last_price'],span=rolling).mean()


    df['rolling_mean1'] = df['wap1'].rolling(rolling).mean()
    df['rolling_std1'] = df['wap1'].rolling(rolling).std()
    df['rolling_min1'] = df['wap1'].rolling(rolling).min()
    df['rolling_max1'] = df['wap1'].rolling(rolling).max()
    df['rolling_skew1'] = df['wap1'].rolling(rolling).skew()
    df['rolling_kurt1'] = df['wap1'].rolling(rolling).kurt()
    df['rolling_quantile1_25'] = df['wap1'].rolling(rolling).quantile(.25)
    df['rolling_quantile1_75'] = df['wap1'].rolling(rolling).quantile(.75)


    df['rolling_mean3'] = df['wap3'].rolling(rolling).mean()
    df['rolling_var3'] = df['wap3'].rolling(rolling).var()
    df['rolling_min3'] = df['wap3'].rolling(rolling).min()
    df['rolling_max3'] = df['wap3'].rolling(rolling).max()
    df['rolling_skew3'] = df['wap3'].rolling(rolling).skew()
    df['rolling_kurt3'] = df['wap3'].rolling(rolling).kurt()
    df['rolling_median3'] = df['wap3'].rolling(rolling).median()
    df['rolling_quantile3_25'] = df['wap3'].rolling(rolling).quantile(.25)
    df['rolling_quantile3_75'] = df['wap3'].rolling(rolling).quantile(.75)


    df['rolling_mean5'] = df['wap5'].rolling(rolling).mean()
    df['rolling_std5'] = df['wap5'].rolling(rolling).std()
    df['rolling_min5'] = df['wap5'].rolling(rolling).min()
    df['rolling_max5'] = df['wap5'].rolling(rolling).max()
    df['rolling_skew5'] = df['wap5'].rolling(rolling).skew()
    df['rolling_kurt5'] = df['wap5'].rolling(rolling).kurt()
    df['rolling_median5'] = df['wap5'].rolling(rolling).median()
    df['rolling_quantile5_25'] = df['wap5'].rolling(rolling).quantile(.25)
    df['rolling_quantile5_75'] = df['wap5'].rolling(rolling).quantile(.75)

    # ewm
    df['ewm_mean3'] = pd.DataFrame.ewm(df['wap3'], span=rolling).mean()
    df['ewm_mean5'] = pd.DataFrame.ewm(df['wap5'], span=rolling).mean()

    df['wap_balance1'] = abs(df['wap1'] - df['wap3'])
    df['price_spread1'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1']) / 2)

    print(df.columns)
    return df

#%%
def trade_preprocessor(data):
    df = data
    # df['log_return'] = np.log(df['last_price']).shift()
    df['size'] = df['volume'] - df['volume'].shift(1)
    df['amount'] = df['last_price'] * df['size']

    rolling =120

    df['rolling_mean_size'] = df['size'].rolling(rolling).mean()
    df['rolling_var_size'] = df['size'].rolling(rolling).var()
    df['rolling_std_size'] = df['size'].rolling(rolling).std()
    df['rolling_sum_size'] = df['size'].rolling(rolling).sum()
    df['rolling_min_size'] = df['size'].rolling(rolling).min()
    df['rolling_max_size'] = df['size'].rolling(rolling).max()
    df['rolling_skew_size'] = df['size'].rolling(rolling).skew()
    df['rolling_kurt_size'] = df['size'].rolling(rolling).kurt()
    df['rolling_median_size'] = df['size'].rolling(rolling).median()

    df['ewm_mean_size'] = pd.DataFrame.ewm(df['size'], span=rolling).mean()
    df['ewm_std_size'] = pd.DataFrame.ewm(df['size'], span=rolling).std()

    df['size_percentile_25'] = df['size'].rolling(rolling).quantile(.25)
    df['size_percentile_75'] = df['size'].rolling(rolling).quantile(.75)
    df['size_percentile'] = df['size_percentile_75'] - df['size_percentile_25']

    df['price_percentile_25'] = df['last_price'].rolling(rolling).quantile(.25)
    df['price_percentile_75'] = df['last_price'].rolling(rolling).quantile(.75)
    df['price_percentile'] = df['price_percentile_75'] - df['price_percentile_25']



    df['rolling_mean_amount'] = df['amount'].rolling(rolling).mean()
    df['rolling_quantile_25_amount'] = df['amount'].rolling(rolling).quantile(.25)
    df['rolling_quantile_50_amount'] = df['amount'].rolling(rolling).quantile(.50)
    df['rolling_quantile_75_amount'] = df['amount'].rolling(rolling).quantile(.75)

    df['ewm_mean_amount'] = pd.DataFrame.ewm(df['amount'], span=rolling).mean()

    print (df.columns)
    return df

#%%
df1=book_preprocessor(data1)
train_data = trade_preprocessor(df1)

# #%%
train_data.to_csv('F:/XDQ/p2001_20190802_20191125.csv')