from IPython.core.display import display, HTML
import pandas as pd
import numpy as np
import pandas as pd

#%%
data_2203 = pd.read_csv('Z:/Tick_TQ/ni/ni2108.csv')

col =['datetime','timestamp','last_price','highest','lowest','volume','amount','interest',
      'bid_price1','bid_size1','ask_price1','ask_size1','bid_price2','bid_size2','ask_price2','ask_size2',
      'bid_price3','bid_size3','ask_price3','ask_size3','bid_price4','bid_size4','ask_price4','ask_size4',
      'bid_price5','bid_size5','ask_price5','ask_size5']
data_2203.columns = col

# data_2203 = data_2203.drop(['highest','lowest','interest'],axis=1)
# data_2103['last_price_target'] = (data_2103['ask_price1']+data_2103['bid_price1'])/2

#%%
data = data_2203[(data_2203.datetime>='2021-07-01 21:00:00')& (data_2203.datetime<='2022-07-21 15:00:00')]
#%%
data = data.dropna(axis=0,how='any')
#%%
data = data.iloc[1:,:]
#%%
# from sklearn.preprocessing import FunctionTransformer
#
#
# def sin_transformer(period):
#     return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))
#
#
# def cos_transformer(period):
#     return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))
#
# data['sin'] = sin_transformer(60).fit_transform(data)['stime']
#%%
data1 = data.iloc[:,1:].astype(float)
#%%
data1['datetime'] = data['datetime']
#%%
data1 = data1.reset_index(drop=True).reset_index()
# train = data_2103[(data_2103.datetime>='2021-01-04 0:00:00')& (data_2103.datetime<='2021-01-08 23:59:59')]
# test = data_2103[(data_2103.datetime>='2021-01-11 0:00:00')& (data_2103.datetime<='2021-01-11 23:59:59')]

#%%
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

def calc_wap5(df):
    rolling = 15
    df['size'] = df['volume'] - df['volume'].shift(1)
    df['pv'] = df['last_price'] * df['size']
    last_price_wap = df['pv'].rolling(rolling).sum() / df['size'].rolling(rolling).sum()
    return last_price_wap

# Calculate the realized volatility
# def realized_volatility(series):
#     return np.sqrt(np.sum(series ** 2))
#
# def realized_quarticity(series):
#     return (np.sum(series**4)*series.shape[0]/3)*100000
#
# def reciprocal_transformation(series):
#     return np.sqrt(1/series)*100000
#
# def square_root_translation(series):
#     return series**(1/2)

# def realized_quadpower_quarticity(series):
#     series = abs(series.rolling(window=4).apply(np.product, raw=True))
#     return ((np.sum(series) * series.shape[0] * (np.pi**2))/4)*1000000

# def realized_tripower_quarticity(series):
#     series = series ** (4/3)
#     series = abs(series).rolling(window=3).apply(np.prod, raw=True)
#     return (series.shape[0]*0.25*((gamma(1/2)**3)/(gamma(7/6)**3))*np.sum(series))*1000000


#%%
# data['wap1'] = calc_wap1(data)
#%%
# data['1'] = np.log(data['wap1'].shift(1)/data['wap1'].shift(2))*100
#%%
# df_feature = data.groupby('datetime').agg({'1' : [realized_quadpower_quarticity,realized_tripower_quarticity]}).reset_index()
#%%

# df_feature.columns = ['_'.join(col) for col in df_feature.columns]
# train['log_return1'] = train.groupby(['datetime'])['wap1'].apply(log_return)
#%%
def book_preprocessor(data):

    df = data

    rolling = 120
    df['volume_size'] = df['volume'] - df['volume'].shift(1)


    df['last_price_shift2'] = df['last_price'].shift(1) - df['last_price'].shift(2)
    df['last_price_shift3'] = df['last_price'].shift(1) - df['last_price'].shift(3)
    df['last_price_shift4'] = df['last_price'].shift(1) - df['last_price'].shift(4)
    df['last_price_shift5'] = df['last_price'].shift(1) - df['last_price'].shift(5)
    df['last_price_shift6'] = df['last_price'].shift(1) - df['last_price'].shift(6)
    df['last_price_shift7'] = df['last_price'].shift(1) - df['last_price'].shift(7)
    df['last_price_shift8'] = df['last_price'].shift(1) - df['last_price'].shift(8)
    df['last_price_shift9'] = df['last_price'].shift(1) - df['last_price'].shift(9)


    df['mid_price'] = np.where(df.volume_size > 0, (df.amount - df.amount.shift(1)) / df.volume_size, df.last_price)
    df['mid_price1'] = (df['ask_price1']+df['bid_price1'])/2
    df['mid_price2'] = (df['ask_price2']+df['bid_price2'])/2
    df['mid_price3'] = (df['ask_price3'] + df['bid_price3']) / 2

    df['press_buy1'] = ((df['mid_price1']/(df['mid_price1']-df['bid_price1']))/((df['mid_price1']/(df['mid_price1']-df['bid_price1']))+(df['mid_price2']/(df['mid_price2']-df['bid_price2']))+(df['mid_price3']/(df['mid_price3']-df['bid_price3']))))*(df['bid_size1']+df['bid_size2']+df['bid_size3'])
    df['press_sell1'] = ((df['mid_price1']/(df['ask_price1']-df['mid_price1']))/((df['mid_price1']/(df['ask_price1']-df['mid_price1']))+(df['mid_price2']/(df['ask_price2']-df['mid_price2']))+(df['mid_price3']/(df['ask_price3']-df['mid_price3']))))*(df['ask_size1']+df['ask_size2']+df['ask_size3'])

    df['order_BS1'] = np.log(df['press_buy1'])-np.log(df['press_sell1'])

    df['HR1'] = ((df['bid_price1']-df['bid_price1'].shift(1))-(df['ask_price1']-df['ask_price1'].shift(1)))/((df['bid_price1']-df['bid_price1'].shift(1))+(df['ask_price1']-df['ask_price1'].shift(1)))

    df['pre_vtA'] = np.where(df.ask_price1==df.ask_price1.shift(1),df.ask_size1-df.ask_size1.shift(1),0)
    df['vtA'] = np.where(df.ask_price1>df.ask_price1.shift(1),df.ask_size1,df.pre_vtA)
    df['pre_vtB'] = np.where(df.bid_price1==df.bid_price1.shift(1),df.bid_size1-df.bid_size1.shift(1),0)
    df['vtB'] = np.where(df.bid_price1>df.bid_price1.shift(1),df.bid_size1,df.pre_vtB)

    df['Oiab'] = df['vtB']-df['vtA']


    df['bid_size1_2_minus'] = df['bid_size1']-df['bid_size2']
    df['ask_size1_2_minus'] = df['ask_size1']-df['ask_size2']

    df['bid_ask_size1_minus'] = df['bid_size1']-df['ask_size1']
    df['bid_ask_size1_plus'] = df['bid_size1']+df['ask_size1']

    df['bid_ask_size2_minus'] = df['bid_size2']-df['ask_size2']
    df['bid_ask_size2_plus'] = df['bid_size2']+df['ask_size2']

    df['bid_size1_shift'] = df['bid_size1']-df['bid_size1'].shift()
    df['bid_size2_shift'] = df['bid_size2']-df['bid_size2'].shift()
    df['ask_size1_shift'] = df['ask_size1']-df['ask_size1'].shift()
    df['ask_size2_shift'] = df['ask_size2']-df['ask_size2'].shift()

    df['bid_ask_size1_spread'] = df['bid_ask_size1_minus']/df['bid_ask_size1_plus']
    df['bid_ask_size2_spread'] = df['bid_ask_size2_minus'] / df['bid_ask_size2_plus']
    df['bid_amount_abs1'] = abs((df['amount']-df['amount'].shift(1))/df['volume_size']-df['bid_price1'])-abs((df['amount']-df['amount'].shift(1))/df['volume_size']-df['ask_price1'])

    # Calculate log returns

    df['rolling_mid_price_mean'] = df['mid_price'].rolling(rolling).mean()
    df['rolling_mid_price_std'] = df['mid_price'].rolling(rolling).std()
    df['roliing_mid_price1_mean'] = df['mid_price1'].rolling(rolling).mean()
    df['rolling_mid_price1_std'] = df['mid_price1'].rolling(rolling).std()
    df['roliing_mid_price2_mean'] = df['mid_price2'].rolling(rolling).mean()
    df['rolling_mid_price2_std'] = df['mid_price2'].rolling(rolling).std()
    df['roliing_mid_price3_mean'] = df['mid_price3'].rolling(rolling).mean()
    df['rolling_mid_price3_std'] = df['mid_price3'].rolling(rolling).std()

    df['rolling_order_BS1_mean'] = df['order_BS1'].rolling(rolling).mean()
    df['rolling_order_BS1_std'] = df['order_BS1'].rolling(rolling).std()


    df['rolling_HR1_mean'] = df['HR1'].rolling(rolling).mean()
    df['rolling_HR1_std'] = df['HR1'].rolling(rolling).std()

    df['rolling_vtA_mean'] = df['vtA'].rolling(rolling).mean()
    df['rolling_vtA_std'] = df['vtA'].rolling(rolling).std()
    df['rolling_vtB_mean'] = df['vtB'].rolling(rolling).mean()
    df['rolling_vtB_std'] = df['vtB'].rolling(rolling).std()

    df['rolling_Oiab_mean'] = df['Oiab'].rolling(rolling).mean()
    df['rolling_Oiab_std'] = df['Oiab'].rolling(rolling).std()

    df['rolling_bid_size1_2_minus_mean1'] = df['bid_size1_2_minus'].rolling(rolling).mean()
    df['rolling_bid_size1_2_minus_std1'] = df['bid_size1_2_minus'].rolling(rolling).std()
    df['rolling_ask_size1_2_minus_mean1'] = df['ask_size1_2_minus'].rolling(rolling).mean()
    df['rolling_ask_size1_2_minus_std1'] = df['ask_size1_2_minus'].rolling(rolling).std()

    df['rolling_bid_ask_size1_minus_mean1'] = df['bid_ask_size1_minus'].rolling(rolling).mean()
    df['rolling_bid_ask_size1_minus_std1'] = df['bid_ask_size1_minus'].rolling(rolling).std()
    df['rolling_bid_ask_size2_minus_mean1'] = df['bid_ask_size2_minus'].rolling(rolling).mean()
    df['rolling_bid_ask_size2_minus_std1'] = df['bid_ask_size2_minus'].rolling(rolling).std()

    df['rolling_bid_size1_shift_mean1'] = df['bid_size1_shift'].rolling(rolling).mean()
    df['rolling_bid_size1_shift_std1'] = df['bid_size1_shift'].rolling(rolling).std()
    df['rolling_ask_size1_shift_mean1'] = df['ask_size1_shift'].rolling(rolling).mean()
    df['rolling_ask_size1_shift_std1'] = df['ask_size1_shift'].rolling(rolling).std()

    df['rolling_bid_ask_size1_spread_mean1'] = df['bid_ask_size1_spread'].rolling(rolling).mean()
    df['rolling_bid_ask_size1_spread_std1'] = df['bid_ask_size1_spread'].rolling(rolling).std()

    df['rolling_bid_ask_size2_spread_mean1'] = df['bid_ask_size2_spread'].rolling(rolling).mean()
    df['rolling_bid_ask_size2_spread_std1'] = df['bid_ask_size2_spread'].rolling(rolling).std()




    df['log_return1'] = np.log(df['wap1'].shift(1)/df['wap1'].shift(2))*100
    df['log_return2'] = np.log(df['wap2'].shift(1)/df['wap2'].shift(2))*100
    df['log_return3'] = np.log(df['wap3'].shift(1)/df['wap3'].shift(2))*100
    df['log_return4'] = np.log(df['wap4'].shift(1)/df['wap4'].shift(2))*100

    df['log_return_wap1_shift3'] = np.log(df['wap1'].shift(1)/df['wap1'].shift(3))*100
    df['log_return_wap1_shift4'] = np.log(df['wap1'].shift(1)/df['wap1'].shift(4))*100
    df['log_return_wap1_shift5'] = np.log(df['wap1'].shift(1) / df['wap1'].shift(5))*100
    df['log_return_wap1_shift6'] = np.log(df['wap1'].shift(1) / df['wap1'].shift(6))*100
    df['log_return_wap1_shift7'] = np.log(df['wap1'].shift(1) / df['wap1'].shift(7))*100
    df['log_return_wap1_shift8'] = np.log(df['wap1'].shift(1) / df['wap1'].shift(8))*100
    df['log_return_wap1_shift9'] = np.log(df['wap1'].shift(1) / df['wap1'].shift(9))*100


    df['log_return_last_price_shift2'] = np.log(df['last_price'].shift(1)/df['last_price'].shift(2))*100
    df['log_return_last_price_shift3'] = np.log(df['last_price'].shift(1)/df['last_price'].shift(3))*100
    df['log_return_last_price_shift4'] = np.log(df['last_price'].shift(1)/df['last_price'].shift(4))*100
    df['log_return_last_price_shift5'] = np.log(df['last_price'].shift(1)/df['last_price'].shift(5))*100
    df['log_return_last_price_shift6'] = np.log(df['last_price'].shift(1)/df['last_price'].shift(6))*100
    df['log_return_last_price_shift7'] = np.log(df['last_price'].shift(1)/df['last_price'].shift(7))*100
    df['log_return_last_price_shift8'] = np.log(df['last_price'].shift(1)/df['last_price'].shift(8))*100
    df['log_return_last_price_shift9'] = np.log(df['last_price'].shift(1)/df['last_price'].shift(9))*100

    df['rolling_last_price_mean'] = df['last_price'].rolling(rolling).mean()
    df['rolling_last_price_var'] = df['last_price'].rolling(rolling).var()
    df['rolling_last_price_std'] = df['last_price'].rolling(rolling).std()
    df['rolling_last_price_min'] = df['last_price'].rolling(rolling).min()
    df['rolling_last_price_max'] = df['last_price'].rolling(rolling).max()
    df['rolling_last_price_sum'] = df['last_price'].rolling(rolling).sum()
    df['rolling_last_price_corr'] = df['last_price'].rolling(rolling).corr(df['last_price'])
    df['rolling_last_price_skew'] = df['last_price'].rolling(rolling).skew()
    df['rolling_last_price_kurt'] = df['last_price'].rolling(rolling).kurt()
    df['rolling_last_price_median'] = df['last_price'].rolling(rolling).median()
    df['last_price_quantile_25'] = df['last_price'].rolling(rolling).quantile(.25)
    df['last_price_quantile_50'] = df['last_price'].rolling(rolling).quantile(.50)
    df['last_price_quantile_75'] = df['last_price'].rolling(rolling).quantile(.75)

    df['ewm_last_price_mean'] = pd.DataFrame.ewm(df['last_price'],span=rolling).mean()
    df['ewm_last_price_std'] = pd.DataFrame.ewm(df['last_price'], span=rolling).std()
    df['ewm_last_price_var'] = pd.DataFrame.ewm(df['last_price'], span=rolling).var()
    df['ewm_last_price_corr'] = pd.DataFrame.ewm(df['last_price'], span=rolling).corr(df['last_price'])
    df['ewm_last_price_cov'] = pd.DataFrame.ewm(df['last_price'], span=rolling).cov()


    df['rolling_mean1'] = df['wap1'].rolling(rolling).mean()
    df['rolling_var1'] = df['wap1'].rolling(rolling).var()
    df['rolling_std1'] = df['wap1'].rolling(rolling).std()
    df['rolling_sum1'] = df['wap1'].rolling(rolling).sum()
    df['rolling_min1'] = df['wap1'].rolling(rolling).min()
    df['rolling_max1'] = df['wap1'].rolling(rolling).max()
    df['rolling_corr1'] = df['wap1'].rolling(rolling).corr(df['last_price'])
    df['rolling_skew1'] = df['wap1'].rolling(rolling).skew()
    df['rolling_kurt1'] = df['wap1'].rolling(rolling).kurt()
    df['rolling_median1'] = df['wap1'].rolling(rolling).median()
    df['rolling_quantile1_25'] = df['wap1'].rolling(rolling).quantile(.25)
    df['rolling_quantile1_50'] = df['wap1'].rolling(rolling).quantile(.50)
    df['rolling_quantile1_75'] = df['wap1'].rolling(rolling).quantile(.75)

    df['rolling_mean2'] = df['wap2'].rolling(rolling).mean()
    df['rolling_var2'] = df['wap2'].rolling(rolling).var()
    df['rolling_std2'] = df['wap2'].rolling(rolling).std()
    df['rolling_sum2'] = df['wap2'].rolling(rolling).sum()
    df['rolling_min2'] = df['wap2'].rolling(rolling).min()
    df['rolling_max2'] = df['wap2'].rolling(rolling).max()
    df['rolling_corr2'] = df['wap2'].rolling(rolling).corr(df['last_price'])
    df['rolling_skew2'] = df['wap2'].rolling(rolling).skew()
    df['rolling_kurt2'] = df['wap2'].rolling(rolling).kurt()
    df['rolling_median2'] = df['wap2'].rolling(rolling).median()
    df['rolling_quantile2_25'] = df['wap2'].rolling(rolling).quantile(.25)
    df['rolling_quantile2_50'] = df['wap2'].rolling(rolling).quantile(.50)
    df['rolling_quantile2_75'] = df['wap2'].rolling(rolling).quantile(.75)

    df['rolling_mean3'] = df['wap3'].rolling(rolling).mean()
    df['rolling_var3'] = df['wap3'].rolling(rolling).var()
    df['rolling_std3'] = df['wap3'].rolling(rolling).std()
    df['rolling_sum3'] = df['wap3'].rolling(rolling).sum()
    df['rolling_min3'] = df['wap3'].rolling(rolling).min()
    df['rolling_max3'] = df['wap3'].rolling(rolling).max()
    df['rolling_corr3'] = df['wap3'].rolling(rolling).corr(df['last_price'])
    df['rolling_skew3'] = df['wap3'].rolling(rolling).skew()
    df['rolling_kurt3'] = df['wap3'].rolling(rolling).kurt()
    df['rolling_median3'] = df['wap3'].rolling(rolling).median()
    df['rolling_quantile3_25'] = df['wap3'].rolling(rolling).quantile(.25)
    df['rolling_quantile3_50'] = df['wap3'].rolling(rolling).quantile(.50)
    df['rolling_quantile3_75'] = df['wap3'].rolling(rolling).quantile(.75)

    df['rolling_mean4'] = df['wap4'].rolling(rolling).mean()
    df['rolling_var4'] = df['wap4'].rolling(rolling).var()
    df['rolling_std4'] = df['wap4'].rolling(rolling).std()
    df['rolling_sum4'] = df['wap4'].rolling(rolling).sum()
    df['rolling_min4'] = df['wap4'].rolling(rolling).min()
    df['rolling_max4'] = df['wap4'].rolling(rolling).max()
    df['rolling_corr4'] = df['wap4'].rolling(rolling).corr(df['last_price'])
    df['rolling_skew4'] = df['wap4'].rolling(rolling).skew()
    df['rolling_kurt4'] = df['wap4'].rolling(rolling).kurt()
    df['rolling_median4'] = df['wap4'].rolling(rolling).median()
    df['rolling_quantile4_25'] = df['wap4'].rolling(rolling).quantile(.25)
    df['rolling_quantile4_50'] = df['wap4'].rolling(rolling).quantile(.50)
    df['rolling_quantile4_75'] = df['wap4'].rolling(rolling).quantile(.75)

    df['rolling_mean5'] = df['wap5'].rolling(rolling).mean()
    df['rolling_var5'] = df['wap5'].rolling(rolling).var()
    df['rolling_std5'] = df['wap5'].rolling(rolling).std()
    df['rolling_sum5'] = df['wap5'].rolling(rolling).sum()
    df['rolling_min5'] = df['wap5'].rolling(rolling).min()
    df['rolling_max5'] = df['wap5'].rolling(rolling).max()
    df['rolling_corr5'] = df['wap5'].rolling(rolling).corr(df['last_price'])
    df['rolling_skew5'] = df['wap5'].rolling(rolling).skew()
    df['rolling_kurt5'] = df['wap5'].rolling(rolling).kurt()
    df['rolling_median5'] = df['wap5'].rolling(rolling).median()
    df['rolling_quantile5_25'] = df['wap5'].rolling(rolling).quantile(.25)
    df['rolling_quantile5_50'] = df['wap5'].rolling(rolling).quantile(.50)
    df['rolling_quantile5_75'] = df['wap5'].rolling(rolling).quantile(.75)

    # ewm
    df['ewm_mean1'] = pd.DataFrame.ewm(df['wap1'], span=rolling).mean()
    df['ewm_std1'] = pd.DataFrame.ewm(df['wap1'], span=rolling).std()
    df['ewm_var1'] = pd.DataFrame.ewm(df['wap1'], span=rolling).var()
    df['ewm_corr1'] = pd.DataFrame.ewm(df['wap1'], span=rolling).corr(df['last_price'])
    df['ewm_cov1'] = pd.DataFrame.ewm(df['wap1'], span=rolling).cov()


    df['ewm_mean2'] = pd.DataFrame.ewm(df['wap2'], span=rolling).mean()
    df['ewm_std2'] = pd.DataFrame.ewm(df['wap2'], span=rolling).std()
    df['ewm_var2'] = pd.DataFrame.ewm(df['wap2'], span=rolling).var()
    df['ewm_corr2'] = pd.DataFrame.ewm(df['wap2'], span=rolling).corr(df['last_price'])
    df['ewm_cov2'] = pd.DataFrame.ewm(df['wap2'], span=rolling).cov()

    df['ewm_mean3'] = pd.DataFrame.ewm(df['wap3'], span=rolling).mean()
    df['ewm_std3'] = pd.DataFrame.ewm(df['wap3'], span=rolling).std()
    df['ewm_var3'] = pd.DataFrame.ewm(df['wap3'], span=rolling).var()
    df['ewm_corr3'] = pd.DataFrame.ewm(df['wap3'], span=rolling).corr(df['last_price'])
    df['ewm_cov3'] = pd.DataFrame.ewm(df['wap3'], span=rolling).cov()

    df['ewm_mean4'] = pd.DataFrame.ewm(df['wap4'], span=rolling).mean()
    df['ewm_std4'] = pd.DataFrame.ewm(df['wap4'], span=rolling).std()
    df['ewm_var4'] = pd.DataFrame.ewm(df['wap4'], span=rolling).var()
    df['ewm_corr4'] = pd.DataFrame.ewm(df['wap4'], span=rolling).corr(df['last_price'])
    df['ewm_cov4'] = pd.DataFrame.ewm(df['wap4'], span=rolling).cov()

    df['ewm_mean5'] = pd.DataFrame.ewm(df['wap5'], span=rolling).mean()
    df['ewm_std5'] = pd.DataFrame.ewm(df['wap5'], span=rolling).std()
    df['ewm_var5'] = pd.DataFrame.ewm(df['wap5'], span=rolling).var()
    df['ewm_corr5'] = pd.DataFrame.ewm(df['wap5'], span=rolling).corr(df['last_price'])
    df['ewm_cov5'] = pd.DataFrame.ewm(df['wap5'], span=rolling).cov()

    # Calculate wap balance
    df['wap_balance1'] = abs(df['wap1'] - df['wap2'])



    # Calculate spread
    df['price_spread1'] = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1']) / 2)
    df['price_spread2'] = (df['ask_price2'] - df['bid_price2']) / ((df['ask_price2'] + df['bid_price2']) / 2)
    df['total_volume1'] = (df['ask_size1'] + df['ask_size2']) + (df['bid_size1'] + df['bid_size2'])
    df['volume_imbalance1'] = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))
    df['size_weight'] = ((df['bid_size1']+df['ask_size1'])*0.7)-((df['bid_size2']+df['ask_size2'])*0.3)

    # Dict for aggregations
    create_feature_dict = {
        'wap1_shift2': [np.mean, reciprocal_transformation, square_root_translation],
        'wap1_shift3': [np.mean, reciprocal_transformation, square_root_translation],
        'wap1_shift4': [np.mean, reciprocal_transformation, square_root_translation],
        'wap1_shift5': [np.mean, reciprocal_transformation, square_root_translation],
        'wap2_shift2': [np.mean, reciprocal_transformation, square_root_translation],
        'wap2_shift3': [np.mean, reciprocal_transformation, square_root_translation],
        'wap2_shift4': [np.mean, reciprocal_transformation, square_root_translation],
        'wap2_shift5': [np.mean, reciprocal_transformation, square_root_translation],
        'wap3_shift2': [np.mean, reciprocal_transformation, square_root_translation],
        'wap3_shift3': [np.mean, reciprocal_transformation, square_root_translation],
        'wap3_shift4': [np.mean, reciprocal_transformation, square_root_translation],
        'wap3_shift5': [np.mean, reciprocal_transformation, square_root_translation],
        'wap4_shift2': [np.mean, reciprocal_transformation, square_root_translation],
        'wap4_shift3': [np.mean, reciprocal_transformation, square_root_translation],
        'wap4_shift4': [np.mean, reciprocal_transformation, square_root_translation],
        'wap4_shift5': [np.mean, reciprocal_transformation, square_root_translation],
        'wap5_shift2': [np.mean, reciprocal_transformation, square_root_translation],
        'wap5_shift3': [np.mean, reciprocal_transformation, square_root_translation],
        'wap5_shift4': [np.mean, reciprocal_transformation, square_root_translation],
        'wap5_shift5': [np.mean, reciprocal_transformation, square_root_translation],
        'last_price_shift2': [np.mean, reciprocal_transformation, square_root_translation],
        'last_price_shift3': [np.mean, reciprocal_transformation, square_root_translation],
        'last_price_shift4': [np.mean, reciprocal_transformation, square_root_translation],
        'last_price_shift5': [np.mean, reciprocal_transformation, square_root_translation],
        'last_price_shift6': [np.mean, reciprocal_transformation, square_root_translation],
        'last_price_shift7': [np.mean, reciprocal_transformation, square_root_translation],
        'last_price_shift8': [np.mean, reciprocal_transformation, square_root_translation],
        'last_price_shift9': [np.mean, reciprocal_transformation, square_root_translation],
        'log_return_last_price_shift2': [np.mean, realized_volatility, realized_quarticity, reciprocal_transformation, square_root_translation],
        'log_return_last_price_shift3': [np.mean, realized_volatility, realized_quarticity, reciprocal_transformation, square_root_translation],
        'log_return_last_price_shift4': [np.mean, realized_volatility, realized_quarticity, reciprocal_transformation, square_root_translation],
        'log_return_last_price_shift5': [np.mean, realized_volatility, realized_quarticity, reciprocal_transformation,square_root_translation],
        'log_return_last_price_shift6': [np.mean, realized_volatility, realized_quarticity, reciprocal_transformation,square_root_translation],
        'log_return_last_price_shift7': [np.mean, realized_volatility, realized_quarticity, reciprocal_transformation,square_root_translation],
        'log_return_last_price_shift8': [np.mean, realized_volatility, realized_quarticity, reciprocal_transformation,square_root_translation],
        'log_return_last_price_shift9': [np.mean, realized_volatility, realized_quarticity, reciprocal_transformation,square_root_translation],
        'wap1': [np.mean, reciprocal_transformation, square_root_translation],
        'wap2': [np.mean, reciprocal_transformation, square_root_translation],
        'wap3': [np.mean, reciprocal_transformation, square_root_translation],
        'wap4': [np.mean, reciprocal_transformation, square_root_translation],
        'wap5': [np.mean, reciprocal_transformation, square_root_translation],
        'mid_price': [np.mean, reciprocal_transformation, square_root_translation],
        'mid_price1': [np.mean, reciprocal_transformation, square_root_translation],
        'mid_price2': [np.mean, reciprocal_transformation, square_root_translation],
        'mid_price3': [np.mean, reciprocal_transformation, square_root_translation],
        'press_buy1': [np.mean, reciprocal_transformation, square_root_translation],
        'press_sell1': [np.mean, reciprocal_transformation, square_root_translation],
        'order_BS1': [np.mean, reciprocal_transformation, square_root_translation],
        'HR1': [np.mean, reciprocal_transformation, square_root_translation],
        'vtA': [np.mean, reciprocal_transformation, square_root_translation],
        'vtB': [np.mean, reciprocal_transformation, square_root_translation],
        'Oiab': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_mid_price_mean': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_mid_price_std': [np.mean, reciprocal_transformation, square_root_translation],
        'roliing_mid_price1_mean': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_mid_price1_std': [np.mean, reciprocal_transformation, square_root_translation],
        'roliing_mid_price2_mean': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_mid_price2_std': [np.mean, reciprocal_transformation, square_root_translation],
        'roliing_mid_price3_mean': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_mid_price3_std': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_order_BS1_mean': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_order_BS1_std': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_HR1_mean': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_HR1_std': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_vtA_mean': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_vtA_std': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_vtB_mean': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_vtB_std': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_Oiab_mean': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_Oiab_std': [np.mean, reciprocal_transformation, square_root_translation],
        'log_return1': [np.mean, reciprocal_transformation, square_root_translation,realized_volatility, realized_quarticity],
        'log_return2': [np.mean, reciprocal_transformation, square_root_translation, realized_volatility, realized_quarticity],
        'log_return3': [np.mean, reciprocal_transformation, square_root_translation, realized_volatility, realized_quarticity],
        'log_return4': [np.mean, reciprocal_transformation, square_root_translation, realized_volatility, realized_quarticity],
        'log_return_wap1_shift3': [np.mean, reciprocal_transformation, square_root_translation, realized_volatility, realized_quarticity],
        'log_return_wap1_shift4': [np.mean, reciprocal_transformation, square_root_translation, realized_volatility, realized_quarticity],
        'log_return_wap1_shift5': [np.mean, reciprocal_transformation, square_root_translation, realized_volatility,realized_quarticity],
        'log_return_wap1_shift6': [np.mean, reciprocal_transformation, square_root_translation, realized_volatility,realized_quarticity],
        'log_return_wap1_shift7': [np.mean, reciprocal_transformation, square_root_translation, realized_volatility,realized_quarticity],
        'log_return_wap1_shift8': [np.mean, reciprocal_transformation, square_root_translation, realized_volatility,realized_quarticity],
        'log_return_wap1_shift9': [np.mean, reciprocal_transformation, square_root_translation, realized_volatility,realized_quarticity],
        'rolling_last_price_mean': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_last_price_var': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_last_price_std': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_last_price_min': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_last_price_max': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_last_price_sum': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_last_price_corr': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_last_price_skew': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_last_price_kurt': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_last_price_median': [np.mean, reciprocal_transformation, square_root_translation],
        'last_price_quantile_25': [np.mean, reciprocal_transformation, square_root_translation],
        'last_price_quantile_50': [np.mean, reciprocal_transformation, square_root_translation],
        'last_price_quantile_75': [np.mean, reciprocal_transformation, square_root_translation],
        'ewm_last_price_mean': [np.mean, reciprocal_transformation, square_root_translation],
        'ewm_last_price_std': [np.mean, reciprocal_transformation, square_root_translation],
        'ewm_last_price_var': [np.mean, reciprocal_transformation, square_root_translation],
        'ewm_last_price_corr': [np.mean, reciprocal_transformation, square_root_translation],
        'ewm_last_price_cov': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_mean1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_var1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_std1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_sum1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_min1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_max1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_corr1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_kurt1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_skew1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_median1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_quantile1_25': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_quantile1_50': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_quantile1_75': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_mean2': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_var2': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_std2': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_sum2': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_min2': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_max2': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_corr2': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_kurt2': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_skew2': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_median2': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_quantile2_25': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_quantile2_50': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_quantile2_75': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_mean3': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_var3': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_std3': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_sum3': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_min3': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_max3': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_corr3': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_kurt3': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_skew3': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_median3': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_quantile3_25': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_quantile3_50': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_quantile3_75': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_mean4': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_var4': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_std4': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_sum4': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_min4': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_max4': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_corr4': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_kurt4': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_skew4': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_median4': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_quantile4_25': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_quantile4_50': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_quantile4_75': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_mean5': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_var5': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_std5': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_sum5': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_min5': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_max5': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_corr5': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_kurt5': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_skew5': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_median5': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_quantile5_25': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_quantile5_50': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_quantile5_75': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_mean1': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_std1': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_var1': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_corr1': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_cov1': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_mean2': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_std2': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_var2': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_corr2': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_cov2': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_mean3': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_std3': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_var3': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_corr3': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_cov3': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_mean4': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_std4': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_var4': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_corr4': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_cov4': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_mean5': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_std5': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_var5': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_corr5': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_cov5': [np.mean,reciprocal_transformation, square_root_translation],
        'wap_balance1': [np.mean,reciprocal_transformation, square_root_translation],
        'total_volume1': [np.mean,reciprocal_transformation, square_root_translation],
        'volume_imbalance1': [np.mean,reciprocal_transformation, square_root_translation],
        'size_weight': [np.mean,reciprocal_transformation, square_root_translation],
        'bid_size1_2_minus': [np.mean, reciprocal_transformation, square_root_translation],
        'ask_size1_2_minus': [np.mean, reciprocal_transformation, square_root_translation],
        'bid_ask_size1_minus': [np.mean, reciprocal_transformation, square_root_translation],
        'bid_ask_size1_plus': [np.mean, reciprocal_transformation, square_root_translation],
        'bid_ask_size2_minus': [np.mean, reciprocal_transformation, square_root_translation],
        'bid_ask_size2_plus': [np.mean, reciprocal_transformation, square_root_translation],
        'bid_size1_shift': [np.mean, reciprocal_transformation, square_root_translation],
        'bid_size2_shift': [np.mean, reciprocal_transformation, square_root_translation],
        'ask_size1_shift': [np.mean, reciprocal_transformation, square_root_translation],
        'ask_size2_shift': [np.mean, reciprocal_transformation, square_root_translation],
        'bid_ask_size1_spread': [np.mean, reciprocal_transformation, square_root_translation],
        'bid_ask_size2_spread': [np.mean, reciprocal_transformation, square_root_translation],
        'bid_amount_abs1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_bid_size1_2_minus_mean1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_bid_size1_2_minus_std1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_ask_size1_2_minus_mean1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_ask_size1_2_minus_std1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_bid_ask_size1_minus_mean1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_bid_ask_size1_minus_std1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_bid_ask_size2_minus_mean1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_bid_ask_size2_minus_std1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_bid_size1_shift_mean1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_bid_size1_shift_std1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_ask_size1_shift_mean1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_ask_size1_shift_std1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_bid_ask_size1_spread_mean1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_bid_ask_size1_spread_std1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_bid_ask_size2_spread_mean1': [np.mean, reciprocal_transformation, square_root_translation],
        'rolling_bid_ask_size2_spread_std1': [np.mean, reciprocal_transformation, square_root_translation]
    }
    # create_feature_dict_time = {
    #     'log_return_last_price_shift2': [np.mean, realized_volatility, realized_quarticity],
    #     'log_return_last_price_shift3': [np.mean, realized_volatility, realized_quarticity],
    #     'log_return_last_price_shift4': [np.mean, realized_volatility, realized_quarticity],
    #     'log_return1': [np.mean, realized_volatility, realized_quarticity],
    #     'log_return2': [np.mean, realized_volatility, realized_quarticity],
    #     'log_return3': [np.mean, realized_volatility, realized_quarticity],
    #     'log_return4': [np.mean, realized_volatility, realized_quarticity],
    #     'log_return_wap1_shift3': [np.mean, realized_volatility, realized_quarticity],
    #     'log_return_wap1_shift4': [np.mean, realized_volatility, realized_quarticity]
    # }

    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(fe_dict, index, add_suffix=False):
        # Group by the window
        df_feature = df[df['index'] >= index].groupby(['datetime']).agg(fe_dict).reset_index()
        # Rename columns joining suffix
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(index))
        return df_feature
    # def get_stats_window(fe_dict, add_suffix=False):
    #     # Group by the window
    #     df_feature = df.groupby(['datetime']).agg(fe_dict).reset_index()
    #     # Rename columns joining suffix
    #     df_feature.columns = ['_'.join(col) for col in df_feature.columns]
    #     # Add a suffix to differentiate windows
    #     # if add_suffix:
    #     #     df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
    #     return df_feature

    # Get the stats for different windows
    df_feature = get_stats_window(create_feature_dict, index=0, add_suffix=False)
    # df_feature_500 = get_stats_window(create_feature_dict_time, index=25, add_suffix=True)
    # df_feature_400 = get_stats_window(create_feature_dict_time, index=20, add_suffix=True)
    # df_feature_300 = get_stats_window(create_feature_dict_time, index=15, add_suffix=True)
    # df_feature_200 = get_stats_window(create_feature_dict_time, index=10, add_suffix=True)
    # df_feature_100 = get_stats_window(create_feature_dict_time, index=5, add_suffix=True)
    #
    # # Merge all
    # df_feature = df_feature.merge(df_feature_500, how='left', left_on='datetime_', right_on='datetime__25')
    # df_feature = df_feature.merge(df_feature_400, how='left', left_on='datetime_', right_on='datetime__20')
    # df_feature = df_feature.merge(df_feature_300, how='left', left_on='datetime_', right_on='datetime__15')
    # df_feature = df_feature.merge(df_feature_200, how='left', left_on='datetime_', right_on='datetime__10')
    # df_feature = df_feature.merge(df_feature_100, how='left', left_on='datetime_', right_on='datetime__5')
    # print(df_feature)
    # # Drop unnecesary time_ids
    # df_feature.drop(['datetime__25', 'datetime__20', 'datetime__15', 'datetime__10', 'datetime__5'], axis=1,
    #                 inplace=True)

    # Create row_id so we can merge
    # stock_id = file_path.split('=')[1]
    # df_feature['row_id'] = df_feature['time_id_'].apply(lambda x: f'{stock_id}-{x}')
    # df_feature.drop(['datetime_'], axis=1, inplace=True)
    return df_feature

#%%
def trade_preprocessor(data):
    df = data
    # df['log_return'] = np.log(df['last_price']).shift()
    df['size'] = df['volume'] - df['volume'].shift(1)
    df['amount'] = df['last_price'] * df['size']

    rolling = 15

    df['rolling_mean_size'] = df['size'].rolling(rolling).mean()
    df['rolling_var_size'] = df['size'].rolling(rolling).var()
    df['rolling_std_size'] = df['size'].rolling(rolling).std()
    df['rolling_sum_size'] = df['size'].rolling(rolling).sum()
    df['rolling_min_size'] = df['size'].rolling(rolling).min()
    df['rolling_max_size'] = df['size'].rolling(rolling).max()
    df['rolling_corr_size'] = df['size'].rolling(rolling).corr(df['size'])
    df['rolling_skew_size'] = df['size'].rolling(rolling).skew()
    df['rolling_kurt_size'] = df['size'].rolling(rolling).kurt()
    df['rolling_median_size'] = df['size'].rolling(rolling).median()

    df['ewm_mean_size'] = pd.DataFrame.ewm(df['size'], span=rolling).mean()
    df['ewm_std_size'] = pd.DataFrame.ewm(df['size'], span=rolling).std()
    df['ewm_var_size'] = pd.DataFrame.ewm(df['size'], span=rolling).var()
    df['ewm_corr_size'] = pd.DataFrame.ewm(df['size'], span=rolling).corr(df['size'])
    df['ewm_cov_size'] = pd.DataFrame.ewm(df['size'], span=rolling).cov()

    df['size_percentile_25'] = df['size'].rolling(rolling).quantile(.25)
    df['size_percentile_75'] = df['size'].rolling(rolling).quantile(.75)
    df['size_percentile'] = df['size_percentile_75'] - df['size_percentile_25']

    df['price_percentile_25'] = df['last_price'].rolling(rolling).quantile(.25)
    df['price_percentile_75'] = df['last_price'].rolling(rolling).quantile(.75)
    df['price_percentile'] = df['price_percentile_75'] - df['price_percentile_25']



    df['rolling_mean_amount'] = df['amount'].rolling(rolling).mean()
    df['rolling_var_amount'] = df['amount'].rolling(rolling).var()
    df['rolling_std_amount'] = df['amount'].rolling(rolling).std()
    df['rolling_sum_amount'] = df['amount'].rolling(rolling).sum()
    df['rolling_min_amount'] = df['amount'].rolling(rolling).min()
    df['rolling_max_amount'] = df['amount'].rolling(rolling).max()
    df['rolling_corr_amount'] = df['amount'].rolling(rolling).corr(df['size'])
    df['rolling_skew_amount'] = df['amount'].rolling(rolling).skew()
    df['rolling_kurt_amount'] = df['amount'].rolling(rolling).kurt()
    df['rolling_median_amount'] = df['amount'].rolling(rolling).median()
    df['rolling_quantile_25_amount'] = df['amount'].rolling(rolling).quantile(.25)
    df['rolling_quantile_50_amount'] = df['amount'].rolling(rolling).quantile(.50)
    df['rolling_quantile_75_amount'] = df['amount'].rolling(rolling).quantile(.75)

    df['ewm_mean_amount'] = pd.DataFrame.ewm(df['amount'], span=rolling).mean()
    df['ewm_std_amount'] = pd.DataFrame.ewm(df['amount'], span=rolling).std()
    df['ewm_var_amount'] = pd.DataFrame.ewm(df['amount'], span=rolling).var()
    df['ewm_corr_amount'] = pd.DataFrame.ewm(df['amount'], span=rolling).corr(df['size'])
    df['ewm_cov_amount'] = pd.DataFrame.ewm(df['amount'], span=rolling).cov()

    # def tendency(price, vol):
    #     df_diff = np.diff(price)
    #     val = (df_diff / price[1:]) * 100
    #     power = np.cumsum(val * vol[1:])
    #     return (power)
    # df['trendency'] = tendency(df['last_price'].values, df['size'].values)

    # Dict for aggregations
    create_feature_dict = {
        # 'log_return': [realized_volatility],
        # 'seconds_in_bucket': [count_unique],
        'size': [np.mean],
        'rolling_mean_size': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_var_size': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_std_size': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_sum_size': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_min_size': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_max_size': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_corr_size': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_kurt_size': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_skew_size': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_median_size': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_mean_size': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_std_size': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_var_size': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_corr_size': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_cov_size': [np.mean,reciprocal_transformation, square_root_translation],
        # 'order_count': [np.sum, np.max],
        'amount': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_mean_amount': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_var_amount': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_std_amount': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_sum_amount': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_min_amount': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_max_amount': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_corr_amount': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_kurt_amount': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_skew_amount': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_median_amount': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_quantile_25_amount': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_quantile_50_amount': [np.mean,reciprocal_transformation, square_root_translation],
        'rolling_quantile_75_amount': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_mean_amount': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_std_amount': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_var_amount': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_corr_amount': [np.mean,reciprocal_transformation, square_root_translation],
        'ewm_cov_amount': [np.mean,reciprocal_transformation, square_root_translation]
    }
    # create_feature_dict_time = {
    #     # 'log_return': [realized_volatility],
    #     # 'seconds_in_bucket': [count_unique],
    #     'size': [np.mean],
    #     'rolling_mean_size': [np.mean],
    #     'rolling_var_size': [np.mean],
    #     'rolling_std_size': [np.mean],
    #     'rolling_sum_size': [np.mean],
    #     'rolling_min_size': [np.mean],
    #     'rolling_max_size': [np.mean],
    #     'rolling_corr_size': [np.mean],
    #     'rolling_kurt_size': [np.mean],
    #     'rolling_skew_size': [np.mean],
    #     'rolling_median_amount': [np.mean],
    #     'rolling_quantile_25_amount': [np.mean],
    #     'rolling_quantile_50_amount': [np.mean],
    #     'rolling_quantile_75_amount': [np.mean],
    #     'ewm_mean_size': [np.mean],
    #     'ewm_std_size': [np.mean],
    #     'ewm_var_size': [np.mean],
    #     'ewm_corr_size': [np.mean],
    #     'ewm_cov_size': [np.mean],
    #     'ewm_mean_amount': [np.mean],
    #     'ewm_std_amount': [np.mean],
    #     'ewm_var_amount': [np.mean],
    #     'ewm_corr_amount': [np.mean],
    #     'ewm_cov_amount': [np.mean]
    #     # 'order_count': [np.sum],
    # }

    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(fe_dict, index, add_suffix=False):
        # Group by the window
        df_feature = df[df['index'] >= index].groupby(['datetime']).agg(fe_dict).reset_index()
        # Rename columns joining suffix
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(index))
        return df_feature
    # df = df.dropna(axis=0, how='any')
    # Get the stats for different windows
    df_feature = get_stats_window(create_feature_dict, index=0, add_suffix=False)
    # df_feature_500 = get_stats_window(create_feature_dict_time, index=25, add_suffix=True)
    # df_feature_400 = get_stats_window(create_feature_dict_time, index=20, add_suffix=True)
    # df_feature_300 = get_stats_window(create_feature_dict_time, index=15, add_suffix=True)
    # df_feature_200 = get_stats_window(create_feature_dict_time, index=10, add_suffix=True)
    # df_feature_100 = get_stats_window(create_feature_dict_time, index=5, add_suffix=True)
    # # Merge all
    # df_feature = df_feature.merge(df_feature_500, how='left', left_on='datetime_', right_on='datetime__25')
    # df_feature = df_feature.merge(df_feature_400, how='left', left_on='datetime_', right_on='datetime__20')
    # df_feature = df_feature.merge(df_feature_300, how='left', left_on='datetime_', right_on='datetime__15')
    # df_feature = df_feature.merge(df_feature_200, how='left', left_on='datetime_', right_on='datetime__10')
    # df_feature = df_feature.merge(df_feature_100, how='left', left_on='datetime_', right_on='datetime__5')
    # print(df_feature)
    # # Drop unnecesary time_ids
    # df_feature.drop(['datetime__25', 'datetime__20', 'datetime__15', 'datetime__10', 'datetime__5'], axis=1,
    #                 inplace=True)

    print(df_feature)

    return df_feature

#%%
# def get_time_stock(df):
#     vol_cols = ['log_return1_realized_volatility', 'log_return2_realized_volatility',
#                 'log_return1_realized_volatility_400', 'log_return2_realized_volatility_400',
#                 'log_return1_realized_volatility_300', 'log_return2_realized_volatility_300',
#                 'log_return1_realized_volatility_200', 'log_return2_realized_volatility_200',
#                 'trade_log_return_realized_volatility', 'trade_log_return_realized_volatility_400',
#                 'trade_log_return_realized_volatility_300', 'trade_log_return_realized_volatility_200']
#
#     # Group by the stock id
#     df_stock_id = df.groupby(['stock_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
#     # Rename columns joining suffix
#     df_stock_id.columns = ['_'.join(col) for col in df_stock_id.columns]
#     df_stock_id = df_stock_id.add_suffix('_' + 'stock')
#
#     # Group by the stock id
#     df_time_id = df.groupby(['time_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', ]).reset_index()
#     # Rename columns joining suffix
#     df_time_id.columns = ['_'.join(col) for col in df_time_id.columns]
#     df_time_id = df_time_id.add_suffix('_' + 'time')
#
#     # Merge with original dataframe
#     df = df.merge(df_stock_id, how='left', left_on=['stock_id'], right_on=['stock_id__stock'])
#     df = df.merge(df_time_id, how='left', left_on=['time_id'], right_on=['time_id__time'])
#     df.drop(['stock_id__stock', 'time_id__time'], axis=1, inplace=True)
#     return df
#%%
train_data = pd.merge(book_preprocessor(data1), trade_preprocessor(data1), on='datetime_',how='left')