#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#实时资金流：主力资金流

import MySQLdb
import datetime as dt
import sys,re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from sqlalchemy import create_engine
import os
import time
import math

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


startDate='20210506'
endDate='20211109'
fee=0.0005

#Daily=dt.datetime.now().strftime('%Y-%m-%d')  #当天日期
short_value=0.5
long_value=-0.5


def maxdown_time(data):
    Max_cumret=[]
    retracement=[]
    Re_date=[]
    for i in range(len(data)):
        if i==0:
            Max_cumret.append(data.iloc[0])
            retracement.append((1+Max_cumret[0])/(1+data.iloc[0])-1)
        else:
            #计算策略回撤
            Max_cumret.append(max(Max_cumret[i-1],data.iloc[i]))
            retracement.append(float((1+Max_cumret[i])/(1+data.iloc[i])-1))
            #计算最大回撤时间
        if retracement[i]==0:
            Re_date.append(0)
        else:
            Re_date.append(Re_date[i-1]+1)
    #计算最最大回撤幅度
    retracement=np.nan_to_num(retracement)
    Max_re=max(retracement)
    #计算最大回撤时间
    Max_reDate=max(Re_date) 
    return Max_reDate

def strategy(future):

#      从数据库中查找合约乘数、最小变动价位
    conn1 = MySQLdb.connect(host="rm-bp1yvd1jr36kmmv834o.mysql.rds.aliyuncs.com",user="LJD",passwd="HZLJDcl123456", db="lgt", port=3306, charset='utf8' )
    cursor1 = conn1.cursor()
    sql = "select * from contractsize where future='%s'" % (future)
    cursor1.execute(sql)
    resContractsize = cursor1.fetchall()
    futureTick = resContractsize[0][1]
    futureSize = resContractsize[0][2]
    conn1.close()
    
#  读取本地数据
#     all_data=pd.read_csv('F:\XDQ\RSH\\ni_2min.csv')
    all_data=pd.read_csv('ni_2min.csv')
#     获取交易日
    conn = MySQLdb.connect(host="rm-bp1yvd1jr36kmmv834o.mysql.rds.aliyuncs.com",user="LJD",passwd="HZLJDcl123456", db="datainfo", port=3306, charset='utf8' )
    cursor = conn.cursor()
    sql = "select distinct date from day_price where date>='%s' and date<='%s' order by date ASC "% (startDate,endDate)
    cursor.execute(sql)
    dateTuple = cursor.fetchall()
    
    strategy_data=pd.DataFrame()
    for dateTrade in dateTuple: 
        daily_price=all_data[all_data['date']==str(dateTrade[0])]
        daily_price['time']=daily_price['datetime'].str.extract('\s(.{1,5})')
        daily_price=daily_price.reset_index(drop = True)
        
        position=pd.DataFrame()
        profit=[]
        for k in range(0,len(daily_price)):
            diff_pcnt=(daily_price['close'][k]-daily_price['predictvalue'][k])/daily_price['close'][k]*100
            if daily_price['time'][k]>'09:02' and  diff_pcnt>short_value and daily_price['time'][k]<'14:50' and daily_price['predictvalue'][k]!=0:
                a=pd.DataFrame({'datetime':daily_price['datetime'][k],'pos':[-1]})
                position=position.append(a)
            elif daily_price['time'][k]>'09:02'  and diff_pcnt<long_value and daily_price['time'][k]<'14:50' and daily_price['predictvalue'][k]!=0:
                a=pd.DataFrame({'datetime':daily_price['datetime'][k],'pos':[1]})
                position=position.append(a)
            else:
                if position.empty==True:
                    a=pd.DataFrame({'datetime':daily_price['datetime'][k],'pos':[0]})
                    position=position.append(a)
                else:
                    if position['pos'].iloc[-1]==1 and  diff_pcnt>0 :
                        a=pd.DataFrame({'datetime':daily_price['datetime'][k],'pos':[0]})
                        position=position.append(a)
                    elif position['pos'].iloc[-1]==-1 and  diff_pcnt<0 :
                        a=pd.DataFrame({'datetime':daily_price['datetime'][k],'pos':[0]})
                        position=position.append(a)
                    elif position['pos'].iloc[-1]!=0 and daily_price['time'][k]>='14:55':
                        a=pd.DataFrame({'datetime':daily_price['datetime'][k],'pos':[0]})
                        position=position.append(a)
                    else:
                        a=pd.DataFrame({'datetime':daily_price['datetime'][k],'pos':[position['pos'].iloc[-1]]})
                        position=position.append(a)
            
            if len(position)>=2:
                if position['pos'].iloc[-1]==position['pos'].iloc[-2] :
                    a=(daily_price['close'].astype(float).iloc[k]-daily_price['close'].astype(float).iloc[k-1])*position['pos'].iloc[-1]*futureSize
                    profit.append(a)
                else:
                    if position['pos'].iloc[-1]==0 and position['pos'].iloc[-2]!=0:
                        a=(daily_price['close'].astype(float).iloc[k]-daily_price['close'].astype(float).iloc[k-1])*position['pos'].iloc[-2]*futureSize-daily_price['close'].astype(float).iloc[k]*futureSize*fee
                        profit.append(a)
                    else:
                        a=(daily_price['close'].astype(float).iloc[k]-daily_price['close'].astype(float).iloc[k-1])*position['pos'].iloc[-1]*futureSize-daily_price['close'].astype(float).iloc[k]*futureSize*fee
                        profit.append(a)
                        
        day_sum=sum(profit)
        if strategy_data.empty==False :
            d1=pd.DataFrame({'strategy':"strategy_LSTM_01",'date':dateTrade[0],'margin_long':[0],'margin_short':[0],'equity':strategy_data['equity'][-1:]+day_sum,'cost':[0]})
            strategy_data=strategy_data.append(d1)
        else:
            d1=pd.DataFrame({'strategy':"strategy_LSTM_01",'date':dateTrade[0],'margin_long':[0],'margin_short':[0],'equity':day_sum,'cost':[0]})
            strategy_data=strategy_data.append(d1)

    conn.close()
    
#     strategy_data.set_index('strategy',inplace=True)
#     engine = create_engine('mysql+mysqldb://root:HZLJDcl123456@rm-bp1yvd1jr36kmmv834o.mysql.rds.aliyuncs.com:3306/dynamicmanage_db?charset=utf8',encoding='utf-8')
#     pd.io.sql.to_sql(strategy_data,'strategy_tb2',engine,schema='dynamicmanage_db',if_exists='append')
#     cursor.close()
#     cursor1.close()
    
    #     ===================Report=======================
    #保证金收益率
    strategy_data=strategy_data.reset_index(drop=True)
    strategy_data['sum_profit']=strategy_data['equity']
    Com_Margin=20000
    Com_return=((strategy_data['sum_profit']-strategy_data['sum_profit'].iloc[0])/(Com_Margin))
    comm_accum=(strategy_data['sum_profit']-strategy_data['sum_profit'].iloc[0])
    Com_yet=Com_return.iloc[-1]/len(Com_return)*250
    #最大回撤
    index_j=np.argmax(np.maximum.accumulate(strategy_data['sum_profit'])-strategy_data['sum_profit'])
    index_i=np.argmax(strategy_data['sum_profit'][:index_j])
    maxdrown=(strategy_data['sum_profit'][index_j]-strategy_data['sum_profit'][index_i])
    maxdrownratio=(strategy_data['sum_profit'][index_j]-strategy_data['sum_profit'][index_i])/(Com_Margin)
    Account_retrun=strategy_data['sum_profit']
    log_returns=np.log(Account_retrun.pct_change()+1)
    #最长横盘时间
    Longtime=maxdown_time(strategy_data['sum_profit'])

    truemargin=Com_Margin
    avgargin=Com_Margin

    print("================= %s开始_组合报表 ==============="%(startDate))
    print ("最大保证金:%d"%truemargin)
    print ("平均保证金:%d"%avgargin)
    print ("最大回撤:%s"%int(maxdrown))
    print ("最大保证金回撤率:%s"%format(maxdrownratio,'.2%'))
    print ("年化保证金收益率:%s"%format(Com_yet,'.2%'))
    print ("年化收益风险比:%s"%format(Com_yet/abs(maxdrownratio),'.2f'))
    print ("夏普比率:%s"%format((log_returns.mean()*250)/(log_returns.std()*math.sqrt(250)),'.2f'))
    print ("单日保证金最大亏损率:%s"%format(Account_retrun.pct_change().min(),'.2%'))
    print ("单日最大亏损:%s"%int(strategy_data['sum_profit'].diff().min()))
    print ("组合年化波动率：%s"%format(log_returns.std()*math.sqrt(250),'.2%'))
    print ("最长横盘时间：%s"%Longtime)
    print ("===============================  End  =============================")
    
    plt.figure()
    plt.figure(figsize=(14,8))
    plt.plot(strategy_data.index,strategy_data['equity'])
    plt.title("strategy_LSTM_01")
    plt.show() 
    
    
if __name__ == '__main__':
    startTime = dt.datetime.now()
    print (startTime)
    strategy('ni')
    
    endTime = dt.datetime.now()
    print('Running Time:', (endTime - startTime).seconds, 'seconds')


# In[ ]:




