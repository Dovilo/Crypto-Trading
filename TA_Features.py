HistDataList = {}
exec(open('Data_Manipulation.py').read())
import talib as ta
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels as sm
import statsmodels.tsa.api as smt
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.regression.linear_model import GLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)

Coin = 'BTC'

def Add_Lagged_CHL(Dict, coin, columns, lag):
    #Add lagged values to the data
    for key in columns:
        for i in range(1,lag):
            Dict[coin][key + '_lag' + str(i)] = Dict[coin][key].shift(i)

def Add_MA(Dict, coin, column, MA, periods):
    #Add MA to the data
    for i in periods:
        if MA == 'SMA':
            Dict[Coin]['SMA_' + str(i)] = ta.SMA(Dict[Coin][column].values,
                                                 timeperiod=i)
        elif MA == 'EMA':
            Dict[Coin]['EMA_' + str(i)] = ta.EMA(Dict[Coin][column].values,
                                                 timeperiod=i)
        elif MA == 'CMA':
            Dict[Coin]['CMA_' + str(i)] = ta.CMA(Dict[Coin][column].values,
                                                 timeperiod=i)
        elif MA == 'WMA':
            Dict[Coin]['WMA_' + str(i)] = ta.WMA(Dict[Coin][column].values,
                                                 timeperiod=i)
        else: raise ValueError('Incorrect Moving Average provided')

def Add_RSI(Dict, coin, column, periods):
    for i in periods:
        Dict[Coin]['RSI_' + str(i)] = ta.RSI(Dict[Coin][column].values,
                                                 timeperiod=i)

def Delete_Unnecessary_Columns_And_Drop_NA(Dict, coin, columns):
    Dict.get(coin).drop(columns=columns, inplace=True)
    Dict[coin].dropna(inplace=True)

def Normalize_Data(Dict, coin, columns):
    normalized = Dict[coin][columns]
    rest = Dict[coin].drop(columns, axis=1)
    normalized = (normalized - normalized.mean()) / normalized.std()
    normalized = pd.concat((normalized, rest), axis=1)
    return normalized

X = normalized.loc[:, normalized.columns != 'close']
Y = normalized['close']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
X_train = sm.tools.tools.add_constant(X_train, has_constant='add')
X_test = sm.tools.tools.add_constant(X_test, has_constant='add')

for i in range(X.shape[1]-1):
    lm_1 = GLS(Y_train,X_train).fit()
    
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X_train, i) for i in range(X_train.shape[1])]
    vif["features"] = X_train.columns
    vif.sort_values('VIF Factor', ascending=False, inplace=True, ignore_index=True)
    if vif['VIF Factor'][0] > 10:
        del X_train[vif['features'][0]]
    elif lm_1.pvalues.sort_values(ascending=False)[0] > 0.05:
        del X_train[lm_1.pvalues.sort_values(ascending=False).index[0]]
    else: break

lm_1.summary()
Y_test_p = lm_1.predict(X_test[X_train.columns])
plt.figure(figsize=(16,10))
plt.plot(range(0,len(Y_test)),Y_test, label='Actual')
plt.plot(range(0,len(Y_test)),Y_test_p, label='Predicted')
plt.legend(loc='upper left')

r2_score(Y_test, Y_test_p)
error = Y_test - Y_test_p
plt.plot(range(0,len(Y_test)),error)

acf = smt.graphics.plot_acf(lm_1.resid, lags=40 , alpha=0.05)
acf.show()

Add_Lagged_CHL(HistDataList, 'BTC', ['close', 'high', 'low'], 8)
Add_MA(HistDataList, 'BTC', 'close', 'SMA', [6, 12, 18, 24])
Add_RSI(HistDataList, 'BTC', 'close', [6, 12, 18, 24])
Delete_Unnecessary_Columns_And_Drop_NA(HistDataList, 'BTC', ['volumefrom', 'high', 'low', 'open'])
Normalize_Data(HistDataList, 'BTC', HistDataList['BTC'].columns[HistDataList['BTC'].columns != 'close'])
