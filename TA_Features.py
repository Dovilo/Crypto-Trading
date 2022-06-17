HistDataList = {}
exec(open('Data_Manipulation.py').read())
import talib as ta
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels as sm
import statsmodels.tsa.api as smt
from sklearn.model_selection import train_test_split
from statsmodels.regression.linear_model import GLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score

Coin = 'AAVE'

for i in range(1,8):
    HistDataList[Coin]['close_lag' + str(i)] = HistDataList[Coin]['close'].shift(i)

for i in [6,12,18,24]:
    HistDataList[Coin]['ma_' + str(i)] = ta.SMA(HistDataList[Coin]['close'].values,
                                                 timeperiod=i) / HistDataList[Coin]['close']
    
    HistDataList[Coin]['rsi_' + str(i)] = ta.RSI(HistDataList[Coin]['close'].values,
                                                 timeperiod=i)

HistDataList.get(Coin).drop(columns=['high', 'low', 'open', 'volumefrom'], inplace=True)
HistDataList[Coin].dropna(inplace=True)

normalized = HistDataList[Coin].loc[:, HistDataList[Coin].columns != 'close']
normalized = (normalized - normalized.mean()) / normalized.std()
normalized = pd.concat((normalized, HistDataList[Coin].loc[:,'close']), axis=1)

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


