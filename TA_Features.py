HistDataList = {}
exec(open('Data_Manipulation.py').read())
import talib as ta
import pandas as pd
import statsmodels as sm
from sklearn.model_selection import train_test_split
from statsmodels.regression.linear_model import GLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score, max_error, mean_absolute_percentage_error

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)

def Add_Lagged_CHL(DF, columns, lag):
    #Add lagged values to the data
    for key in columns:
        for i in range(1,lag):
            DF[key + '_lag' + str(i)] = DF[key].shift(i)

def Add_MA(DF, column, MA, periods):
    #Add MA to the data
    for i in periods:
        if MA == 'SMA':
            DF['SMA_' + str(i)] = ta.SMA(DF[column].values, timeperiod=i)
        elif MA == 'EMA':
            DF['EMA_' + str(i)] = ta.EMA(DF[column].values, timeperiod=i)
        elif MA == 'WMA':
            DF['WMA_' + str(i)] = ta.WMA(DF[column].values, timeperiod=i)
        elif MA == 'DEMA':
            DF['DEMA_' + str(i)] = ta.DEMA(DF[column].values, timeperiod=i)
        elif MA == 'KAMA':
            DF['KAMA_' + str(i)] = ta.KAMA(DF[column].values, timeperiod=i)
        elif MA == 'MIDPOINT':
            DF['MIDPOINT_' + str(i)] = ta.MIDPOINT(DF[column].values, timeperiod=i)
        elif MA == 'TEMA':
            DF['TEMA_' + str(i)] = ta.TEMA(DF[column].values, timeperiod=i)
        elif MA == 'TRIMA':
            DF['TRIMA_' + str(i)] = ta.TRIMA(DF[column].values, timeperiod=i)
        else: raise ValueError('Incorrect Moving Average provided')

def Add_RSI(DF, column, periods):
    #Adds RSI indicator
    for i in periods:
        DF['RSI_' + str(i)] = ta.RSI(DF[column].values, timeperiod=i)

def Delete_Unnecessary_Columns_And_Drop_NA(DF, columns):
    #Drops specified columns and drops rows with NAs
    
    Cleaned = DF
    if(any(x in DF.columns for x in columns)):
        DF.drop(columns=columns, inplace=True)
    DF.dropna(inplace=True)
    return(Cleaned)

def Normalize_Data(DF, columns):
    #Normalizes the data by substracting the mean and dividing by the standard deviation
    normalized = DF[columns]
    rest = DF.drop(columns, axis=1)
    normalized = (normalized - normalized.mean()) / normalized.std()
    normalized = pd.concat((normalized, rest), axis=1)
    return normalized

def Train_Linear_Model(X_train, Y_train, VIF_Limit, p):
    for i in range(X.shape[1]-1):
        lm_1 = GLS(Y_train, X_train).fit()
        
        #Delete variables with high VIF or p
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(X_train, i) for i in range(X_train.shape[1])]
        vif["features"] = X_train.columns
        vif.sort_values('VIF Factor', ascending=False, inplace=True, ignore_index=True)
        #Close_lag1 factor is always needed
        if vif['VIF Factor'][0] > VIF_Limit and vif["features"][0] != 'close_lag1':
            del X_train[vif['features'][0]]
        elif vif['VIF Factor'][1] > VIF_Limit:
            del X_train[vif['features'][1]]
        elif lm_1.pvalues.sort_values(ascending=False)[0] > p:
            del X_train[lm_1.pvalues.sort_values(ascending=False).index[0]]
        else: break
    return lm_1

#Specify coin for the analysis
Coin = 'BTC'

#Copy data
Coin_Data = HistDataList[Coin].copy()

#Add TA
Add_Lagged_CHL(Coin_Data, ['close', 'high', 'low', 'volumeto'], 8)
Add_MA(Coin_Data, 'close', 'TEMA', [7, 14, 21, 35, 70])
Add_RSI(Coin_Data, 'close', [7, 14, 21, 35, 70])

#Clean and normalize the data
Dict_Clean = Delete_Unnecessary_Columns_And_Drop_NA(Coin_Data, ['volumefrom', 'volumeto', 'high', 'low', 'open'])
#normalized = Normalize_Data(Dict_Clean, Dict_Clean.columns[Dict_Clean.columns != 'close'])
normalized = Dict_Clean

#Split the data
X = normalized.loc[:, normalized.columns != 'close']
Y = normalized['close']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
X_train = sm.tools.tools.add_constant(X_train, has_constant='add')
X_test = sm.tools.tools.add_constant(X_test, has_constant='add')

#Create models
CoinLinearModel = Train_Linear_Model(X_train, Y_train, 10, 0.05)
CoinLinearModel.summary()

#Evaluate models
Y_test_p = CoinLinearModel.predict(X_test[X_train.columns])
r2_score(Y_test, Y_test_p)
max_error(Y_test, Y_test_p)
mean_absolute_percentage_error(Y_test, Y_test_p)
