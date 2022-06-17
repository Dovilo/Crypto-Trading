import requests as req
import pandas as pd

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)

urlHistData = "https://min-api.cryptocompare.com/data/v2/histoday"
urlPairs = "https://min-api.cryptocompare.com/data/v2/pair/mapping/exchange"

ApiKey = "c3002d0ec8ad7dbb7ab359e3530d32fc2b09e7b94f568f55c7080daf84bdbe2d"

def Download_Pairs_Historical_Data(InPair, url, OutPair = 'USD', allData = True):
    payload = {
        "fsym": InPair,
        "tsym": OutPair,
        "allData": allData
    }
    
    response = req.post(url, json=payload).json()
    if response['Response'] == 'Error':
        return 'Error'
    else:
        FirstDF = pd.DataFrame.from_dict(response)
        SecondDF = pd.DataFrame(FirstDF['Data'][3])
        return(SecondDF)

def Download_Available_Pairs(url, ApiKey, Exchange = 'Binance'):
    payload = {
        "e": Exchange,
        "api_key": ApiKey
    }
    
    response = req.post(url, json=payload).json()
    FirstDF = pd.DataFrame.from_dict(response)
    SecondDF = pd.DataFrame(FirstDF['Data'][0])
    return(SecondDF)

AvailablePairs = Download_Available_Pairs(urlPairs, ApiKey)
FilteredPair = AvailablePairs[AvailablePairs['tsym'] == 'USDT']['fsym'].tolist()
pd.DataFrame(FilteredPair).to_csv('Additional Data/_Pairs.csv', header=False, index=False)

HistDataList = {}

for i in FilteredPair:
    Download = Download_Pairs_Historical_Data(i, urlHistData)
    if isinstance(Download, pd.DataFrame):
        Download['time'] = pd.to_datetime(Download['time'], unit='s')
        HistDataList[i] = Download[Download['close'] != 0]
        HistDataList[i].set_index('time', inplace = True)
        HistDataList[i].to_csv('Data/' + i + '.csv')
    else:
        print(i)
        continue
    