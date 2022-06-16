import requests as r
import pandas as pd

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)

urlHistData = "https://min-api.cryptocompare.com/data/v2/histoday"
urlPairs = "https://min-api.cryptocompare.com/data/v2/pair/mapping/exchange"

ApiKey = "c3002d0ec8ad7dbb7ab359e3530d32fc2b09e7b94f568f55c7080daf84bdbe2d"

def Download_Pairs_Historical_Data(InPair, url, OutPair = 'USD'):
    payload = {
        "fsym": InPair,
        "tsym": OutPair,
        "allData": True
    }
    
    response = r.post(url, json=payload).json()
    FirstDF = pd.DataFrame.from_dict(response)
    SecondDF = pd.DataFrame(FirstDF['Data'][3])
    return(SecondDF)

def Download_Available_Pairs(url, ApiKey, Exchange = 'Binance'):
    payload = {
        "e": Exchange,
        "api_key": ApiKey
    }
    
    response = r.post(url, json=payload).json()
    FirstDF = pd.DataFrame.from_dict(response)
    SecondDF = pd.DataFrame(FirstDF['Data'][0])
    return(SecondDF)

AvailablePairs = Download_Available_Pairs(urlPairs, ApiKey)
FilteredPair = AvailablePairs[AvailablePairs['tsym'] == 'BTC']['fsym'].tolist()
