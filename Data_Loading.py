import requests as req
import pandas as pd

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)

#Urls used for API requests
urlHistData = "https://min-api.cryptocompare.com/data/v2/histoday"
urlPairs = "https://min-api.cryptocompare.com/data/v2/pair/mapping/exchange"

ApiKey = "c3002d0ec8ad7dbb7ab359e3530d32fc2b09e7b94f568f55c7080daf84bdbe2d"

def Download_Pairs_Historical_Data(InPair, url, OutPair = 'USD', allData = True):
    #A function to download historical data for cryptocurrencies from the API
    payload = {
        "fsym": InPair,
        "tsym": OutPair,
        "allData": allData
    }
    
    response = req.post(url, json=payload).json()
    
    #If for some reason the API sends an error back
    if response['Response'] == 'Error':
        return 'Error'
    else:
        FirstDF = pd.DataFrame.from_dict(response)
        SecondDF = pd.DataFrame(FirstDF['Data'][3])
        return(SecondDF)

def Download_Available_Pairs(url, ApiKey, Exchange = 'Binance'):
    #A function to download all available crypto pairs on the provided exchange
    #Seems to download more symbols than there is data for to download
    payload = {
        "e": Exchange,
        "api_key": ApiKey
    }
    
    response = req.post(url, json=payload).json()
    FirstDF = pd.DataFrame.from_dict(response)
    SecondDF = pd.DataFrame(FirstDF['Data'][0])
    return(SecondDF)

#Download available pairs and save them as csv for further use
AvailablePairs = Download_Available_Pairs(urlPairs, ApiKey)
FilteredPair = AvailablePairs[AvailablePairs['tsym'] == 'USDT']['fsym'].tolist()
pd.DataFrame(FilteredPair).to_csv('Additional Data/_Pairs.csv', header=False, index=False)

#Assign data frames with data to the dictionary with symbols as keys
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
    