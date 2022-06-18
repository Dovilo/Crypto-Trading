#exec(open('Data_Loading.py').read())
from datetime import datetime, timedelta
import pandas as pd
import os

#Set last analysis date
today = '2022-06-17'
today = datetime.strptime(today, '%Y-%m-%d')

#Recreate data downloaded in the Data_Loading.py.
#If no data available or stale, exec that file first
HistDataList = {}
FilteredPair = pd.read_csv('Additional Data/_Pairs.csv', header=None).values.tolist()
for i in range(len(os.listdir('Data'))):
    HistDataList[os.listdir('Data')[i].split('.')[0]] = pd.read_csv('Data/' + os.listdir('Data')[i], index_col='time', parse_dates=True)

#Remove crypto with small present volume
for key in HistDataList.copy():
    if HistDataList.get(key).iloc[-1]['volumeto'] < 1000000:
        HistDataList.pop(key)

#Remove crypto with little historical data
for key in HistDataList.copy():
    if today - HistDataList.get(key).iloc[0].name < timedelta(days = 365):
        HistDataList.pop(key)

#Drop conversion data
for key in HistDataList.copy():
    HistDataList.get(key).drop(columns=['conversionType', 'conversionSymbol'], inplace=True)
