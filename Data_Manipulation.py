import Data_Loading.py
#exec(Data_Loading.py)
from datetime import datetime, date, time, timedelta

today = datetime.now()

for key in HistDataList.copy():
    if HistDataList.get(key).iloc[-1]['volumeto'] < 1000000:
        HistDataList.pop(key)

datetime.timestamp(today)

for key in HistDataList.copy():
    if today - HistDataList.get(key).iloc[0].name < timedelta(days = 365):
        HistDataList.pop(key)