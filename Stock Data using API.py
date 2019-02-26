import pandas as pd
import numpy as np
import datetime as dt
import urllib.request, json
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

##API KEY

key = 'BK3D542YESJ4IB8F'

##Stock
ticker = "FDX"

##API URL
url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,key)

##File Store Location
file_to_save = '/Users/neerajjeswani/Desktop/Quant/Stock/Data/%s.csv'%ticker

with urllib.request.urlopen(url_string) as url:
    data = json.loads(url.read().decode())
    # extract stock market data
    data = data['Time Series (Daily)']
    df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
    for a,b in data.items():
        date = dt.datetime.strptime(a, '%Y-%m-%d')
        data_row = [date.date(),float(b['3. low']),float(b['2. high']),
                    float(b['4. close']),float(b['1. open'])]
        df.loc[-1,:] = data_row
        df.index = df.index + 1      
df.to_csv(file_to_save)

##Sorting data by Date
df = df.sort_values('Date')

df.head()

##Plot
plt.figure(figsize = (14,8))
plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.show()