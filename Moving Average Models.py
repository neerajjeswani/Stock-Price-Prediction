import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

##Stock
ticker = "FDX"

##Read Data
df = pd.read_csv('/Users/neerajjeswani/Desktop/Quant/Stock/Data/%s.csv'%ticker)
df=df.iloc[:,1:]

##Sort Data by Date
df = df.sort_values('Date')

df.head()

##Using average of high and Low as the price of interest
df['Mid'] = (df['Low']+df['High'])/2.0

Mid = df['Mid'].as_matrix()
len(Mid)

train = Mid[:4250].astype(float)
test = Mid[4250:].astype(float)

##Transforming Data
scaler = MinMaxScaler()
train = train.reshape(-1,1)
test = test.reshape(-1,1)

size = 1000
for i in range(0,4000,size):
    scaler.fit(train[i:i+size,:])
    train[i:i+size,:] = scaler.transform(train[i:i+size,:])

scaler.fit(train[i+size:,:])
train[i+size:,:] = scaler.transform(train[i+size:,:])

train = train.reshape(-1)

test = scaler.transform(test).reshape(-1)

##Smoothing Data
EMA = 0.0
gamma = 0.1
for i in range(4250):
  EMA = gamma*train[i] + (1-gamma)*EMA
  train[i] = EMA

total = np.concatenate([train,test])


##Standard Moving Average
size = 100
N = len(train)
avg = []
avg_x = []
mse = []

for pred_idx in range(size,N):

    if pred_idx >= N:
        date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
    else:
        date = df.loc[pred_idx,'Date']

    avg.append(np.mean(train[pred_idx-size:pred_idx]))
    mse.append((avg[-1]-train[pred_idx])**2)
    avg_x.append(date)

print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse)))

##Plot
plt.figure(figsize = (14,8))
plt.plot(range(df.shape[0]),total,color='b',label='True')
plt.plot(range(size,N),avg,color='orange',label='Prediction')
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show()

##Exponential Moving Average
size = 100
N = len(train)
exp_avg = []
exp_avg_x = []
mse = []

mean = 0.0
exp_avg.append(mean)

decay = 0.5

for pred_idx in range(1,N):

    mean = mean*decay + (1.0-decay)*train[pred_idx-1]
    exp_avg.append(mean)
    mse.append((exp_avg[-1]-train[pred_idx])**2)
    exp_avg_x.append(date)

print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse)))

##Plot
plt.figure(figsize = (14,8))
plt.plot(range(df.shape[0]),total,color='b',label='True')
plt.plot(range(0,N),exp_avg,color='orange',label='Prediction')
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show()