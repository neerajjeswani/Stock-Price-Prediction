import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

##Stock
ticker = "FDX"

##Read Data
df = pd.read_csv('/Users/neerajjeswani/Desktop/Quant/Stock/Data/%s.csv'%ticker)
df=df.iloc[:,1:]

##Sort Data by Date
df = df.sort_values('Date')

df.head()

df['Mid'] = (df['Low']+df['High'])/2.0

mid = df['Mid'].as_matrix()
len(mid)

##Transform Data
scaler = MinMaxScaler()
mid=mid.reshape(-1,1)
mid = scaler.fit_transform(mid)

X,Y =[],[]
time = 7
for i in range(len(mid)-time-1):
    X.append(mid[i:(i+time),0])
    Y.append(mid[(i+time),0])

X, Y = np.array(X), np.array(Y)

##Splitting into Train & Test
X_train,X_test = X[:int(len(X)*0.80)],X[int(len(X)*0.80):]
Y_train,Y_test = Y[:int(len(Y)*0.80)],Y[int(len(Y)*0.80):]

#Build the model
model = Sequential()
model.add(LSTM(256,input_shape=(7,1)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
#Reshape data for (Sample,Timestep,Features) 
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
#Fit model with history to check for overfitting
history = model.fit(X_train,Y_train,epochs=300,validation_data=(X_test,Y_test),shuffle=False)

Xt = model.predict(X_test)
plt.plot(scaler.inverse_transform(Y_test.reshape(-1,1)))
plt.plot(scaler.inverse_transform(Xt))


