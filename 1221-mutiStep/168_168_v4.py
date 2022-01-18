#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv


# %%


# load the new file
df = read_csv('./data/bitcoin_data.csv', date_parser = True)
df.drop(df.columns[0], axis=1,inplace = True)
len(df)


# %%


df = df[df['NewDateTime']>= '2021-01-01'].copy()
len(df)


# %%


df.drop(df.head(len(df)%192).index,inplace=True)
len(df)


# %%


int(len(df)/8*7)


# %%


data_training = df.iloc[0:int(len(df)-168),:]
data_test = df.iloc[int(len(df)-168):int(len(df)),:]

len(data_training)


# %%


training_data = data_training.drop(['NewDateTime'], axis = 1)


training_data


# %%


scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)
training_data


# %%


scaler.scale_


# %%


X_train = []
Y_train = []


# %%


training_data.shape


# %%


for i in range(336, training_data.shape[0],8):
    
    X_train.append(training_data[i-336:i-168])
    Y_train.append(training_data[i-168:i,0])


# %%


X_train, Y_train = np.array(X_train), np.array(Y_train)


# %%


def shuffle(X,Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]


# %%


X_train, Y_train = shuffle(X_train,Y_train)


# %%


X_train.shape


# %%


Y_train.shape


# %%


Y_train[824,0]


# %%


Y_train=Y_train.reshape(Y_train.shape[0],168,1)


# %%


# Y_train[1,:,0]


# %%



from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout,RepeatVector,TimeDistributed


# %%



model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 10)))
model.add(Dropout(0.1))
model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.15))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 120, activation = 'relu' ,return_sequences = True))
model.add(Dropout(0.25))

model.add(TimeDistributed(Dense(1)))


# %%


model.summary()


# %%


model.compile(loss='mse', optimizer='adam')


# %%


history = model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.1)


# %%


model.save("168_168_V4.h5")




