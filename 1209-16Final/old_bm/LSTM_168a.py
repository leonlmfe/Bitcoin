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
df.head()


# %%


data_training = df[df['NewDateTime']>= '2021-01-01'].copy()
data_training = data_training[data_training['NewDateTime']< '2021-10-08'].copy()


# %%


data_test = df[df['NewDateTime']>= '2021-10-08'].copy()
data_test


# %%


training_data = data_training.drop(['NewDateTime'], axis = 1)
training_data.head()


# %%


scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)
training_data


# %%


X_train = []
Y_train = []


# %%


training_data.shape[0]


# %%


for i in range(840, training_data.shape[0]):
    X_train.append(training_data[i-840:i-168])
    Y_train.append(training_data[i-168:i])


# %%


X_train, Y_train = np.array(X_train), np.array(Y_train)


# %%


X_train.shape


# %%


Y_train.shape


# %%


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout,RepeatVector,TimeDistributed


# %%
regressor = Sequential()
# regressor.add(LSTM(units = 10, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 10)))
# regressor.add(Dropout(0.1))
# regressor.add(LSTM(units = 168, activation = 'relu'))
# regressor.add(Dropout(0.1))

# regressor.add(Dense(units =168))

regressor.add(LSTM(168, activation='relu', input_shape = (X_train.shape[1], 10)))
# regressor.add(RepeatVector(168))
# regressor.add(LSTM(200, activation='relu', return_sequences=True))
# regressor.add(TimeDistributed(Dense(100, activation='relu')))
# regressor.add(TimeDistributed(Dense(1, activation='relu')))
regressor.add(Dense(168))

# %%


regressor.summary()


# %%


regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# %%


history = regressor.fit(X_train, Y_train, epochs = 20, batch_size =128)


# %%


regressor.save("LSTM_168_1216.h5")






