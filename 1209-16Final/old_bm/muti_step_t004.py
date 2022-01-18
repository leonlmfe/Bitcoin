#!/usr/bin/env python
# coding: utf-8
# %%

# %%


# univariate multi-step lstm
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM, Dropout ,RepeatVector,TimeDistributed
from sklearn.preprocessing import MinMaxScaler


# %%


# load the new file
dataset = read_csv('./data/bitcoin_data.csv', date_parser = True)
dataset.drop(dataset.columns[0], axis=1,inplace = True)
dataset.head()


# %%


# split into train and test
PDate = '2021-10-08'
train = dataset[dataset['NewDateTime']>= '2021-01-01'].copy()
train = train[train['NewDateTime']< PDate].copy()
test = dataset[dataset['NewDateTime']>= PDate].copy()
# restructure into windows of weekly data

train.drop(train.head(len(train)%168).index,inplace=True)
train = train.drop(['NewDateTime'], axis=1)
test.drop(test.tail(len(test)%168).index,inplace=True)
test = test.drop(['NewDateTime'], axis=1)

scaler = MinMaxScaler()
train = scaler.fit_transform(train)
test = scaler.fit_transform(test)
train


# %%


train = array(split(train, len(train)/24))
# print(len(test))

test = array(split(test, len(test)/24))


# %%


# evaluate model and get scores
n_input = 168


# %%



# history is a list of weekly data
# prepare data
# train_x, train_y = to_supervised(train, n_input)
n_out=24
# flatten data
data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
X, y = list(), list()
in_start = 0
# step over the entire history one time step at a time
for _ in range(len(data)):
	# define the end of the input sequence
	in_end = in_start + n_input
	out_end = in_end + n_out
	# ensure we have enough data for this instance
	if out_end < len(data):
		X.append(data[in_start:in_end, :])
		y.append(data[in_end:out_end, 0])
	# move along one time step
	in_start += 1
train_x ,train_y =array(X), array(y)


# %%


train_x.shape


# %%


train_y.shape


# %%


# define parameters
verbose, epochs, batch_size = 1, 20, 32
n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
# reshape output into [samples, timesteps, features]
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
# define model
# define model
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))

model.add(RepeatVector(n_outputs))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(Dropout(0.1))
model.add(TimeDistributed(Dense(100, activation='relu')))
model.add(TimeDistributed(Dense(1)))

model.summary()


# %%


model.compile(loss='mse', optimizer='adam')
# fit network
model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size)


# %%


model.save("ms_model_t004.h5")

