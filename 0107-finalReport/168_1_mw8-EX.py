#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv


# In[2]:


# load the new file
df = read_csv('./data/bitcoin_data.csv', date_parser = True)
df.drop(df.columns[0], axis=1,inplace = True)
len(df)


# In[3]:


df = df[df['NewDateTime']>= '2021-03-01'].copy()
len(df)


# In[4]:


# df.drop(df.head(len(df)%168).index,inplace=True)
# len(df)


# In[5]:


viewdata = 168*3
viewdata


# In[6]:


df = df.drop_duplicates(subset=['NewDateTime'], keep="last")
data_training = df.iloc[0:int(len(df)-viewdata),:]
data_test = df.iloc[int(len(df)-viewdata):int(len(df)),:]

data_training


# In[7]:


training_data = data_training.drop(['NewDateTime'], axis = 1)


training_data


# In[8]:


scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)
training_data


# In[9]:


X_train = []
Y_train = []


# In[10]:


training_data.shape


# In[11]:


for i in range(168, training_data.shape[0],8):
    X_train.append(training_data[i-168:i])
    Y_train.append(training_data[i])


# In[12]:


X_train, Y_train = np.array(X_train), np.array(Y_train)


# In[13]:


X_train.shape


# In[14]:


Y_train.shape


# In[15]:



from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout,RepeatVector,TimeDistributed


# In[16]:


model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 10)))
model.add(Dropout(0.1))
model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.15))
model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.25))

model.add(Dense(units =10))


# In[17]:


model.summary()


# In[18]:


model.compile(loss='mse', optimizer='adam')


# In[ ]:


history = model.fit(X_train, Y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.1)




model.save("168_1_mw8.h5")


