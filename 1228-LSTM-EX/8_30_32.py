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


# In[15]:


df = df[df['NewDateTime']>= '2021-03-01'].copy()
len(df)


# In[16]:




data_training = df.iloc[0:int(len(df)-168),:]
data_test = df.iloc[int(len(df)-168):int(len(df)),:]

len(data_training)


# In[19]:


training_data = data_training.drop(['NewDateTime'], axis = 1)


training_data


# In[20]:


scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)
training_data


# In[21]:


scaler.inverse_transform(training_data)


# In[22]:


X_train = []
Y_train = []


# In[23]:


training_data.shape


# In[24]:


for i in range(336, training_data.shape[0],8):
    
    X_train.append(training_data[i-336:i-168])
    Y_train.append(training_data[i-168:i,0])


# In[25]:


X_train, Y_train = np.array(X_train), np.array(Y_train)



# In[29]:


X_train.shape


# In[30]:


Y_train.shape


# In[32]:


Y_train=Y_train.reshape(Y_train.shape[0],168,1)


# In[34]:



from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout,RepeatVector,TimeDistributed


# In[35]:



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


# In[36]:


model.summary()


# In[37]:


model.compile(loss='mse', optimizer='adam')


# In[ ]:


history = model.fit(X_train, Y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.1)


# In[33]:





model.save("168_168_V5.h5")







