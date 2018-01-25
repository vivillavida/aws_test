
# coding: utf-8

# In[6]:


import numpy as np
import csv
import os

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

filename = '/Users/Xinyue/Documents/learn_py/cs231/deeplearning/voice.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)

x = list(reader)
voice = np.array(x)
print voice.shape


voice1 = voice[1:, :]
print voice1.shape



X_all = voice1[:,:20]
Y_all = voice1[:,20]
Y_all = (Y_all == '"male"')
Y_all = Y_all.astype(int)

x_train, x_test = X_all[:2500,:], X_all[2500:,:]
np.random.shuffle(Y_all)
y_train, y_test = Y_all[:2500], Y_all[2500:]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train -= np.mean(x_train, axis=0)
x_test -= np.mean(x_test, axis=0)

print x_train.shape[0], 'train samples'
print x_test.shape[0], 'test samples'

y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)


model = Sequential()
#model.add(Dense(1000, activation='relu', input_shape=(20,)))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(2, activation='softmax'))


model.add(Dense(35,activation = "tanh", input_shape = (20,)))
model.add(Dense(20,activation = "tanh"))
model.add(Dropout(0.4))
model.add(Dense(7,activation = "sigmoid"))
model.add(Dense(2,activation = "sigmoid"))
model.summary()


model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size = 15,
                    epochs=20,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print 'Test loss:', score[0]
print 'Test accuracy:', score[1]

