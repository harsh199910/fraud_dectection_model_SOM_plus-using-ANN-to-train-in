# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 00:41:17 2019

@author: HARSH
"""

#integrating supervised and unsupervised machine learning 

#step one building SOM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Credit_Card_Applications.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

from minisom import MiniSom 

som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

from pylab import bone, pcolor, colorbar, plot, show

bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
mapping = som.win_map(X)
frauds = np.concatenate((mapping[(6, 8)], mapping[(1, 2)]), axis = 0)
frauds = sc.inverse_transform(frauds)  

#building ANN integrating the values.

import keras
from keras.models import Sequential
from keras.layers import Dense

customers = dataset.iloc[:, 1:].values
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1
        
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)
        
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)

# Part 3 - Making predictions and evaluating the model

# Predicting the propability on fraud
y_pred = classifier.predict(customers)        
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)        

y_pred = y_pred[y_pred[:, 2].argsort()]        




















