# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 08:10:52 2020

@author: matix
"""

"""
1. Importing the libraries
2. Importing the data-set
3. Encoding categorical data
    i.  Encoding the Independent Variable
    ii. Encoding the Dependent Variable
4. Extracting Features & Labels
5. Taking care of missing data
6. Splitting the data-set into the Training set and Test set
7. Feature Scaling 
8. Training the model on the Training set
9. Predicting the Test set results
10. Deploy Application 
"""

""" 1. Importing the libraries """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""2. Importing the data-set"""
dataset = pd.read_csv('Companies.csv')
temp = dataset.copy()

""" 3. Encoding categorical data
        i.  Encoding the Independent Variable """
city = pd.get_dummies(dataset['State'] , drop_first=True)
dataset = pd.concat([dataset, city],axis=1 )
dataset.drop(['State'], axis=1, inplace=True)

"""4. Extracting Features & Labels"""
X = dataset.drop('Profit', axis=1).values
y = dataset['Profit'].values

"""5. Splitting the data-set into the Training set and Test set"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 1/3, 
                                                    random_state = 0)

""" 6. Feature Scaling """
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 0:3] = sc.fit_transform(X_train[:, 0:3])
X_test[:, 0:3] = sc.transform(X_test[:, 0:3])

"""Training the model on the Training set"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

"""Predicting the Test set results"""
y_pred = regressor.predict(X_test)

