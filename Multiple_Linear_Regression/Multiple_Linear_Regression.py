# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:30:05 2019

@author: Abhishek Kamal
"""
# Multiple Linear Regression
 

# 1. Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 2. Import the datasets
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

# 3. Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

# 4. Avoiding Dummy Trap
X = X[:,1:]

# 5. Splitting the dataset into testing and training dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=0)

# 5.Fitting Multiple Linear Regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)



# 6. Building the optimal model using Backward Elimination
import statsmodels.api as sm
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1) 
X_opt = X[:,[0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0, 1, 2, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0, 1, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()






