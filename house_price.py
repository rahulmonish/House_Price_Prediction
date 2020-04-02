#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 01:40:21 2020

@author: rahul
"""

import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.stats



df= pd.read_csv('train.csv')

df= df.loc[ (df['MSSubClass']<60) & (df['MSZoning']<'RL') ,  'Street']

df= df.dropna(how='any', axis=1)
df.isnull().sum()

y= df['SalePrice']
df= df.drop(['SalePrice'], axis=1)

ds= df.corr()
column= df.columns


#Replacing string values with dummies
df_subset = df.select_dtypes(exclude=[np.number])
for i in df_subset.columns:
    cont = pd.crosstab(df_subset[i], y)
    chi= scipy.stats.chi2_contingency(cont)
    print(chi[1])
    if chi[1]<=0.05:
        #print(chi[1])
        df_subset= df_subset.drop(i, axis=1)



df_subset= pd.get_dummies(df_subset)

df= df.select_dtypes(np.number)
df= pd.concat([df, df_subset],axis=1, sort=False)


columns= df.columns
#Checking for correlation
for i in columns:
    for j in columns:
        print((df[i].corr(df[j])))
        if df[i].corr(df[j]) > .4:
            if i!=j and i in df.columns:
                df= df.drop(j, axis=1)

x= df.values

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=.2)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import keras.optimizers




model= Lasso()
model.fit(x_train, y_train)
y_predict= model.score(x_test, y_test)


regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(x_train, y_train)

y_predict= regressor.score(x_test, y_test)



model= xg_reg = xgb.XGBRegressor(subsample= 1.0,
                                 max_depth=2,
                                 min_child_weight= 10,
                                 learning_rate= 0.1,
                                 gamma= 1.5,
                                 colsample_bytree= 1.0)
model.fit(x_train, y_train)
accuracy= model.score(x_test, y_test)


def get_model():
    model= Sequential()
    sgd = keras.optimizers.Adam(lr=0.1)
    model.add(Dense(30,input_dim=133,kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(400,kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(1000,kernel_initializer='he_normal', activation='relu'))

    #model.add(Dropout(.2))
    model.add(Dense(1,activation='linear' ))
    model.compile(loss='mse', optimizer=sgd, metrics=['mse'])
    return model


#model = KerasClassifier(build_fn=get_model)
from sklearn.preprocessing import StandardScaler
encoder= StandardScaler()
x_train= encoder.fit_transform(x_train)
x_test= encoder.fit_transform(x_test)
model= get_model()
model.fit(x_train, y_train, epochs=100)
y_pred= model.predict(x_test)
error= keras.losses.mean_absolute_error(y_test, y_pred)

