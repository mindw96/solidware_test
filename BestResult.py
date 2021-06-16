import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *


df = pd.read_csv('data/winequality-red.csv', sep=';')

q1=df.quantile(0.10)
q2=df.quantile(0.90)
qua =q2-q1

df=df[~((df<(q1-(1.9*qua)))|(df>(q2+(1.9*qua)))).any(axis=1)]

y = df['quality']
X = df.drop(labels=['quality'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=96)

standard_scaler = StandardScaler()
X_train_SS = standard_scaler.fit_transform(X_train)
X_test_SS = standard_scaler.transform(X_test)

xgb_model = xgb.XGBRegressor(booster='gbtree',learning_rate=0.005, subsample=0.75, colsample_bytree=1, max_depth=10, 
            eval_metric='rmse', scale_pos_weight=6, objective='reg:squarederror', base_score=5.5, n_estimators=1000)
xgb_model.fit(X_train_SS, y_train)

predictions = xgb_model.predict(X_test_SS)
mse = np.sum((y_test - predictions)**2) / len(predictions)
rmse = np.sqrt(mse)
mae = np.sum(np.abs((y_test - predictions)/y_test))/len(predictions)
mape = mae * 100

print('MSE : {0:0.4f}\nRMSE : {1:0.4f}\nMAE : {2:0.4f}\nMAPE : {3:0.4f}'.format(mse, rmse, mae, mape))
