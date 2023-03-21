# Insurance-Claims-Regression

# importing libs and Reading the dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

df = pd.read_csv("Downloads\\insurance_claims.txt")
df.head()

df.info()

#remove rows with missing values
df=df.dropna()

df['marital'].value_counts()

df['claim_type'].value_counts()

df['gender'].value_counts()

#Data cleaning and preprocessing

df["gender_code"] = df["gender"].apply(lambda x: 1 if x == "Male" else 0)
df["marital_code"] = df["marital"].apply(lambda x: 1 if x == "Unmarried" else 0)
df.head()

#Creating 4 dummy columns for claim_type

df["dummy_1"] = df["claim_type"].apply(lambda x: 1 if x == "Contamination" else 0)
df["dummy_2"] = df["claim_type"].apply(lambda x: 1 if x == "Wind" else 0)
df["dummy_3"] = df["claim_type"].apply(lambda x: 1 if x == "Fire" else 0)
df["dummy_4"] = df["claim_type"].apply(lambda x: 1 if x == "Water damage" else 0)
df.head()

x = df.drop(['claim_amount'], axis=1)
y=df['claim_amount']
print(x)
print(y)

#splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                              test_size=0.3)

from sklearn.linear_model import LinearRegression

#fitting the multiple regression model
mlr=LinearRegression()
mlr.fit(x_train,y_train)

#prediction of test set 
y_pred_mlr=mlr.predict(x_test)

print(y_pred_mlr)

# model evaluation

from sklearn import metrics
meanAE = metrics.mean_absolute_error(y_test,y_pred_mlr)
meanSE = metrics.mean_squared_error(y_test,y_pred_mlr)
RMSE = np.sqrt(metrics.mean_squared_error(y_test,y_pred_mlr))
print('R squared: ',(mlr.score(x,y)*100))
print('Mean absolute Error:',meanAE)
print('Mean square Error:',meanSE)
print('Root Mean Square Error:',RMSE)

# Applying Decision Tree Model

from sklearn.tree import DecisionTreeRegressor

dtr=DecisionTreeRegressor()
dtr.fit(x_train,y_train)

y_pred_dtr=dtr.predict(x_test)

print(y_pred_dtr)

# model evaluation

meanAE_dtr = metrics.mean_absolute_error(y_test,y_pred_dtr)
meanSE_dtr = metrics.mean_squared_error(y_test,y_pred_dtr)
RMSE_dtr = np.sqrt(metrics.mean_squared_error(y_test,y_pred_dtr))
print('R squared: ',(dtr.score(x,y)*100))
print('Mean absolute Error:',meanAE)
print('Mean square Error:',meanSE)
print('Root Mean Square Error:',RMSE)

# Applying RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor()
rfr.fit(x_train,y_train)

y_pred_rfr=rfr.predict(x_test)

print(y_pred_rfr)

# model evaluation

meanAE_rfr = metrics.mean_absolute_error(y_test,y_pred_rfr)

meanSE_rfr = metrics.mean_squared_error(y_test,y_pred_rfr)

RMSE_rfr = np.sqrt(metrics.mean_squared_error(y_test,y_pred_rfr))

print('R squared: ',(rfr.score(x,y)*100))

print('Mean absolute Error:',meanAE)

print('Mean square Error:',meanSE)

print('Root Mean Square Error:',RMSE)
