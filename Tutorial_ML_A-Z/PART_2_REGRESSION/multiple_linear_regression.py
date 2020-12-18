from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

CURRENT_WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = 'DATASET'
CSV_FILE_NAME = '50_Startups.csv'
PATH_DATASET = os.path.join(CURRENT_WORKING_DIR,DATASET_FOLDER,CSV_FILE_NAME)
dataset = pd.read_csv(PATH_DATASET)
print(dataset)

X = dataset.iloc[:,:-1].values
# print(X)
y = dataset.iloc[:,-1].values
# print(y)

labelencoder_X = LabelEncoder()
X[:,-1] = labelencoder_X.fit_transform(X[:,-1])
onehotencoder = OneHotEncoder(categorical_features= [-1])
X = onehotencoder.fit_transform(X).toarray()
print(X)

################ Avoiding the dummy variable trap ################
X = X[:,1:]

######### Spliting data ###################################
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=0)
# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)
#################### Multiple Linear Regression to the training set #######
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# print(regressor.fit(X_train, y_train))

################ Predicting the Test set results ##########################
y_pred_test = regressor.predict(X_test)
print("Real y: ", y_test)
print("Predicted y: ", np.round(y_pred_test,2))

############### Building original model using Backward Elimation ###########
import statsmodels.formula.api as sm
#### append new column to old data [append at front]
print("Type of X: ",type(X))
print(X)
X2 = X[:]
# X = np.c_[np.ones(len(X)).astype(int),X]
X = np.c_[np.ones([len(X),1]).astype(int),X]

print(X)
print("Length of X: ", len(X))

X2 = np.append(arr = np.ones((50,1)).astype(int),values=X2,axis=1)
print(X2)
print("Type of X2: ",type(X2))
print("Length of X2: ", len(X2))

print("Is X equal X2? -> {}".format(np.array_equal(X,X2)))
print("Type of X: ",type(X))
print("Shape of X: ",X.shape)

X_opt = X[:,[0,1,2,3,4,5]]
# print(X_opt)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS)
OLS_summary =regressor_OLS.summary()
print(OLS_summary)

### Repeat steps above but removing insignicant independent var ##########
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
OLS_summary =regressor_OLS.summary()
print(OLS_summary)

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
OLS_summary =regressor_OLS.summary()
print(OLS_summary)

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
OLS_summary =regressor_OLS.summary()
print(OLS_summary)

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
OLS_summary =regressor_OLS.summary()
print(OLS_summary)

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
OLS_summary =regressor_OLS.summary()
print(OLS_summary)
