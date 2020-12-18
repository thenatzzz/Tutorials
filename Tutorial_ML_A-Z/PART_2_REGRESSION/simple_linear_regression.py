from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

CURRENT_WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = 'DATASET'
CSV_FILE_NAME = 'Salary_Data.csv'
PATH_DATASET = os.path.join(CURRENT_WORKING_DIR,DATASET_FOLDER,CSV_FILE_NAME)
dataset = pd.read_csv(PATH_DATASET)
print(dataset)
X = dataset.iloc[:,:-1].values
print(X)
y = dataset.iloc[:,1].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3,random_state=0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

############### Fitting simple linear regression to the training set ########
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()  # return object
regressor.fit(X_train,y_train)
print(regressor.fit(X_train,y_train))

################## Predicting the Test Set results #########################
y_pred_test = regressor.predict(X_test)
print("y_test: ", y_test)  # real y value
print("y_pred: ", y_pred_test) # predicted y value

####### Visualizing the Training set results ###############################
plt.scatter(X_train, y_train, color = 'red')
y_pred_train = regressor.predict(X_train)
plt.plot(X_train, y_pred_train, color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

####### Visualizing the Test set results ###############################
plt.scatter(X_test, y_test, color = 'red')
# plt.plot(X_test, y_pred_test, color = 'blue')
plt.plot(X_train, y_pred_train, color = 'blue') # no need to change to x_test since x_train is our model

plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
