import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# FILE_NAME = 'Position_Salaries.csv'
CURRENT_WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = 'DATASET'
CSV_FILE_NAME = 'Position_Salaries.csv'
PATH_DATASET = os.path.join(CURRENT_WORKING_DIR,DATASET_FOLDER,CSV_FILE_NAME)
dataset = pd.read_csv(PATH_DATASET)
print(dataset)
X = dataset.iloc[:,[1]].values
y = dataset.iloc[:,[-1]].values
print(X.shape)
print(y.shape)

# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
#
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
print(lin_reg)
lin_pred_ans = lin_reg.predict(6.5)

print("Result of Lin Pred: ",lin_pred_ans)

##### fitting polynomial regression to dataset ###############
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

##### Visualizing the Linear Regression Results ##################
plt.scatter(X,y,color='red')
y_pred_linear = lin_reg.predict(X)
plt.plot(X,y_pred_linear,color= 'blue')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
# plt.show()

##### Visualizing the Polynomial Regression Results ##################
plt.scatter(X,y,color='red')
X_poly = poly_reg.fit_transform(X)
y_pred_poly = lin_reg_2.predict(X_poly)
plt.plot(X,y_pred_poly,color= 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
# plt.show()

########## Visualizing result with more resolutions ##############
# X_grid = np.arange(min(X),max(X),0.1)
# X_grid = X_grid.reshape((len(X_grid),1))
# plt.scatter(X,y,color='red')
# plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color='blue')
# plt.title('Polynomial Regression')
# plt.xlabel('Position Level')
# plt.ylabel('Salary')
# # plt.show()

######## Predicting a new result with Linear Regression ###############
print("Start predicting !")
lin_pred_ans = lin_reg.predict(6.5)

######## Predicting a new result with Polynomial Regression ###############
# lin_poly_no_trans_ans = lin_reg_2.predict(6.5)
lin_poly_ans = lin_reg_2.predict(poly_reg.fit_transform(6.5))

print("Result of Lin Pred: ",lin_pred_ans)
# print("Result of Lin Poly: ",lin_poly_no_trans_ans)
print("Result of Lin Poly transformed: ",lin_poly_ans)
