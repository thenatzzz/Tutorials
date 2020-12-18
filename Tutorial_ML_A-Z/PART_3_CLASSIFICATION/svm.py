import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

CURRENT_WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = 'DATASET'
CSV_FILE_NAME = 'Position_Salaries.csv'
PATH_DATASET = os.path.join(CURRENT_WORKING_DIR,DATASET_FOLDER,CSV_FILE_NAME)
dataset = pd.read_csv(PATH_DATASET)
print(dataset)
# X = dataset.iloc[:,1:2].values
X = dataset.iloc[:,[1]].values
y = dataset.iloc[:,[-1]].values
print("X: ",X)
print("y: ",y)

'''
############ WITHOUT FEATURES SCALING ######################
######### fitiing SVR to dataset ###################
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

########## predicting a new result ###############
y_pred = regressor.predict(6.5)

########## Visualizing the SVR results #####################
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title("SVR")
plt.xlabel('Position level')
plt.ylabel('Salaries')
plt.show()
###########################################################
'''

####### SVR does not provide feature scaling #########
###### so we need to do ourself #########
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
# X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

######### fitiing SVR to dataset ###################
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

########## predicting a new result ###############
y_pred = regressor.predict(6.5) ############ CAREFUL we dont transform 6.5  yet
# y_pred = regressor.predict(sc_X.transform(6.5))
y_pred = regressor.predict(sc_X.transform(np.array([6.5]).reshape(-1,1)))
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
print("y_pred after rescling: ", y_pred)
###### but we want numerical original value ########
y_pred = sc_y.inverse_transform(y_pred)
print("y_pred after inverse transforming back: ", y_pred)

########## Visualizing the SVR results #####################
# plt.scatter(X,y,color='red')
# plt.plot(X,regressor.predict(X),color='blue')
# plt.title("SVR")
# plt.xlabel('Position level')
# plt.ylabel('Salaries')
# plt.show()

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('SVR')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
