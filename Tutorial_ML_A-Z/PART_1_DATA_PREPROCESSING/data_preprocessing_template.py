import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
##############   LOADING & ASSIGNING DEPENDENT + INDEPENDENT VARIABLE #####
CURRENT_WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = 'DATASET'
CSV_FILE_NAME = 'Data.csv'

path_dataset = os.path.join(CURRENT_WORKING_DIR,DATASET_FOLDER,CSV_FILE_NAME)
# print(path_dataset)
dataset = pd.read_csv(path_dataset)
# print(dataset)
# Independent variable: Country, Age, Salary
# Dependent variable: Purchased
# Predict whether Purchased or not

X = dataset.iloc[:,:-1].values # note: 1st index = row, 2nd index = col
                            # note2: .values = to get all value into line array
print(X)
y = dataset.iloc[:,-1].values
print(y)

############# DEALING WITH MISSING VALUES ###################################
# replace missing data with MEAN
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy= 'mean', axis = 0) # axis = 0 -> along col
imputer = imputer.fit(X[:,1:3]) # replace just missing value in col 2, 3 (index = 1,2)
X[:,1:3] = imputer.transform(X[:,1:3]) # function transfor for assigning correct value back to X
print(X)

############# ENCODING CATEGORICAL DATA ################################
# CATEGORICAL var = var that is not number ie Country & Purchased
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X = LabelEncoder()
label_encoder_X = label_encoder_X.fit_transform(X[:,0])
X[:,0] = label_encoder_X
# print(label_encoder_X)
print(X)

#create dummy variable: varible that does not care rank of elements
onehotencoder = OneHotEncoder(categorical_features = [0]) # 0 = index of country
X = onehotencoder.fit_transform(X).toarray()
print(X)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(y)

########## SPLITING DATASET: training set & test set #############################################
from sklearn.cross_validation import train_test_split

# random_state may not need (just for checking result)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=0)
print(X_train)
print("X length: ", len(X))
print("X_train length: ",len(X_train))
print("X_test length: ",len(X_test))
print("y_train length: ",len(y_train))
print("y_test length: ",len(y_test))


############## FEATURE SCALING ###########################
# problem: range of age and salary are in different scale
# ml relies on euclidean distance
# fix: scale -1 to 1 value
# standardisation & normalization

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
print("X_train before fit_transform: ",X_train)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # do not need fit anymore since training set already fit
# fit vs fit_transform ref: https://datascience.stackexchange.com/questions/12321/difference-between-fit-and-fit-transform-in-scikit-learn-models
print("X_train after fit_transform: ",X_train)
