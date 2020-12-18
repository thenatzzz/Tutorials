import pandas as pd
import quandl
import numpy as np
import math
import time
import datetime

import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from matplotlib import style
style.use('ggplot')

# np.set_printoptions(threshold='nan')
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def modify_column(data, new_col):
    data = data[new_col]
    return data

def calculate_percent_change(data,col_1,col_2):
    return (data[col_1]-data[col_2])/data[col_2] * 100.0

def main():

    df = quandl.get("WIKI/GOOGL",authtoken=API_KEY)
    print(df.head())
    # print_full(df)
    print('\n')

    new_col = ['Adj. Open','Adj. High','Adj. Low', 'Adj. Close', 'Adj. Volume']
    df = modify_column(df, new_col)
    print(df.head())
    # print_full(df.head())
    print('\n')

    ###### Make new Column with percentage change ######################
    df['HL_PCT'] = calculate_percent_change(df,'Adj. High', 'Adj. Low')
    df['PCT_change'] = calculate_percent_change(df,'Adj. Close', 'Adj. Open')
    new_col = ['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']
    df = modify_column(df, new_col)
    print(df.head())
    print('\n')

    ####### Create forcaste col filled with random number ##################
    forecast_col = 'Adj. Close'
    df.fillna(value=-99999, inplace= True)
    # print(df.head())
    print("Length of whole data: ",len(df))
    PERCENTAGE_PREDICTED_DATA = 0.01
    num_forecast_out = int(math.ceil(PERCENTAGE_PREDICTED_DATA * len(df)))
    print("num of forecast_out: ", num_forecast_out)

    df['label'] = df[forecast_col].shift(-num_forecast_out)
    print(df.head())
    print(df)
    ########### drop any still NaN info. from the dataframe################
    df.dropna(inplace=True)
    print(df)
    print(df.head())

    ########### Make input(x) data and label(y) data################
    X = np.array(df.drop(['label'],1))
    print("X: ", X)
    print("Length of X: ", len(X))
    y = np.array(df['label'])
    print("y: ", y)
    print("Length of y: ", len(y))
    print('\n')

    ############# Preprocessing X data to same range for each col for easy calculation #####
    X = preprocessing.scale(X)
    print("X: ", X)

    ########### Creating y label ###################################
    y = np.array(df['label'])
    print("y: ", y)
    print('\n')

    ######### Spliting Data #######################################
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
    print("Length overall data: ", len(X))
    print("X_train length: ", len(X_train))
    print("X_test length: ", len(X_test))
    print("y_train length: ", len(y_train))
    print("y_test length: ", len(y_test))

    #########  Using SVM ###########################################
    svm_start_time = time.time()
    clf = svm.SVR()
    print("clf: ", clf)
    clf.fit(X_train, y_train)
    # print("clf: ", clf)
    confidence = clf.score(X_test,y_test)
    svm_end_time = time.time()
    print("Confidence score: ", confidence)
    print("Total time to run: ", svm_end_time-svm_start_time)
    print('\n')

    ############ Using Linear Regression ################################
    lin_start_time = time.time()
    clf_2 = LinearRegression()
    clf_2.fit(X_train, y_train)
    confidence = clf_2.score(X_test,y_test)
    lin_end_time = time.time()
    print("Confidence score: ", confidence)
    print("Total time to run LinearRegression on single CPU ", lin_end_time-lin_start_time)
    print('\n')

    ############## Using Linear Regression ####################################
    lin_start_time = time.time()
    clf_3 = LinearRegression(n_jobs= -1)
    clf_3.fit(X_train,y_train)
    confidence = clf_3.score(X_test,y_test)
    lin_end_time = time.time()
    print("Confidence score: ", confidence)
    print("Total time to run LinearRegression on Multicore-CPU: ", lin_end_time-lin_start_time)

    ############ Train SVM with different Kernels #########################
    for k in ['linear','poly','rbf','sigmoid']:
        clf = svm.SVR(kernel=k)
        clf.fit(X_train,y_train)
        confidence = clf.score(X_test,y_test)
        print("Kernel '{}' of SVM has confidence score = {}".format(k,confidence))

    ################ Splitting up data ###################################
    X = np.array(df.drop(['label'],1))
    print("X: ",X)
    print("length of X: ", len(X))
    X = preprocessing.scale(X)
    print("X after preprocessing : ",X)

    X_lately = X[-num_forecast_out:]
    print("X_lately: ", X_lately)
    print("length of X_lately: ", len(X_lately))
    X = X[:-num_forecast_out]
    # print("X: ",X)
    print("length of X: ",len(X))

    df.dropna(inplace=True)

    y = np.array(df['label'])
    y = y[:-num_forecast_out]
    print("y: ", y)
    print("length of y: ", len(y))
    print('\n')

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train,y_train)
    confidence = clf.score(X_test, y_test)
    print("Confidence score : ",confidence)

    ##################### Predict only the data of num_forecast_out ###########
    forecast_set = clf.predict(X_lately)
    print("forecast_set length = {} :::: num_forecast_out = {}".format(len(forecast_set),num_forecast_out))
    print(forecast_set)

    ################## Adding more column and fill in with NaN value ######
    df['Forecast'] = np.nan
    # print(df['Forecast'])

    ################# Getting next day for forecast in unix format #########################
    last_date = df.iloc[-1].name
    print("last_date: ", last_date)
    last_unix = last_date.timestamp()
    print("last_date as unix_format: ", last_unix)
    one_day = 86400
    next_unix = last_unix + one_day
    print("next day as unix_format: ", next_unix)

    print(df.iloc[-1])
    print('\n')

    ############### Setting forecast column #############################
    print("Length of forecast_set: ", len(forecast_set))
    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        print("next_date: ", next_date)
        next_unix += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
        print(df.loc[next_date])

    #####  Plotting ##############################################
    df['Adj. Close'].plot()

if __name__ == '__main__':
    main()
