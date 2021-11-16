import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import *


def run_models(filename):
    df = pd.read_csv(filename)
    #df.head()

    df = df.dropna(axis=0) 

    X = df.drop(columns=['Unnamed: 0', 'year', 'label', 'state', 'state_name', 'cnty_name'], errors='ignore')
    #print(len(X.columns))
    y = df['label']

    X = X.to_numpy()
    y = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    knr = KNeighborsRegressor(n_neighbors=5)
    params_knr = {'n_neighbors': np.arange(1, 25)}
    knr_gs = GridSearchCV(knr, params_knr, cv=5)
    knr_gs.fit(X_train, y_train)

    knr_best = knr_gs.best_estimator_
    print("Best KNR params:")
    print(knr_gs.best_params_)

    rfr = RandomForestRegressor()
    params_rfr = {'n_estimators': [50, 100, 200, 400]}
    rfr_gs = GridSearchCV(rfr, params_rfr, cv=5)
    rfr_gs.fit(X_train, y_train)

    rfr_best = rfr_gs.best_estimator_
    print("Best RFR params:")
    print(rfr_gs.best_params_)

    svr = SVR(C=1.0, epsilon=0.2)
    params_svr = {'C': [0.1, 1.0, 10], 'epsilon':[0.1, 0.2, 0.4]}
    svr_gs = GridSearchCV(svr, params_svr, cv=5)
    svr_gs.fit(X_train, y_train)

    svr_best = svr_gs.best_estimator_
    print("Best SVR params:")
    print(svr_gs.best_params_)

    print('knr: {}'.format(knr_best.score(X_test, y_test)))
    print('rfr: {}'.format(rfr_best.score(X_test, y_test)))
    print('svr: {}'.format(svr_best.score(X_test, y_test)))

    estimators=[('knr', knr_best), ('rfr', rfr_best), ('svr', svr_best)]
    ensemble = VotingRegressor(estimators)

    ensemble.fit(X_train, y_train)
    mean_accuracy = ensemble.score(X_test, y_test)
    print("Mean accuracy of ensemble:", mean_accuracy)

    y_pred = ensemble.predict(X_test)
    print("MAE", mean_absolute_error(y_test, y_pred))
    print("MSE", mean_squared_error(y_test, y_pred))

if __name__ == '__main__':
    #run_models("./data/state_pres.csv")
    #run_models("./data/state_sen.csv")
    run_models("./data/county_pres.csv")
    #run_models("./data/county_sen.csv")