import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC, SVR
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import *


def run_models_classifier(filename):
    df = pd.read_csv(filename)
    df = df.dropna(axis=0)

    X = df.drop(columns=['Unnamed: 0', 'year', 'label', 'state',
        'state_name', 'cnty_name'], errors='ignore').to_numpy()
    y = df['label'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Use SMOTE to oversample minority class
    print("Before SMOTE:", Counter(y_train))
    over = SMOTE(sampling_strategy=0.75)
    under = RandomUnderSampler(sampling_strategy=1.0)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    X_train, y_train = pipeline.fit_resample(X_train, y_train)
    print("After SMOTE:", Counter(y_train))

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    def print_results(model_name, y_test, y_pred):
        print(model_name, "results:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("F1 score:", f1_score(y_test, y_pred))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Cross entropy loss:", log_loss(y_test, y_pred))

    sample_weights = np.zeros(y_train.shape[0])
    sample_weights[y_train == 0] = 1.0

    minority_weight = [0.5 * i + 1.0 for i in range(9)]
    for w in minority_weight:
        print("W =", w)
        sample_weights[y_train == 1] = w

        knc = AdaBoostClassifier()
        params_knc = {'n_estimators': np.arange(50, 100, 10)}
        knc_gs = GridSearchCV(knc, params_knc, cv=5)
        knc_gs.fit(X_train, y_train, sample_weight=sample_weights)

        knc_best = knc_gs.best_estimator_
        print("Best ADA params:")
        print(knc_gs.best_params_)

        y_pred = knc_best.predict(X_test)
        print_results("ADA", y_test, y_pred)
    
    rfc = RandomForestClassifier(n_jobs=-1, class_weight={0:1,1:5})
    params_rfc = {'n_estimators': [1500]}
    rfc_gs = GridSearchCV(rfc, params_rfc, cv=5)
    rfc_gs.fit(X_train, y_train)

    rfc_best = rfc_gs.best_estimator_
    print("Best RFC params:")
    print(rfc_gs.best_params_)

    y_pred = rfc_best.predict(X_test)
    print_results("RFC", y_test, y_pred)

    svc = BaggingClassifier(n_jobs=-1)
    params_svc = {'n_estimators': np.arange(10, 100, 10)}
    svc_gs = GridSearchCV(svc, params_svc, cv=5)
    svc_gs.fit(X_train, y_train)

    svc_best = svc_gs.best_estimator_
    print("Best BagC params:")
    print(svc_gs.best_params_)

    y_pred = svc_best.predict(X_test)
    print_results("BagC", y_test, y_pred)

    estimators=[('ADA', knc_best), ('rfc', rfc_best), ('BagC', svc_best)]
    ensemble = VotingClassifier(estimators)
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    print_results("Ensemble", y_test, y_pred)

def run_models_regressor(filename):
    df = pd.read_csv(filename)
    #df.head()

    df = df.dropna(axis=0) 

    X = df.drop(columns=['Unnamed: 0', 'year', 'label', 'state', 'state_name', 'cnty_name'], errors='ignore')
    #print(len(X.columns))
    y = df['label']

    X = X.to_numpy()
    y = y.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    knr = KNeighborsRegressor(n_neighbors=5)
    params_knr = {'n_neighbors': np.arange(1, 25)}
    knr_gs = GridSearchCV(knr, params_knr, cv=5)
    knr_gs.fit(X_train, y_train)

    knr_best = knr_gs.best_estimator_
    print("Best KNR params:")
    print(knr_gs.best_params_)

    rfr = RandomForestRegressor()
    params_rfr = {'n_estimators': [1200]}
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
    r2 = ensemble.score(X_test, y_test)
    print("R^2 of ensemble:", r2)

    y_pred = ensemble.predict(X_test)
    print("MAE", mean_absolute_error(y_test, y_pred))
    print("MSE", mean_squared_error(y_test, y_pred))

if __name__ == '__main__':
    max_diff = 1
    #max_diff = 2
    #max_diff = 3
    run_models_classifier(f"./data/county_sen_{max_diff}_pca.csv")
    run_models_classifier(f"./data/county_pres_{max_diff}_pca.csv")
