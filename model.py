import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split


df = pd.read_csv('') ### To fill
df.head() 

X = df.drop(columns = ) ### To fill
y = df[] ### To fill

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

knn = KNeighborsClassifier()
params_knn = {'n_neighbors': np.arange(1, 25)}
knn_gs = GridSearchCV(knn, params_knn, cv=5)
knn_gs.fit(X_train, y_train)

knn_best = knn_gs.best_estimator_
print(knn_gs.best_params_)

rf = RandomForestClassifier()
params_rf = {'n_estimators': [50, 100, 200]}
rf_gs = GridSearchCV(rf, params_rf, cv=5)
rf_gs.fit(X_train, y_train)

rf_best = rf_gs.best_estimator_
print(rf_gs.best_params_)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

print('knn: {}'.format(knn_best.score(X_test, y_test)))
print('rf: {}'.format(rf_best.score(X_test, y_test)))
print('log_reg: {}'.format(log_reg.score(X_test, y_test)))

estimators=[('knn', knn_best), ('rf', rf_best), ('log_reg', log_reg)]
ensemble = VotingClassifier(estimators, voting='hard')

ensemble.fit(X_train, y_train)
ensemble.score(X_test, y_test)
