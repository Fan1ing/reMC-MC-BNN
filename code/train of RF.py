import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn import metrics
dataset = pd.read_csv("C:/Users/Arthas/Desktop/train.csv",header=None)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
ym1 = 0
ym11 = 0
i = 0
for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    random_state=i)

    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    from sklearn.ensemble import RandomForestRegressor
    feat_labels = dataset.columns[0:]
    regressor = RandomForestRegressor(n_estimators=1000)
    regressor.fit(X_train, y_train)
    y_train_pred = regressor.predict(X_train)
    y_pred = regressor.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    if i==0:
        ym1 = mae
        ym11 = mse
    else:
        ym1 = ym1 + mae
        ym11 = ym11+mse
ym1 = ym1/(i+1)
ym11 = ym11/(i+1)
print('RF：TEST:MAE:', ym1)
print('RF：TEST:MSE:', ym11)