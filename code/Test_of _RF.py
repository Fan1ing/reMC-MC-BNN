import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn import metrics
np.set_printoptions(linewidth=3)
dataset = pd.read_csv("C:/Users/Arthas/Desktop/train.csv",header=None)
dataset2 = pd.read_csv("C:/Users/Arthas/Desktop/test.csv",header=None)#test
X_train = dataset.iloc[:, :43].values
y_train = dataset.iloc[:, 43:].values
X_test = dataset2.iloc[:, :43].values
y_test = dataset2.iloc[:, 43:].values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
feat_labels = dataset.columns[0:]
regressor = RandomForestRegressor(n_estimators=1000,random_state=1)
regressor.fit(X_train, y_train)
y_train_pred = regressor.predict(X_train)
y_pred = regressor.predict(X_test)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
print(y_test)
print(y_pred)
from sklearn import metrics
print('GBR：TEST:MAE:', mae)
