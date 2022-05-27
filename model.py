import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
df1 = load_boston()
df1
df2 = pd.DataFrame(df1['data'], columns=df1['feature_names'])
df2
df3 = pd.DataFrame(df1['target'],columns=['Target'])
df3
from sklearn.model_selection import train_test_split
X = df2.values
y = df3.values
X_train,X_test,y_train,y_test =train_test_split(X,y , test_size=0.20, random_state=42,shuffle=True)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
lin_pred_test = lin_reg.predict(X_test)
lin_pred_train = lin_reg.predict(X_train)
r2_test = r2_score(y_test,lin_pred_test)
r2_train = r2_score(y_train,lin_pred_train)
print('R squared of Linear Regression for Test Date :', r2_test)
print('R squared of Linear Regression for Train Date :', r2_train)
import pickle
filename = 'boston_house.pkl'
pickle.dump(lin_reg, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
