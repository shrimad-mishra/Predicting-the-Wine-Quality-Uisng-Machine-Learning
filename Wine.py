# Import the required module

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset

data=pd.read_csv("winequality-red.csv")

# Deciding the x and y

x=data.iloc[:,:-1].values
y=data.iloc[:, -1].values

# spliting the dataset

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=2/4,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

# Fitting into the model

from sklearn.ensemble import RandomForestClassifier 
regressor=RandomForestClassifier(n_estimators=45)
regressor.fit(x_train,y_train)
'''
from sklearn.linear_model import LogisticRegression
regressor=LogisticRegression()
regressor.fit(x_train,y_train)
'''
# prediction time

y_pred=regressor.predict(x_test) 

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))