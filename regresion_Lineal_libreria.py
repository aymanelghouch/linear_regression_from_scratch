import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


training_set = pd.read_csv('dataset_train.csv')

X_train = training_set['YearsExperience'].values.reshape(-1, 1)
y_train = training_set['Salary'].values.reshape(-1, 1)


model = LinearRegression()
model.fit(X_train, y_train)


print(f'b (intercept) = {model.intercept_}')
print(f'w (coef)      = {model.coef_[0]}')


y_pred = model.predict([[5]])

plt.scatter(X_train, y_train)
plt.plot(X_train, model.predict(X_train), color='red')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()