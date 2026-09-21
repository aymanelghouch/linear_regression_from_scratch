import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

training_set = pd.read_csv('dataset_train.csv')

x_train = training_set['YearsExperience'].values
y_train = training_set['Salary'].values


plt.scatter(x_train, y_train)
plt.xlabel('Years of Experience')
plt.ylabel('Salary')



def cost_function(x,y,b,w):
    m = len(x)
    err = 0 

    for i in range(m):
        f = w * x[i] + b
        cost = (f - y[i]) ** 2
        err += cost


    total_cost = err / (2 * m)
    return total_cost


def gradient_function(x,y,b,w):
    m = len(x)
    dc_dw = 0
    dc_db =0

    for i in range(m):
        f = w*x[i] + b
        dc_dw += (f - y[i]) * x[i]
        dc_db += (f - y[i])

    dc_dw = dc_dw / m
    dc_db = dc_db / m

    return dc_db, dc_dw


def gradient_descent(x, y, learning_rate, iterations):
    w = 0
    b = 0

    for i in range(iterations):
        dc_db, dc_dw = gradient_function(x, y, b, w)

        w = w - learning_rate * dc_dw
        b = b - learning_rate * dc_db

        print(f'Iteration {i+1}: Cost {cost_function(x, y, b, w)}')

    return b, w

learning_rate = 0.01

iterations = 1000

final_b, final_w = gradient_descent(x_train, y_train,learning_rate, iterations)

print(f'Final parameters: b = {final_b}, w = {final_w}')


plt.scatter(x_train, y_train)
x_vals = np.array([min(x_train), max(x_train)])
y_vals = final_w * x_vals + final_b
plt.plot(x_vals, y_vals, color='red')
plt.show()