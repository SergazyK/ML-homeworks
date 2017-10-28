
# coding: utf-8

# In[351]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def gen_data(n = 1100):
    a = (np.random.rand() - 0.5)*10
    b = (np.random.rand() - 0.5)*10
    # line will be a*x + b
    x = (np.random.rand(n) - 0.5)*10
    y = (np.random.rand(n) - 0.5)*10
    z = np.ones(n)
    for i in range(0,n):
        if a*x[i] + b < y[i]:
            z[i] = - 1
        if np.random.rand(1) < 0.1:
            z[i] *= -1
    color = ['red' if l == 0 else 'green' for l in z]
    
    plt.scatter(x, y, c = z)
    plt.ylim(min(y), max(y))
    plt.xlim(min(x), max(x))

    points = np.linspace(min(x),max(x))
    plt.plot(points, points * a + b, label = "blue")
    plt.show()
    return np.stack([x,y],axis = 1), np.array(z), a, b

class LogisticRegression():
    def __init__(self, iterations = 1000, learning_rate = 0.01):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None

    def E(X,Y,w):
        return np.sum(np.log1p(np.exp(-np.dot(X, np.array(w)) * Y)))/len(X)


    def gradient(X,Y,w,index):
        return -Y[index] * X[index]/(1 + np.exp(np.dot(X[index],np.array(w)) * Y[index]))


    def fit(self, X, Y):
        self.weights = 

    def SGD(X,Y,lr = 0.001, iterations = 10000):
        w = np.random.rand(3)
        X = np.c_[X,np.ones(len(X))]
        for t in range(iterations):
            w = w - lr * gradient(X, Y, w, np.random.randint(0,len(X)))
        return w

    # In[ ]:

    def predict(X,w):
        X = np.c_[X,np.ones(len(X))]
        return np.sign(np.dot(X,w))

