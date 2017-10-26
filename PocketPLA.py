
# coding: utf-8

# In[95]:

#
#  Warning: last cell runs for 5-10 mins!
#
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
    
    # plt.scatter(x, y, c = z)
    # plt.ylim(min(y), max(y))
    # plt.xlim(min(x), max(x))
    # 
    # points = np.linspace(min(x),max(x))
    # plt.plot(points, points * a + b, label = "blue")
    # plt.show()
    return x, y, z, a, b


# In[96]:

def E(x,y,z,w):
    misses = 0
    for i in range(0, len(x)):
        if z[i] * (x[i] * w[0] + y[i] * w[1] + w[2]) < 0:
            misses += 1
    return misses/len(x)


# In[97]:

def PLA(X , Y , Z, iterations = 1000):
    x  = X[:100]
    y = Y[:100]
    z = Z[:100]
    it = 0
    w = np.random.rand(3)
    updates = 0
    loss = []
    test_loss = []
    for _ in range(0, iterations): # max number of iterations is 10^5
        count = 0
        for i in range(0, len(x)):
            if z[i] * (x[i] * w[0] + y[i] * w[1] + w[2]) < 0:
                w[0] = w[0] + z[i] * x[i]
                w[1] = w[1] + z[i] * y[i]
                w[2] = w[2] + z[i]
                count += 1
                updates += 1
                loss.append(E(x, y, z, w))
                test_loss.append(E(X[100:], Y[100:], Z[100:], w))
                it += 1
                if it > iterations:
                    return w, loss, test_loss
    return w, loss, test_loss


# In[98]:

def PocketPLA(X, Y, Z, iterations = 1000):
    x  = X[:100]
    y = Y[:100]
    z = Z[:100]
    w = np.random.rand(3)
    loss = 1
    loss_function = []
    test_loss = []
    it = 0
    for _ in range(0, iterations): # max number of iterations is 10^5
        for i in range(0, len(x)):
            if z[i] * (x[i] * w[0] + y[i] * w[1] + w[2]) < 0:
                w0 = w[0] + z[i] * x[i]
                w1 = w[1] + z[i] * y[i]
                w2 = w[2] + z[i]
                newloss = E(x,y,z,[w0,w1,w2])
                it += 1
                if it > iterations:
                    return w, loss_function, test_loss
                if newloss < loss:
                    loss = newloss
                    w = [w0, w1, w2]
                loss_function.append(loss)
                test_loss.append(E(X[100:], Y[100:], Z[100:], w))
    return w, loss_function, test_loss


# In[99]:

def vizusalize(x,y,z,w,a,b):
    plt.scatter(x, y, c = z)
    plt.ylim(min(y), max(y))
    plt.xlim(min(x), max(x))
    
    points = np.linspace(min(x),max(x))
    plt.plot(points, - points * w[0]/w[1] - w[2]/w[1], label = "blue")
    plt.plot(points, points * a + b, label = "red")
    plt.show()
    print("Target function is red line and blue is one learned by perceptron")


# In[104]:

def procedure(n = 1100):
    x, y, z, a, b = gen_data(n)
    w, in_loss, out_loss = PocketPLA(x, y, z)
    w2, in_loss2, out_loss2 = PLA(x, y, z)
    
    print("PocketPLA performace:")
    plt.plot(in_loss, color = "red")
    plt.plot(out_loss, color = "blue")
    plt.xlabel("Number of iterations")
    plt.ylabel("Error")
    plt.show()
    
    print("NaivePLA performace:")
    plt.plot(in_loss2, color = "red")
    plt.plot(out_loss2, color = "blue")
    plt.xlabel("Number of iterations")
    plt.ylabel("Error")
    plt.show()
    
    ave_in_loss = in_loss
    ave_in_loss2 = in_loss2
    for _ in range(19):
        x, y, z, a, b = gen_data(n)
        w, in_loss, out_loss = PocketPLA(x, y, z)
        w2, in_loss2, out_loss2 = PLA(x, y, z)
        for i in range(0, len(ave_in_loss)):
            ave_in_loss[i] += in_loss[i]    
        for i in range(0, len(ave_in_loss2)):
            ave_in_loss2[i] += in_loss2[i]
    for i in range(0, len(ave_in_loss)):
            ave_in_loss[i] /= 20
    for i in range(0, len(ave_in_loss2)):
            ave_in_loss2[i] /= 20
    
    print("Average results for PLA")
    plt.plot(ave_in_loss2, color = "red")
    plt.xlabel("Number of iterations")
    plt.ylabel("Average Error")
    plt.show()
    
    print("Average results for PocketPLA")
    plt.plot(ave_in_loss, color = "red")
    plt.xlabel("Number of iterations")
    plt.ylabel("Average Error")
    plt.show()


# In[105]:

procedure(1100)


# ## Discussion

# As we can see PocketPLA is far stable from NaivePLA, here is the reason: it is just prefix minimum function for NaivePLA. We see but we can see that even PocketPLA cannot prefectly find optimum solution because close to border points and smoothness of updates restricts jumping to best points, so it finds local optima and "converges". Sorry for non-optimal code which runs for :) we could use numpy and python vectorization to do some operations faster, may be next time.
