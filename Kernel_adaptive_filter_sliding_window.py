import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import math

#dict_data = pd.read_pickle('MeasureDict_00269_02543.pickle')
normal_may = np.load("correlated_may_normal.npy")
normal_ju = np.load("correlated_may_abnormal.npy")
len = normal_ju.shape[0]
normal = np.vstack([normal_may[0:90000], normal_ju[90000:len]])
print(normal.shape)
print("data loading===================done")


center = np.zeros((10000,2000))
center_rear = 1
weights = np.zeros(10000)
#lr = 1.1 #0.0008
center[0] = normal[75000]
proba = []
desired = normal[0]
weights[0] = 1
threshold = -0.3



"""train the model(use normal data only)"""
for i in range(75001,85000):
    phi = np.exp((-1*(((center[0:center_rear] - normal[i])** 2).sum(1) ))/2)
    cur_output = np.dot(weights[0:center_rear], phi)

    proba.append(cur_output)
    if cur_output >= np.exp(threshold):
        weights[center_rear] = (1-cur_output)
        center[center_rear] = normal[i]
        center_rear += 1
        #threshold *= 0.5
    print(i)
    print(cur_output)
    print(center_rear)

""" test and update the model"""

time.sleep(10)
for i in range(85000,95000):
    #print("====================test====================")
    print(i)
    phi = np.exp((-1 * (((center[0:center_rear] - normal[i]) ** 2).sum(1))) / 2)
    cur_output = np.dot(weights[0:center_rear], phi)

    proba.append(cur_output)
    if cur_output >= np.exp(threshold):
        weights[:-1] = weights[1:]
        weights[9999] = (1 - cur_output)
        weights[0] = 1
        center[:-1] = center[1:]
        center[9999] = normal[i]
        print("updated")
        # threshold *= 0.5
    print(cur_output)


plt.figure()
plt.plot(proba)




