import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import math

#dict_data = pd.read_pickle('MeasureDict_00269_02543.pickle')
normal = np.load("correlated_may_normal.npy")
normal_ju = np.load("correlated_normal_ju.npy")
print("data loading===================done")

len = normal.shape[0]
center = np.zeros((len+1,2000))
center_rear = 1
weights = np.zeros(len+1)
#lr = 1.1 #0.0008
center[0] = normal[50000]
proba = []
desired = normal[0]
weights[0] = 1
threshold = -0.9

num_center_iter = []


for i in range(50001,60000):
    phi = np.exp((-1*(((center[0:center_rear] - normal[i])** 2).sum(1) ))/2)
    cur_output = np.dot(weights[0:center_rear], phi)

    proba.append(cur_output)
    if cur_output >= np.exp(threshold):
        weights[center_rear] = (1-cur_output)
        center[center_rear] = normal[i]
        center_rear += 1
    num_center_iter.append(num_center_iter)
        #threshold *= 0.5
    print(i)
    print(cur_output)
    print(center_rear)



plt.figure()
plt.plot(proba)

plt.figure()
plt.plot(num_center_iter)
plt.title("number of centers")
plt.xlabel("number of iterations")



