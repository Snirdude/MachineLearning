from matplotlib import pyplot as plt
from load import mnist
import numpy as np

K = 10

# TODO:
def RandomlyClassifyData(X):
    C = list(range(10))
    j = 0
    for j in range(K):
        C[j] = {'points': [], 'mean': np.ndarray(784)}

    for i in range(np.size(X, 0)):
        C[j]['points'].append(X[i])
        j += 1
        j %= K

    for j in range(K):
        C[j]['mean'] = np.average(C[j]['points'], axis=0)

    return C


# TODO:
def Kmeans(i_Data, i_K):
    centroids = RandomlyClassifyData(i_K)
    costFunctionValues = []

    return centroids, costFunctionValues

# TODO:
def CostFunction(C):
    pass

trX, teX, trY, teY = mnist(ntrain=60000, ntest=10000, onehot=True)
C = RandomlyClassifyData(trX)
print(C)
#centroids, costFunctionValues = Kmeans(trX, k_NumOfSamples)
