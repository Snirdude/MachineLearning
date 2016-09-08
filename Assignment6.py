from matplotlib import pyplot as plt
from load import mnist
import numpy as np

k_NumOfSamples = 10

# TODO:
def randomlyClassifyData():
    import random
    return random.shuffle()

# TODO:
def Kmeans(i_Data, i_K):
    centroids = randomlyClassifyData(i_K)
    costFunctionValues = []

    return centroids, costFunctionValues

# TODO:
def CostFunction():
    return 0

#trX, teX, trY, teY = mnist(ntrain=60000, ntest=10000, onehot=True)
#centroids, costFunctionValues = Kmeans(trX, k_NumOfSamples)
