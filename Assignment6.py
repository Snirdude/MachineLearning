from matplotlib import pyplot as plt
from load import mnist
import numpy as np
from scipy.spatial import distance

K = 10

# TODO:
def RandomlyClassifyData(X):
    C = list(range(10))

    for j in range(K):
        C[j] = {'points': [], 'mean': np.ndarray(784)}

    j = 0
    for i in range(np.size(X, 0)):
        C[j]['points'].append(X[i])
        j += 1
        j %= K

    for j in range(K):
        C[j]['mean'] = np.average(C[j]['points'], axis=0)

    return C

def findGivenVectorInAllCentroid(C, vectorToLookFor):
    indexToReturn = -1
    j = 0

    while j < K:
        if vectorToLookFor in C[j]['points']:
            indexToReturn = j
            break

    return indexToReturn

# TODO:
def Kmeans(X):
    k_MaxAllowdExchanges = 20
    C = RandomlyClassifyData(X)
    costFunctionValues = []
    numOfExchanges = np.inf

    while numOfExchanges > k_MaxAllowdExchanges:
        numOfExchanges = 0

        # match all the points to the most compatible centroid
        for i in range(np.size(X, 0)):     # for each point
            distances = []
            for j in range(len(C)):  # check the match to any centroid
                distances.append(distance.euclidean(X[i], C[j]['mean']))

            rightCentroidNumber = np.argmin(distances)
            wrongCentroidNumber = findGivenVectorInAllCentroid(C, X[i])

            if rightCentroidNumber != wrongCentroidNumber:
                numOfExchanges += 1
                C[wrongCentroidNumber]['points'].remove(X[i])
                C[rightCentroidNumber]['points'].append(X[i])

            distances.clear()

        # rearrange the centroids
        for j in range(K):
            C[j]['mean'] = np.average(C[j]['points'], axis=0)

        costFunctionValues.append(CostFunction(C))

    return C, costFunctionValues

# TODO:
def CostFunction(C):
    sum = 0
    for j in range(K):
        for i in range(np.size(C[j]['points'], 0)):
            sum += distance.euclidean(C[j]['points'][i], C[j]['mean'])

    print(sum)
    return sum

trX, teX, trY, teY = mnist(ntrain=60000, ntest=10000, onehot=True)
C, CostFunctionValues = Kmeans(trX)
plt.plot(CostFunctionValues)
plt.show()
#centroids, costFunctionValues = Kmeans(trX, k_NumOfSamples)
