from matplotlib import pyplot as plt
from load import mnist
import numpy as np
from scipy.spatial import distance

K = 10

def RandomlyClassifyData(X):
    C = list(range(10))

    for j in range(K):
        C[j] = {'points': [], 'mean': np.ndarray(784)}

    j = 0
    for i in range(np.size(X, 0)):
        C[j]['points'].append(X[i].tolist())
        j += 1
        j %= K

    for j in range(K):
        C[j]['mean'] = np.average(C[j]['points'], axis=0)

    return C

def findGivenVectorInAllCentroid(C, vectorToLookFor):
    indexToReturn = -1

    for j in range(K):
        if vectorToLookFor in C[j]['points']:
            indexToReturn = j
            break

    return indexToReturn

def Kmeans(X):
    k_MaxAllowdExchanges = int(np.size(X, 0) * 0.03)
    print(int(np.size(X, 0) * 0.05))
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
            wrongCentroidNumber = findGivenVectorInAllCentroid(C, X[i].tolist())

            if rightCentroidNumber != wrongCentroidNumber:
                numOfExchanges += 1
                C[wrongCentroidNumber]['points'].remove(X[i].tolist())
                C[rightCentroidNumber]['points'].append(X[i].tolist())

            distances.clear()

        # rearrange the centroids
        for j in range(K):
            C[j]['mean'] = np.average(C[j]['points'], axis=0)

        costFunctionValues.append(CostFunction(C))

    return C, costFunctionValues

def CostFunction(C):
    sum = 0
    for j in range(K):
        for i in range(np.size(C[j]['points'], 0)):
            sum += distance.euclidean(C[j]['points'][i], C[j]['mean'])

    return sum

trX, teX, trY, teY = mnist(ntrain=60000, ntest=10000, onehot=True)
C, CostFunctionValues = Kmeans(trX[0:5000])
plt.plot(CostFunctionValues)
plt.show()
for j in range(K):
    plt.imshow(C[j]['mean'].reshape((28, 28)))
    plt.show()
