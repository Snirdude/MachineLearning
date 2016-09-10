from matplotlib import pyplot as plt
from load import mnist
import numpy as np
from scipy.spatial import distance
from collections import Counter
K = 10

def RandomlyClassifyDataBestMatch(X):
    """
    initialize the centroids with known images for best centroid start
    :param X: the train data matrix
    :return: list of dictionaries with centroids location and all the images inside
    """
    C = list(range(10))

    for j in range(K):
        C[j] = {'points': [], 'mean': np.ndarray(784)}

    C[0]['points'].append(X[1].tolist())
    C[1]['points'].append(X[3].tolist())
    C[2]['points'].append(X[5].tolist())
    C[3]['points'].append(X[7].tolist())
    C[4]['points'].append(X[2].tolist())
    C[5]['points'].append(X[0].tolist())
    C[6]['points'].append(X[13].tolist())
    C[7]['points'].append(X[15].tolist())
    C[8]['points'].append(X[17].tolist())
    C[9]['points'].append(X[116].tolist())

    for j in range(K):
        C[j]['mean'] = np.average(C[j]['points'], axis=0)

    return C

def RandomlyClassifyData(X):
    """
    initialize the centroids kind of random data
    :param X: the train data matrix
    :return: list of dictionaries with centroids location and all the images inside
    """
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
    k_MaxAllowdExchanges = int(np.size(X, 0) * 0.02)
    C = RandomlyClassifyDataBestMatch(X)    # for best initialization
    # C = RandomlyClassifyData(X)           # for random initialization
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
                if wrongCentroidNumber != -1:
                    C[wrongCentroidNumber]['points'].remove(X[i].tolist())
                C[rightCentroidNumber]['points'].append(X[i].tolist())

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

def successRate(C, X, Y):
    Labels = list(range(10))
    for j in range(K):
        Labels[j] = []

    for i in range(np.size(X, 0)):
        for j in range(K):
            if X[i].tolist() in C[j]['points']:
                Labels[j].append(Y[i])
                break

    correctOccurences = 0
    falseOccurences = 0
    for j in range(K):
        commonLabel, numOfOccurences = Counter(Labels[j]).most_common(1)[0]
        indices = [i for i, x in enumerate(Labels[j]) if x != commonLabel]
        correctOccurences += numOfOccurences
        falseOccurences += len(indices)

    return (correctOccurences / (falseOccurences + correctOccurences)) * 100

trX, teX, trY, teY = mnist(ntrain=60000, ntest=10000, onehot=False)
X = trX[0:200]
Y = trY[0:200]
C, CostFunctionValues = Kmeans(X)

print(successRate(C, X, Y))
plt.plot(CostFunctionValues)
plt.show()