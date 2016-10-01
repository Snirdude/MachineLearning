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

    C[0]['mean'] = X[1]
    C[1]['mean'] = X[3]
    C[2]['mean'] = X[5]
    C[3]['mean'] = X[7]
    C[4]['mean'] = X[2]
    C[5]['mean'] = X[0]
    C[6]['mean'] = X[13]
    C[7]['mean'] = X[15]
    C[8]['mean'] = X[17]
    C[9]['mean'] = X[116]

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

def Kmeans(X, Y):
    k_MaxAllowdExchanges = int(np.size(X, 0) * 0.02)
    C = RandomlyClassifyDataBestMatch(X)    # for best initialization
    # C = RandomlyClassifyData(X)           # for random initialization
    costFunctionValues = []
    numOfExchanges = np.inf

    while numOfExchanges > k_MaxAllowdExchanges:
        numOfExchanges = 0
        rightAnswer = 0
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

        print(rightAnswer)

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
C, CostFunctionValues = Kmeans(X, Y)
print(successRate(C, X, Y))
plt.plot(CostFunctionValues)
plt.show()
