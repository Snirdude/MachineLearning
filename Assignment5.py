import numpy as np
import scipy.io as MathLoader
from matplotlib import pyplot as plt

k_NumOfTrainingDataCells = 8
k_NumOfTestDataCells = 3

def Calculate_K_Projection(i_TestData):
    return PCA_New(i_TestData)

# TODO: finish mession number 6
def CalculateAllEquals(i_TrainData, i_TestData, i_SamplesNumber):
    from scipy.spatial import distance

    locationInTest = 0
    locationInTrain = 0
    k_MaxDistance = 0.5
    indexes = []

    for currentSampleIndex in range(i_SamplesNumber):  # iterate throw all the data
        for i in range(k_NumOfTrainingDataCells):   # for each train in collection of 8
            for j in range(k_NumOfTestDataCells):   # compare each test in collection of 3
                callculatedDistance = distance.euclidean(i_TrainData[i + locationInTrain], i_TestData[locationInTest + j])  # check difference between the pictures
                if callculatedDistance <= k_MaxDistance:
                    indexes.append([i + locationInTrain, locationInTest + j])
        locationInTest += k_NumOfTestDataCells
        locationInTrain += k_NumOfTrainingDataCells

    return indexes

# TODO: finish mession number 4
# 2 options: compare the X' matrices or the projection matrices
#using sklearn.decomposition.PCA
def ComparePCAs1(X):
    from sklearn.decomposition import PCA
    from scipy.spatial import distance

    pcaObj = PCA(n_components=0.9)
    pcaObj.fit(X)
    xTagByPython = pcaObj.transform(X)
    xTagByUs, meanFace = PCA_New(X, 0.9)

    print(xTagByUs.shape)
    print(xTagByPython.shape)
    return np.abs(np.subtract(xTagByUs, xTagByPython))

# using matplotlib.mlab.PCA
def ComparePCAs2(X):
    from matplotlib.mlab import PCA
    from scipy.spatial import distance

    pcaObj = PCA(X)
    xTagByPython = pcaObj.project()
    xTagByUs, meanFace = PCA_New(X, 0.9)

    return np.abs(np.subtract(xTagByUs, xTagByPython))

def SpiltDataToTrainingAndTest(X):
    """
    :param X:
    :return:
    """
    numOfDataCells = int(np.size(X, 0) / (k_NumOfTrainingDataCells + k_NumOfTestDataCells)) # get the num of personnels
    compressedData = np.split(X, numOfDataCells)  # split the data to personnel
    train = []
    test = []
    for i in range(numOfDataCells):
        currentLocationInRow = 0

        while currentLocationInRow < 8:
            train.append(compressedData[i][currentLocationInRow])
            currentLocationInRow += 1
        while currentLocationInRow < 11:
            test.append(compressedData[i][currentLocationInRow])
            currentLocationInRow += 1

    return train, test

def PCA_New(X, Threshold):
    """
    :param X: the data matrix
    :param Threshold: define the total percentage of data to save
    :return: the X' matrix and the mean face as tuple
    """
    X = X - np.mean(X)
    CovX = np.dot(X.T, X) / np.size(X, 0)
    eigenValues, eigenVectors = np.linalg.eig(CovX)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    eigenValuesSum = np.sum(eigenValues)
    k = 0
    accumulatedEigenValues = []
    while np.sum(accumulatedEigenValues) < eigenValuesSum * Threshold:
        accumulatedEigenValues.append(eigenValues[k])
        k += 1

    A = eigenVectors[:, 0:k]

    return np.dot(X, A), eigenVectors[0]

facesData = MathLoader.loadmat('facesData')
trainX, testX = SpiltDataToTrainingAndTest(facesData['faces'])
Y = facesData['labeles']

# X_projected, meanFace = PCA_New(facesData['faces'], 0.9)    # calculate PCA on the data
# k = X_projected.shape[1]                                    # get the K dimensions of the data after PCA algorithm

print(ComparePCAs1(facesData['faces']))

# indexes = CalculateAllEquals(trainX, testX, 15)
# print(indexes)

# plt.imshow(meanFace.real.reshape(32, 32).T)
# plt.gray()
# plt.show()
