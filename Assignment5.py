import numpy as np
import scipy.io as MathLoader
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.decomposition import PCA

k_NumOfTrainingDataCells = 8
k_NumOfTestDataCells = 3

def GetMeanFace(i_TestData):
    return i_TestData[0]

def CalculateAllEquals(trainX, testX):
    """
    :param trainX: train data
    :param testX: test data
    :return: index array of the faces that was recognized
    """
    from scipy.spatial import distance

    List = []
    for i in range(np.size(testX, 0)):
        Min = np.inf
        idx = -1
        for j in range(np.size(trainX, 0)):
            if distance.euclidean(testX[i], trainX[j]) < Min:
                Min = distance.euclidean(testX[i], trainX[j])
                idx = j
        List.append(int(idx / 8) + 1)

    return List

# TODO: finish mission number 4
def PyhtonPCA(X):
    """
    :param X: The data matrix
    :return: new data matrix with fewer features
    """
    return PCA(n_components=0.9).fit_transform(X)

def SplitDataToTrainingAndTest(X):
    """
    :param X: data array
    :return: train array and test array
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

def SplitLabelsToTrainingAndTest(Y):
    """
    :param Y: labels array
    :return: train array and test array
    """
    numOfDataCells = int(np.size(Y, 0) / (k_NumOfTrainingDataCells + k_NumOfTestDataCells))  # get the num of personnels
    train = []
    test = []
    for i in range(numOfDataCells):

        for j in range(8):
            train.append(Y[i * 11 + j][0])
        for j in range(8, 11):
            test.append(Y[i * 11 + j][0])

    return train, test

def PCA_New(X, Threshold):
    """
    :param X: the data matrix
    :param Threshold: define the total percentage of data to save
    :return: the projection matrix that fit to the given X matrix data
    """
    newX = X.copy() - np.mean(X)
    CovX = np.dot(newX.T, newX) / np.size(newX, 0)
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

    return eigenVectors[:, 0:k]

facesData = MathLoader.loadmat('facesData')
X = np.dot(facesData['faces'], PCA_New(facesData['faces'], 0.9))    # calculate X'
trainX, testX = SplitDataToTrainingAndTest(X)
Y = facesData['labeles']
trainY, testY = SplitLabelsToTrainingAndTest(Y)

List = CalculateAllEquals(trainX=trainX, testX=testX)
X = PyhtonPCA(facesData['faces'])
trainX, testX = SplitDataToTrainingAndTest(X)
PythonList = CalculateAllEquals(trainX=trainX, testX=testX)

print("Success rate for face recognition with our PCA: ", len([i for i, j in zip(List, testY) if i == j]) * 100 / np.size(testY))
print("Success rate for face recognition with Python PCA: ", len([i for i, j in zip(PythonList, testY) if i == j]) * 100 / np.size(testY))
