import numpy as np
import scipy.io as MathLoader

from matplotlib import pyplot as plt

k_NumOfTrainingDataCells = 8
k_NumOfTestDataCells = 3

# TODO: mession number 5
def Calculate_K_Projection():
    # do it bitch

# TODO: mession number 6
def CalculateAllEquals(i_TrainData, i_TestData):
    from scipy.spatial import distance

# TODO: mession number 4
def ComparePCAs(X):
    from matplotlib.mlab import PCA

def SpiltDataToTrainingAndTest(X):
    train = []
    test = []
    numOfDataCells = int(np.size(X, 0) / (k_NumOfTrainingDataCells + k_NumOfTestDataCells)) # get the num of personnels
    compressedData = np.split(X, numOfDataCells)  # split the data to personnel

    for i in range(numOfDataCells):
        currentLocationInRow = 0
        trainList = []
        testList = []

        while currentLocationInRow < 8:
            trainList.append(compressedData[i][currentLocationInRow])
            currentLocationInRow += 1
        train.append(trainList)
        while currentLocationInRow < 11:
            testList.append(compressedData[i][currentLocationInRow])
            currentLocationInRow += 1
        test.append(testList)

    return train, test

def PCA_New(X, Threshold):
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
    return np.dot(X, A) # maybe return only the k

facesData = MathLoader.loadmat('facesData')
trainX, testX = SpiltDataToTrainingAndTest(facesData['faces'])
Y = facesData['labeles']
print(Y.shape)

# plt.imshow(testX[0][0].reshape((32,32)).T)
# plt.gray()
# plt.show()



