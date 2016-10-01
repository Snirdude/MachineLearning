import pandas as pd
import numpy as np
from scipy.special import expit
import math

def addExamples(X, Y):
    ones = np.size(np.where(Y == 1)[0])
    zeros = np.size(np.where(Y == 0)[0])
    if ones < zeros:
        index = findClassIndex(Y, 1)
        count = zeros - ones
    else:
        index = findClassIndex(Y, 0)
        count = ones - zeros

    for i in range(count):
        X = np.append(X, [X[index]], axis=0)
        Y = np.append(Y, [Y[index]], axis=0)

    return X, Y

def findClassIndex(Y, i_Class):
    for i in range(np.size(Y, 0)):
        if Y[i] == i_Class:
            return i


# adjust vector w with gradient ascent
def gradientAscent(X, w, y, i_Tao):
    derivative = np.dot(X.T, (y - expit(np.dot(X, w)))) / np.size(X, 0)
    return w + i_Tao * derivative


def replaceNansWithMedians(X):
    for i in range(np.size(X, 0)):
        nonNansList = []
        for j in range(np.size(X[i])):
            if not math.isnan(X[i][j]):
                nonNansList.append(X[i][j])

        median = np.median(nonNansList)
        for j in range(np.size(X[i])):
            if math.isnan(X[i][j]):
                X[i][j] = median

    return X


X = pd.read_csv('Data/train_numeric.csv', nrows=1500, usecols=range(969))
Y = pd.read_csv('Data/train_numeric.csv', nrows=1500, usecols=range(969, 970))
trX = X.values[0:1200]
trY = Y.values[0:1200]
teX = X.values[1200:1500]
teY = Y.values[1200:1500]
trX = replaceNansWithMedians(trX)
teX = replaceNansWithMedians(teX)

w = np.random.normal(0, 1, [969, 1])
tao = 0.0001
trX, trY = addExamples(trX, trY)
print(np.size(trX, 0))

for i in range(50):
    w = gradientAscent(trX, w, trY, tao)

yEstimate = expit(np.dot(teX, w))
yEstimate = np.where(yEstimate > 0.5, 1, 0)
print((yEstimate == teY).sum() * 100 / np.size(teY))
