import pandas as pd
import numpy as np
from scipy.special import expit
import math


def computeMuAndSigma(X):
    mu = np.mean(X)
    sigma = np.std(X)

    print(mu)
    print(sigma)

    return mu, sigma


def addExamples(X, Y):
    ones = np.size(np.where(Y == 1)[0])
    zeros = np.size(np.where(Y == 0)[0])
    if ones < zeros:
        indices = np.where(Y == 1)[0]
        count = zeros - ones
    else:
        indices = np.where(Y == 0)[0]
        count = ones - zeros

    mu, sigma = computeMuAndSigma(X[indices])
    for i in range(count):
        X = np.append(X, [sigma * np.random.randn(969) + mu], axis=0)
        Y = np.append(Y, [Y[indices[0]]], axis=0)

    return X, Y


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


X = pd.read_csv('Data/train_numeric.csv', nrows=5000, usecols=range(969))
Y = pd.read_csv('Data/train_numeric.csv', nrows=5000, usecols=range(969, 970))
trX = X.values[0:4500]
trY = Y.values[0:4500]
teX = X.values[4500:5000]
teY = Y.values[4500:5000]
trX = replaceNansWithMedians(trX)
teX = replaceNansWithMedians(teX)

w = np.random.normal(0, 1, [969, 1])
tao = 0.0001
addExamples(trX, trY)

for i in range(50):
    w = gradientAscent(trX, w, trY, tao)

yEstimate = expit(np.dot(teX, w))
yEstimate = np.where(yEstimate > 0.5, 1, 0)
print((yEstimate == teY).sum() * 100 / np.size(teY))