import pandas as pd
import numpy as np
from scipy.special import expit
import math

# adjust vector w with gradient ascent
def gradientAscent(X, w, y, i_Tao, i_WeightOfOnes):
    derivative = np.dot(X.T, (i_WeightOfOnes * y * (1 - expit(np.dot(X, w))
                                                    - (1 - i_WeightOfOnes) * (1 - y) * expit(np.dot(X, w)))))\
                 / np.size(X, 0)
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


X = pd.read_csv('Data/train_numeric.csv', nrows=20000, usecols=range(969))
Y = pd.read_csv('Data/train_numeric.csv', nrows=20000, usecols=range(969, 970))
trX = X.values[0:17000]
trY = Y.values[0:17000]
teX = X.values[17000:20000]
teY = Y.values[17000:20000]
trX = replaceNansWithMedians(trX)
teX = replaceNansWithMedians(teX)

w = np.random.normal(0, 1, [969, 1])
tao = 0.0001
weightOfOnes = np.size(trY[trY == 1]) / np.size(teY)

for i in range(50):
    w = gradientAscent(trX, w, trY, tao, weightOfOnes)

yEstimate = expit(np.dot(teX, w))
yEstimate = np.where(yEstimate > 0.5, 1, 0)
print((yEstimate == teY).sum() * 100 / np.size(teY))
