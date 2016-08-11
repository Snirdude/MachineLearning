import numpy as np
import csv
import pandas as pd
from matplotlib import pyplot as plt


def computeJbeta(X, b, y, numOfExamples, numOfFeatures, lamda):
    squareError = 0
    for t in range(numOfExamples):
        squareError += (y[t] - b.T.dot(X[t])) ** 2
    squareError /= 2 * numOfExamples
    regularization = 0
    for j in range(numOfFeatures):
        regularization += b[j] ** 2
    regularization *= lamda / (2 * numOfFeatures)

    return squareError + regularization


def gradientAscent(X, b, y, numOfExamples, numOfFeatures, i_Lamda, i_Tao):
    betaUpdated = b.copy()
    for j in range(numOfFeatures):
        squareError = 0
        for t in range(numOfExamples):
            squareError += (y[t] - b.T.dot(X[t])) * X[t][j]
        squareError /= numOfExamples
        #print('squareError ' + str(squareError))
        regularization = (i_Lamda / numOfFeatures) * b[j]
        betaUpdated[j] = b[j] - i_Tao * (squareError + regularization)
        print(betaUpdated[j])

    return betaUpdated


J = []
lamda = 0.03
tao = 0.01
epsilon = 0.001
m = 520
beta = np.random.rand(520, 1) * 3
betaUpdated = beta

X = pd.read_csv('trainingData.csv', usecols=range(0, 520))
y1 = pd.read_csv('trainingData.csv', usecols=range(520, 521))
y2 = pd.read_csv('trainingData.csv', usecols=range(521, 522))
n = 500#len(y1)

X[X == 100] = -104
Jvalue = computeJbeta(X.values, beta, y1.values, n, m, lamda)
J.append(Jvalue)
beta = gradientAscent(X.values, beta, y1.values, n, m, lamda, tao)
Jvalue = computeJbeta(X.values, beta, y1.values, n, m, lamda)
#while abs(Jvalue - J[len(J) - 1]) > epsilon:
for i in range(10):
    print(i, Jvalue)
    J.append(Jvalue)
    beta = gradientAscent(X.values, beta, y1.values, n, m, lamda, tao)
    Jvalue = computeJbeta(X.values, beta, y1.values, n, m, lamda)

plt.plot(J)
plt.show()