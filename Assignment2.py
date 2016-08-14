import numpy as np
import csv
import pandas as pd
from matplotlib import pyplot as plt

k_NoSignalFromRouter = 100


# function J beta with regularization
def computeJbeta(X, b, y, numOfExamples, numOfFeatures, lamda):
    squareError = 0
    for t in range(numOfExamples):
        squareError += (y[t] - b.T.dot(X[t])) ** 2

    squareError /= 2 * numOfExamples
    regularization = b.T.dot(b)
    regularization *= lamda / (2 * numOfFeatures)

    return squareError + regularization


# compute the entire derivative for J beta L1 Norm ( L2 Norm square)
def gradientVectorL1Norm(X, b, y, numOfExamples, numOfFeatures, i_Lamda):
    squareError = 0
    for t in range(numOfExamples):
        squareError += (b.T.dot(X[t]) - y[t]) * X[t]

    squareError /= numOfExamples
    regularization = (i_Lamda / numOfFeatures) * b

    return np.linalg.norm(squareError + regularization, 2) ** 2


# change vector beta simultaneously with gradient descent
def gradientDescent(X, b, y, numOfExamples, numOfFeatures, i_Lamda, i_Tao):
    betaUpdated = b.copy()
    for j in range(numOfFeatures):
        squareError = 0
        for t in range(numOfExamples):
            squareError += (b.T.dot(X[t]) - y[t]) * X[t][j]

        squareError /= numOfExamples
        regularization = (i_Lamda / numOfFeatures) * b[j]
        betaUpdated[j] = b[j] - i_Tao * (squareError + regularization)

    return betaUpdated


def SSres(X, b, y, n):
    squareError = 0
    for t in range(n):
        squareError += (y[t] - b.T.dot(X[t])) ** 2

    return squareError


def SStot(y, n):
    average = y.sum() / n
    sum = 0
    for i in range(n):
        sum += (y[i] - average) ** 2

    return sum


def Rsquare(X, b, y, n):
    return 1 - (SSres(X, b, y, n) / SStot(y, n))


def featureScalingAndNormalization(X, n, m):
    sum = 0
    countOfNot100 = 0
    for i in range(n):
        for j in range(m):
            if X.values[i][j] != 100:
                sum += X.values[i][j]
                countOfNot100 += 1

    averageOfSignals = sum / countOfNot100
    X[X == k_NoSignalFromRouter] = averageOfSignals
    for i in range(n):
        for j in range(m):
            X.values[i][j] /= 104  # max - min, max = 0 min = -104


J1 = [] # stands for the calculated values of J for B1/y1
J2 = [] # stands for the calculated values of J for B2/y2
G1 = [] # stands for the gradient vector for B1
G2 = [] # stands for the gradient vector for B2
lamda = 0.01
tao = 0.001
m = 520
n = 100

X = pd.read_csv('trainingData.csv', usecols=range(0, 520))
y1 = pd.read_csv('trainingData.csv', usecols=range(520, 521))
y2 = pd.read_csv('trainingData.csv', usecols=range(521, 522))

featureScalingAndNormalization(X, n, m)
vectorEx5 = []
vectorEx6 = []
for k in range(5):
    print(str(k))
    J1 = []
    J2 = []
    beta1 = np.random.rand(520, 1)
    beta2 = np.random.rand(520, 1)
    Jvalue = computeJbeta(X.values, beta1, y1.values, n, m, lamda)
    J1.append(Jvalue)
    Jvalue = computeJbeta(X.values, beta2, y2.values, n, m, lamda)
    J2.append(Jvalue)

    # while abs(Jvalue - J[len(J) - 1]) > epsilon:   # until converge calculate gradient descent

    for i in range(20):
        beta1 = gradientDescent(X.values, beta1, y1.values, n, m, lamda, tao)
        beta2 = gradientDescent(X.values, beta2, y2.values, n, m, lamda, tao)
        G1.append(gradientVectorL1Norm(X.values, beta1, y1.values, n, m, lamda))
        G2.append(gradientVectorL1Norm(X.values, beta2, y2.values, n, m, lamda))
        Jvalue = computeJbeta(X.values, beta1, y1.values, n, m, lamda)
        J1.append(Jvalue[0][0])
        print(i, Jvalue)
        Jvalue = computeJbeta(X.values, beta2, y2.values, n, m, lamda)
        J2.append(Jvalue[0][0])

    print('Rsquare for training = ', str((Rsquare(X.values, beta1, y1.values, n) +
                                          Rsquare(X.values, beta2, y2.values, n)) / 2))

    X = pd.read_csv('validationData.csv', usecols=range(0, 520))
    y1 = pd.read_csv('validationData.csv', usecols=range(520, 521))
    y2 = pd.read_csv('validationData.csv', usecols=range(521, 522))

    featureScalingAndNormalization(X, n, m)

    vectorEx6.append((Rsquare(X.values, beta1, y1.values, n) + Rsquare(X.values, beta2, y2.values, n)) / 2)
    print('Rsquare for validation = ', str(vectorEx6[len(vectorEx6) - 1]))

    count = 0
    for i in range(m):
        if beta1[i] < 0.001:
            count += 1

    vectorEx5.append(count)
    lamda *= 10

plt.plot(vectorEx6)
plt.show()