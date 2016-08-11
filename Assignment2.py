import numpy as np
import csv
import pandas as pd
from matplotlib import pyplot as plt


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

    print(np.linalg.norm(squareError + regularization, 2) ** 2)
    return np.linalg.norm(squareError + regularization, 2) ** 2


#change vector beta simultaneously with gradient descent
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


J1 = []
J2 = []
G1 = []
G2 = []
lamda = 0.01
tao = 0.000001
epsilon = 0.001
m = 520
n = 200
beta1 = np.random.rand(520, 1)
beta2 = np.random.rand(520, 1)

X = pd.read_csv('trainingData.csv', usecols=range(0, 520))
y1 = pd.read_csv('trainingData.csv', usecols=range(520, 521))
y2 = pd.read_csv('trainingData.csv', usecols=range(521, 522))

X[X == 100] = -52 # average of signal
Jvalue = computeJbeta(X.values, beta1, y1.values, n, m, lamda)
J1.append(Jvalue)
Jvalue = computeJbeta(X.values, beta2, y2.values, n, m, lamda)
J2.append(Jvalue)
Jvalue = computeJbeta(X.values, beta1, y1.values, n, m, lamda)
#while abs(Jvalue - J[len(J) - 1]) > epsilon:
for i in range(20):
    beta1 = gradientDescent(X.values, beta1, y1.values, n, m, lamda, tao)
    beta2 = gradientDescent(X.values, beta2, y2.values, n, m, lamda, tao)
    G1.append(gradientVectorL1Norm(X.values, beta1, y1.values, n, m, lamda))
    G2.append(gradientVectorL1Norm(X.values, beta2, y2.values, n, m, lamda))
    Jvalue = computeJbeta(X.values, beta1, y1.values, n, m, lamda)
    J1.append(Jvalue[0][0])
    Jvalue = computeJbeta(X.values, beta2, y2.values, n, m, lamda)
    J2.append(Jvalue[0][0])

#plt.plot(J1)
#plt.plot(J2)
plt.plot(G1)
plt.plot(G2)
plt.show()