import numpy as np
import csv
from matplotlib import pyplot as plt


def computeJbeta(X, b, y, n, m, lamda):
    squareError = 0.0
    for t in range(0, n - 1):
        print(type(X[t]))
        squareError += (y[t] - b.t.dot(X[t])) ** 2
    squareError /= 2 * n
    regularization = 0
    for j in range(0, m - 1):
        regularization += pow(beta[j], 2)
    regularization *= lamda / (2 * m)
    return squareError + regularization


def gradientAscent(X, b, y, n, m, lamda):
    squareError = 0
    for t in range(0, n - 1):
        squareError += (y[t] - b.transpose().dot(X[t])) * X[t]
    squareError /= n
    regularization = (lamda / m) * beta
    for i in range(0, m - 1):
        betaUpdated[i] = beta[i] + tao * (squareError + regularization)

    return betaUpdated


X = np.ndarray
y1 = []
y2 = []
J = []
lamda = 3
tao = 5
epsilon = 0.001
m = 520
beta = np.random.rand(520, 1) * 3
betaUpdated = beta

with open('trainingData.csv', 'r') as training:
    reader = csv.reader(training, 'excel')
    n = 0
    for row in reader:
        if n > 0:
            X.put(np.ndarray(row[:520]))
            y1.append(row[52:521])
            y2.append(row[521:522])
        n += 1

Jvalue = computeJbeta(X, beta, y1, n, m, lamda)
J.append(Jvalue)
beta = gradientAscent(X, beta, y1, n, m, lamda)
Jvalue = computeJbeta(X, beta, y1, n, m, lamda)
while abs(Jvalue - J[len(J) - 1]) > epsilon:
    J.append(Jvalue)
    beta = gradientAscent(X, beta, y1, n, m, lamda)
    Jvalue = computeJbeta(X, beta, y1, n, m, lamda)

plt.plot(J)
plt.show()