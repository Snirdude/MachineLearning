from matplotlib import pyplot as plt
from load import mnist
import numpy as np
from math import log

# updated in 21/8

def sigmoid(x):
    if any(y > 100 for y in x):
        return 1
    elif any(y < -100 for y in x):
        return 0.000001
    else:
        calc = 1 / (1 + np.exp(-x))
        return calc

# function J beta with regularization
def computeLw(X, w, numOfExamples):
    likelihood = 0

    for t in range(numOfExamples):
        currentElement = sigmoid(w.T.dot(X[t]))
        likelihood += log(currentElement)

    likelihood /= numOfExamples

    return likelihood

# change vector beta simultaneously with gradient ascent
def gradientAscent(X, w, y, numOfExamples, numOfFeatures, i_Tao):
    wUpdated = w.copy()

    wUpdated = X.T.dot(y - sigmoid(X.dot(w)))

    return wUpdated


L = [] # stands for the calculated values of L for w/y
tao = 0.000001
m = 784
n = 12665
wVec = np.random.normal(0, 1, [784, 1])
trX, teX, trY, teY = mnist(onehot=True)
x = np.array(trX[0, :])
x = x.reshape([28, 28])
#plt.imshow(x)

# binary classification
idtr0 = np.where(np.dot(trY, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) == 1)
idtr1 = np.where(np.dot(trY, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]) == 1)

idtrF = np.concatenate((idtr0, idtr1), axis=1)

idte0 = np.where(np.dot(teY, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) == 1)
idte1 = np.where(np.dot(teY, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]) == 1)

idteF = np.concatenate((idte0, idte1), axis=1)
trX = trX[idtrF].reshape((len(idtrF[0]), 784))

teX = teX[idteF].reshape((2115, 784))
trY = trY[idtrF]
trY = np.dot(trY, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
trY = np.transpose(trY)
teY = teY[idteF]
teY = np.dot(teY, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
teY = np.transpose(teY)

Lvalue = computeLw(trX, wVec, n)
L.append(Lvalue)

# while abs(Jvalue - J[len(J) - 1]) > epsilon:   # until converge calculate gradient descent

for i in range(20):
    print(Lvalue)
    wVec = gradientAscent(trX, wVec, trY, n, m, tao)
    Lvalue = computeLw(trX, wVec, n)
    L.append(Lvalue)

plt.plot(L)
plt.show()