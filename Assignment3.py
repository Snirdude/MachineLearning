from matplotlib import pyplot as plt
from load import mnist
import numpy as np
from scipy.special import expit

# updated in 21/8

# function L(w)
def computeLw(X, w, y):
    return np.sum(np.log(y * expit(np.dot(X, w)) + (1 - y) * expit(1 - np.dot(X, w)))) / np.size(X, 0)

# adjust vector w with gradient ascent
def gradientAscent(X, w, y, i_Tao):
    derivative = np.dot(X.T, (y - expit(np.dot(X, w)))) / np.size(X, 0)
    return w + i_Tao * derivative


L = [] # stands for the calculated values of L for w/y
tao = 0.01
eps = 0.0001
m = 784
n = 12665
wVec = np.random.normal(0, 1, [784, 1])
trX, teX, trY, teY = mnist(onehot=True)

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

Lvalue = computeLw(trX, wVec, trY)
L.append(Lvalue)

for i in range(1000):
    wVec = gradientAscent(trX, wVec, trY, tao)
    Lvalue = computeLw(trX, wVec, trY)
    L.append(Lvalue)

plt.plot(L)
plt.show()

yEstimate = expit(np.dot(wVec.T, teX.T).T)
yEstimate = np.where(yEstimate > 0.5, 1, 0)
print((yEstimate == teY).sum() * 100 / np.size(teY))