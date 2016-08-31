from matplotlib import pyplot as plt
from load import mnist
import numpy as np
#

numOfClasses = 10

# probability function for calculate the probability of seeing the number i in dependence of w & x
# returns matrix n x k
def P(X, w):
    numerator = np.exp(np.dot(X, w)).T
    denominator = np.sum(np.exp(np.dot(X, w)), 1)
    return np.divide(numerator, denominator).T

# function L(w)
def computeLw(X, w, y):
    return np.sum(np.log(np.sum(y * P(X, w))))


# adjust vector w with gradient ascent
def gradientAscent(X, w, y, i_Tao):
    derivative = np.dot(X.T, (y - P(X, w))) / np.size(X, 0)
    return w + i_Tao * derivative


L = [] # stands for the calculated values of L for w/y
tao = 0.8
eps = 0.001
m = 784
n = 12665
wVec = np.random.rand(m, numOfClasses) / 1000
trX, teX, trY, teY = mnist(ntrain=60000, ntest=10000, onehot=True)

Lvalue = computeLw(trX, wVec, trY)
L.append(Lvalue)
wVec = gradientAscent(trX, wVec, trY, tao)
LvalueNew = computeLw(trX, wVec, trY)

while abs(LvalueNew - Lvalue) > eps:
    print(LvalueNew)
    Lvalue = LvalueNew
    L.append(Lvalue)
    wVec = gradientAscent(trX, wVec, trY, tao)
    LvalueNew = computeLw(trX, wVec, trY)

plt.plot(L)
plt.show()

yEstimate = P(teX, wVec)
yEstimateArgmax = np.argmax(yEstimate, axis=1)
yTestArgmax = np.argmax(teY, axis=1)
print((yEstimateArgmax == yTestArgmax).sum() * 100 / np.size(teY, 0))
