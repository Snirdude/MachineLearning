__author__ = 'abekker'
from load import mnist
import numpy as np
from matplotlib import pyplot as plt

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

#at this stage trX contains a matrix with only 1&2 digits and trY is the labels vector containing 0/1 accordingly

#.
#.
#.
### train your classifier here based on trX and trY

#.
#.
#.
### test your classifier here based on teX and teY

