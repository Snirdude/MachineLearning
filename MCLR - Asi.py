import numpy as np
from load import mnist
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy as sp

def Ysigmoid(Z):
  return (np.divide(np.exp(Z).T,np.sum(np.exp(Z),1))).T


trX, teX, trY, teY = mnist(ntrain=60000, ntest=10000, onehot=True)

Wight = np.random.rand(np.size(trX, 1), np.size(trY, 1))
Alpha = 0.8
eps = 300

# image = np.reshape(trX[2, :], [28, 28])
# plt.imshow(image)
# plt.show()

cost = np.sum(np.log(np.sum(trY*Ysigmoid(np.dot(trX,Wight)))))
dcost_dw = np.ones([np.size(trX, 0), np.size(trY, 1)])

while np.linalg.norm(dcost_dw,ord=1)>eps:
  cost = np.sum(np.log(np.sum(trY*Ysigmoid(np.dot(trX,Wight)))))
  dcost_dw = np.dot((trY- Ysigmoid(np.dot(trX,Wight))).T,trX)
  Wight = (Wight.T + (Alpha/np.size(trX, 0))*dcost_dw).T
  print(np.linalg.norm(dcost_dw,ord=1))

Yestimate = Ysigmoid(np.dot(trX,Wight))
maxlike = np.argmax(Yestimate,axis=1)
Ytrain = np.argmax(trY,axis=1)

classification_rate=(sum(maxlike==Ytrain)*100.0)/np.size(Ytrain,0)
print("the classification train is")
print(classification_rate)

Yestimate_test = Ysigmoid(np.dot(teX,Wight))
maxlike_test = np.argmax(Yestimate_test,axis=1)
Ytest = np.argmax(teY,axis=1)

classification_rate_test=(sum(maxlike_test==Ytest)*100.0)/np.size(Ytest,0)
print("the classification Test is")
print(classification_rate_test)