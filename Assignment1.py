import numpy
from matplotlib import pyplot as plt

n = 1000
m = 40

xMatrix = numpy.random.rand(n, m) * 100
bParametersVec = numpy.random.rand(m, 1) * 5
noiseVec = numpy.random.rand(n, 1) * 100
yVec = xMatrix.dot(bParametersVec) + noiseVec
bEstimated = numpy.linalg.inv(xMatrix.transpose().dot(xMatrix)).dot(xMatrix.transpose().dot(yVec))

differenceVec = numpy.power((bParametersVec - bEstimated), 2)

plt.plot(differenceVec)
plt.show()