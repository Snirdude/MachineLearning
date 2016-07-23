import numpy
from matplotlib import pyplot as plt

n = 1000
m = 40

sigmaVec = []
xMatrix = numpy.random.rand(n, m) * 100
bParametersVec = numpy.random.rand(m, 1) * 5

for i in range(1, 1000):
    noiseVec = numpy.random.rand(n, 1) * i
    yVec = numpy.dot(xMatrix, bParametersVec) + noiseVec
    bEstimated = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(xMatrix.T, xMatrix)), xMatrix.T), yVec)
    differenceNorma = numpy.power(numpy.linalg.norm(bParametersVec - bEstimated), 2)
    sigmaVec.append(differenceNorma)

plt.plot(sigmaVec)
plt.show()
