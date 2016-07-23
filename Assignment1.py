import numpy
from matplotlib import pyplot as plt

n = 1000
m = 40

xMatrix = numpy.random.rand(n, m) * 100
bParametersVec = numpy.random.rand(m, 1) * 5
noiseVec = numpy.random.rand(n, 1)
yVec = xMatrix.dot(bParametersVec) + noiseVec
bEstimated = numpy.l
numpy.linalg.inv(xMatrix.transpose().dot(xMatrix)).dot(xMatrix.transpose().dot(yVec))
differenceNorma = numpy.linalg.matrix_power(numpy.linalg.norm(bParametersVec - bEstimated, axis=1), 2)

print(differenceNorma)

plt.plot(differenceNorma)
plt.show()