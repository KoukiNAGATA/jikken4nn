import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm

# images
X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
X.shape # (60000, 28, 28)
# labels
Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")

plt.figure()
plt.imshow(x[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Preprocessing
X = X / 255.0

# input layer
X = X.reshape(60000, 784)

# 

try:
  # get number
  val = input('Enter a number 0 or more to 9999 or less: ')
  image = x[val]
except IndexError:
  # Error
  print('Invalid number.')
except :
  # Error
  print('Invalid string.')




def sigmoid(a):
  s = 1 / (1 + e**(-a))
  return s

#get seed
seed = numpy.random.seed(40000)

idx = 100
plt.imshow(X[idx], cmap=cm.gray)
plt.show()
print (Y[idx])