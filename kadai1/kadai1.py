import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm

# Define dimension
d = 784 # input
m = 100 # middle
c = 10 # output
dev1 = np.sqrt(1/d)
dev2 = np.sqrt(1/m)

# Set seed
np.random.seed(seed=4)

# Initialize weight
w1 = np.random.normal(0, dev1, (d, m))
w2 = np.random.normal(0, dev2, (m, c))

# Initialize bias
b1 = np.random.normal(0, dev1, m)
b2 = np.random.normal(0, dev2, c)

# Functions
def sigmoid(t):
    a = 1 / (1 + np.exp(-t))
    return a

def softmax(a):
    alpha = np.max(a)
    y = np.exp(a-alpha)/np.sum(np.exp(a-alpha))
    return y

# Preprocessing
def normalize(x):
    x = x / 255.0
    return x

# Input layer
def input_layer(x):
    x = x.reshape(d)
    return x

# Forward propagation
def forward(x):
    x = input_layer(normalize(x))
    y1 = sigmoid(np.dot(x, w1) + b1)
    y2 = softmax(np.dot(y1, w2) + b2)
    return y2

#####################################################################

# Download images
X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
# Download labels
Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")

# Input number
try:
    val = int(input('Enter a number 0 or more to 9999 or less: '))
    image = X[val]
    if(val > 9999):
        raise IndexError
except IndexError:
    # IndexError
    print('Invalid index.')
    raise
except:
    # Error
    print('Invalid string.')
    raise

# Run task
y = forward(image)
num = np.argmax(y)
print(num)