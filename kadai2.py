import numpy as np
import random
import mnist
import functions as func

# Define const
d = 784 # input

# Number of mini batch
batch_size = 100

# Set seed
np.random.seed(seed=4)
random.seed(4)

#####################################################################

# Download images
X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
# Download labels
Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")

# Preprocessing
X = func.normalize(X)
X = X.reshape(X.shape[0], d)

# Get mini batch
X = X[np.random.choice(X.shape[0], 100, replace = False)]

# ここまでできた

# Calculate cross entropy
e = func.cross_entropy_error(X, 1)

print("Cross entropy error: " + e)
