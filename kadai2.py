import numpy as np
import mnist
import functions as func

# Define const
d = 784 # input
c = 10 # labels

# Number of mini batch
batch_size = 100

# Set seed
np.random.seed(seed=4)

#####################################################################

# Download labels
Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")

# Transform to one hot vector
Y_one_hot = np.eye(c)[Y]

# Get mini batch
Y_one_hot = Y_one_hot[np.random.choice(Y_one_hot.shape[0], batch_size, replace = False)]
print(Y_one_hot.shape)

# Calculate cross entropy
e = func.cross_entropy_error(Y_one_hot, c)
print(f"Cross entropy error: {e}")
