import numpy as np
import functions as func

# Download images
X = func.load_images("train-images-idx3-ubyte.gz")

# Download labels
Y = func.download("train-labels-idx1-ubyte.gz")

# Transform to one hot vector
Y_one_hot = func.get_one_hot(Y)

# Get mini batch
X = func.get_mini_batch(X)
X = func.forward(X)
Y_one_hot = func.get_mini_batch(Y_one_hot)

# Calculate cross entropy loss
e = func.cross_entropy_loss(X, Y_one_hot)
print(f"Cross entropy loss: {e}")
