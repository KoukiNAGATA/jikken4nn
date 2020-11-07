import numpy as np
import functions3 as func

# Download images
X = func.load_images("train-images-idx3-ubyte.gz")
# Download labels
L = func.download("train-labels-idx1-ubyte.gz")

# Transform L to one hot vector
l_one_hot = func.get_one_hot(L)

# Training
func.train(X, l_one_hot)