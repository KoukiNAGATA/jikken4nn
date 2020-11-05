import numpy as np
import functions as func

# Download images
X = func.load_images("train-images-idx3-ubyte.gz")
# Download labels
L = func.download("train-labels-idx1-ubyte.gz")

# Training
func.train(X, L)