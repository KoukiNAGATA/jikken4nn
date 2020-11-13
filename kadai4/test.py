import numpy as np
import functions4 as func

# Download test images
X = func.load_images("t10k-images-idx3-ubyte.gz")
# Download test labels
L = func.download("t10k-labels-idx1-ubyte.gz")
# Download parameters
parameters = np.load('parameter/kadai3.npz')
w1 = parameters['arr_0']
w2 = parameters['arr_1']
b1 = parameters['arr_2']
b2 = parameters['arr_3']

# Run task
func.test(X, L, w1, w2, b1, b2)