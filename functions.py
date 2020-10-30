import numpy as np

# Functions
def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def softmax(a):
    alpha = np.max(a)
    return np.exp(a-alpha)/np.sum(np.exp(a-alpha))

def cross_entropy_error(y, c):
    return  -np.sum(c * np.log(np.maximum(y, 1e-7))) / y.shape[0]

def normalize(x):
    return x / 255.0
