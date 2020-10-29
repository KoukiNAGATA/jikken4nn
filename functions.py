import numpy as np

# Functions
def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def softmax(a):
    alpha = np.max(a)
    return np.exp(a-alpha)/np.sum(np.exp(a-alpha))

def cross_entropy_error(x, t):
    return  -np.sum(t * np.log(np.maximum(x, 1e-7)))/x.shape[0]

def normalize(x):
    return x / 255.0
