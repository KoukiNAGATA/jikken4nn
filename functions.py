import numpy as np
import mnist

# Define const
d = 784 # input
m = 100 # middle
c = 10 # output labels
dev1 = np.sqrt(1/d)
dev2 = np.sqrt(1/m)

# Number of mini batch
batch_size = 100

# Initialize weight
w1 = np.random.normal(0, dev1, (d, m))
w2 = np.random.normal(0, dev2, (m, c))

# Initialize bias
b1 = np.random.normal(0, dev1, m)
b2 = np.random.normal(0, dev2, c)

# Set seed
np.random.seed(seed=4)

#####################################################################

# Forward propagation
def forward(x):
    y1 = sigmoid(np.dot(x, w1) + b1)
    y2 = softmax(np.dot(y1, w2) + b2)
    return y2

def preprocessing(x):
    x = normalize(x)
    x = x.reshape(x.shape[0], d)
    return x

def download(data):
    # Download data
    y = mnist.download_and_parse_mnist_file(data)
    return y

def load_images(images):
    # Download images
    x = download(images)
    x = preprocessing(x)
    return x

def get_one_hot(y):
    return np.eye(c)[y]

def get_mini_batch(x):
    return x[np.random.choice(x.shape[0], batch_size, replace = False)]

#####################################################################

# Functions
def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def softmax(a):
    alpha = np.max(a, axis = 1)
    return np.exp(a-alpha[:,np.newaxis]) / np.sum(np.exp(a-alpha[:,np.newaxis]))

def cross_entropy_loss(x, y):
    return  -np.sum(y * np.log(x+(1e-7))) / y.shape[0]

def normalize(x):
    return x / 255.0
