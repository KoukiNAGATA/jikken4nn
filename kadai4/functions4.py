import numpy as np
import mnist

# Define const
D = 784 # input
M = 100 # middle
C = 10 # output labels
DEV1 = np.sqrt(1/D)
DEV2 = np.sqrt(1/M)
EPOCH_SIZE = 10 # number of epoch

# Number of mini batch
BATCH_SIZE = 100

# Initialize learning rate
lr = 0.01

#####################################################################

def initialize():
    # Initialize weights
    w1 = np.random.normal(0, DEV1, (D, M))
    w2 = np.random.normal(0, DEV2, (M, C))

    # Initialize biases
    b1 = np.random.normal(0, DEV1, M)
    b2 = np.random.normal(0, DEV2, C)
    return (w1, w2, b1, b2)

def preprocessing(x):
    x = normalize(x)
    x = x.reshape(x.shape[0], D)
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
    return np.eye(C)[y]

# Forward propagation
def forward(x, w1, w2, b1, b2):
    y1 = sigmoid(np.dot(x, w1) + b1)
    y2 = softmax(np.dot(y1, w2) + b2)
    return y2

# Forward propagation
def forward_one(x, w1, w2, b1, b2):
    y1 = sigmoid(np.dot(x, w1) + b1)
    y2 = softmax_one(np.dot(y1, w2) + b2)
    return y2

#####################################################################

# Functions
def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def softmax(a):
    # Get the maximum and the sum per column.
    alpha = np.max(a, axis = 1)
    column_sum = np.sum(np.exp(a-alpha[:, np.newaxis]), axis = 1)
    return np.exp(a-alpha[:, np.newaxis]) / column_sum[:, np.newaxis]

def softmax_one(a):
    # Get the maximum and the sum per column.
    alpha = np.max(a)
    column_sum = np.sum(np.exp(a-alpha))
    return np.exp(a-alpha) / column_sum

def cross_entropy_loss(x, y):
    return  -np.sum(y * np.log(x+(1e-7))) / y.shape[0]

def normalize(x):
    return x / 255.0

def ReLU(x):
    

#####################################################################

# Test
def test(x, l, w1, w2, b1, b2):
    image_size = len(x)
    correct_number = 0
    for i in range(image_size):
        y = forward_one(x[i], w1, w2, b1, b2)
        num = np.argmax(y)
        if l[i] == num:
            correct_number += 1
    correct_answer_rate = correct_number / image_size * 100
    print(f"Correct answer rate: {correct_answer_rate}%")
