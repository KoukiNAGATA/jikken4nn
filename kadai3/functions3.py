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

#####################################################################

# Functions
def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def softmax(a):
    # Get the maximum and the sum per column.
    alpha = np.max(a, axis = 1)
    column_sum = np.sum(np.exp(a-alpha[:, np.newaxis]), axis = 1)
    return np.exp(a-alpha[:, np.newaxis]) / column_sum[:, np.newaxis]

def cross_entropy_loss(x, y):
    return  -np.sum(y * np.log(x+(1e-7))) / y.shape[0]

def normalize(x):
    return x / 255.0

#####################################################################

# Learning
def train(x, l):
    # Get the size of dataset
    image_size = x.shape[0]

    # Initialize weights and biases
    w1, w2, b1, b2 = initialize()

    for i in range(EPOCH_SIZE):
        # Print the number of the epoch
        print(f"Epoch: {i+1}")
        En = np.zeros(0)

        for j in range(int(image_size / BATCH_SIZE)):
            # Learning
            w1, w2, b1, b2, En_tmp = learn(w1, w2, b1, b2, x, l)
            En = np.append(En, En_tmp)

        # Print cross entropy loss average of the epoch
        En_average = np.average(En)
        print(f"Cross entropy loss: {En_average}")

    np.savez('parameter/kadai3', w1, w2, b1, b2)

def learn(w1, w2, b1, b2, x, l):
    # Input layer
    # Get mini batch
    random_index = np.random.choice(x.shape[0], BATCH_SIZE, replace = False)
    x = x[random_index]
    l = l[random_index]

    # Forward propagation
    y1 = sigmoid(np.dot(x, w1) + b1)
    y2 = softmax(np.dot(y1, w2) + b2)

    # Back propagation(Softmax function and Cross entropy loss)
    dEn_da2 = (y2 - l) / BATCH_SIZE

    # Back propagation(Fully connected layer)
    dEn_dX2 = np.dot(dEn_da2, w2.T)
    dEn_dw2 = np.dot(y1.T, dEn_da2)
    dEn_db2 = np.sum(dEn_da2 , axis=0)

    # Back propagation(Sigmoid function)
    dEn_da1 = dEn_dX2 * (1 - y1) * y1

    # Back propagation(Fully connected layer)
    dEn_dw1 = np.dot(x.T, dEn_da1)
    dEn_db1 = np.sum(dEn_da1 , axis=0)

    # Update weights and biases
    w1 -= lr * dEn_dw1
    w2 -= lr * dEn_dw2
    b1 -= lr * dEn_db1
    b2 -= lr * dEn_db2

    # Calculate cross entropy loss
    En = cross_entropy_loss(forward(x, w1, w2, b1, b2), l)
    return (w1, w2, b1, b2, En)