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

# Initialize weight
w1 = np.random.normal(0, DEV1, (D, M))
w2 = np.random.normal(0, DEV2, (M, C))

# Initialize bias
b1 = np.random.normal(0, DEV1, M)
b2 = np.random.normal(0, DEV2, C)

# Initialize learning rate
lr = 0.01

# Set seed
np.random.seed(seed=4)

#####################################################################

# Forward propagation
def forward(x):
    y1 = sigmoid(np.dot(x, w1) + b1)
    y2 = softmax(np.dot(y1, w2) + b2)
    return y2

def forward(x, w1, w2, b1, b2):
    y1 = sigmoid(np.dot(x, w1) + b1)
    y2 = softmax(np.dot(y1, w2) + b2)
    return y2

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

def get_mini_batch(x):
    return x[np.random.choice(x.shape[0], BATCH_SIZE, replace = False)]

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
def learn(w1, w2, b1, b2, X, l_one_hot):
    # Input layer
    # Get mini batch
    x = get_mini_batch(X)
    l_one_hot = get_mini_batch(l_one_hot)

    # Forward propagation
    y1 = sigmoid(np.dot(x, w1) + b1)
    y2 = softmax(np.dot(y1, w2) + b2)

    # Back propagation(Softmax function and Cross entropy loss)
    dEn_dak = (y2 - l_one_hot) / BATCH_SIZE
    dEn_dw2 = np.dot(dEn_dak, y2.T)
    dEn_db2 = np.sum(dEn_dak , axis=1)

    # Bach propagation(Sigmoid function)
    dEn_dx = dEn_dak * (1 - y1) * y1
    dEn_dw1 = np.dot(dEn_dx, y1.T)
    dEn_db1 = np.sum(dEn_dx , axis=1)

    # Update weights and biases
    w1 -= lr * dEn_dw1
    w2 -= lr * dEn_dw2
    b1 -= lr * dEn_db1
    b2 -= lr * dEn_db2
    return (w1, w2, b1, b2)

def train(X, L):
    # Get the size of dataset
    image_size = X.shape[0]
    # Transform L to one hot vector
    l_one_hot = get_one_hot(L)

    for i in range(EPOCH_SIZE):
        # Print the number of the epoch
        print(f"Epoch: {i+1}")

        for j in range(int(X.shape[0] / BATCH_SIZE)):
            # Learning
            (w1, w2, b1, b2) = learn(w1, w2, b1, b2, X, l_one_hot)

        # Calculate cross entropy loss
        En = cross_entropy_loss((get_mini_batch(X), w1, w2, b1, b2), get_mini_batch(l_one_hot))
        # Print cross entropy loss at the end of the epoch
        print(f"Cross entropy loss: {En}")
    return