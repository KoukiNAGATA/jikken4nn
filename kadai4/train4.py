import numpy as np
import mnist
import matplotlib.pyplot as plt

# Define const
D = 784 # input
M = 100 # middle
C = 10 # output labels
DEV1 = np.sqrt(1/D)
DEV2 = np.sqrt(1/M)
EPOCH_SIZE = 10 # number of epoch

# Number of mini batch
BATCH_SIZE = 32

# Initialize learning rate
lr = 0.01

#####################################################################

# processing
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

def normalize(x):
    return x / 255.0

#####################################################################

# Functions

# Sigmoid function
def sigmoid_forward(t):
    return 1 / (1 + np.exp(-t))

def sigmoid_backward(dEn_dy, t):
    return dEn_dy * (1 - t) * t

# ReLU function
def ReLU_forward(t):
    return t * (t > 0)

def ReLU_backward(dEn_dy, t):
    return dEn_dy * (t > 0)

# Dropout function
def dropout_forward(t):
    return t

# Softmax function
def softmax_forward(a):
    if(a.ndim == 2):
        # Get the maximum and the sum per column.
        alpha = np.max(a, axis = 1)
        column_sum = np.sum(np.exp(a-alpha[:, np.newaxis]), axis = 1)
        return np.exp(a-alpha[:, np.newaxis]) / column_sum[:, np.newaxis]
    else:
        # Get the maximum and the sum.
        alpha = np.max(a)
        column_sum = np.sum(np.exp(a-alpha))
        return np.exp(a-alpha) / column_sum

def softmax_backward(yk2, yk):
    return (yk2 - yk) / BATCH_SIZE

# Cross entropy loss
def cross_entropy_loss(x, y):
    return -np.sum(y * np.log(x+(1e-7))) / y.shape[0]

#####################################################################

class Network():
    def __init__(self):
        # Download test images
        self.x = load_images("train-images-idx3-ubyte.gz")
        # Download test labels
        self.l = get_one_hot(download("train-labels-idx1-ubyte.gz"))
        # Initialize weights
        self.w1 = np.random.normal(0, DEV1, (D, M))
        self.w2 = np.random.normal(0, DEV2, (M, C))

        # Initialize biases
        self.b1 = np.random.normal(0, DEV1, M)
        self.b2 = np.random.normal(0, DEV2, C)

    # Forward propagation
    def forward(self, x):
        y1 = sigmoid_forward(np.dot(x, self.w1) + self.b1)
        y2 = softmax_forward(np.dot(y1, self.w2) + self.b2)
        return y2

    # Learning
    def learn(self):
        # Input layer
        x = self.x
        l = self.l
        # Get mini batch
        random_index = np.random.choice(x.shape[0], BATCH_SIZE, replace = False)
        x = x[random_index]
        l = l[random_index]

        # Forward propagation
        y1 = sigmoid_forward(np.dot(x, self.w1) + self.b1)
        y2 = softmax_forward(np.dot(y1, self.w2) + self.b2)

        # Back propagation(softmax function and Cross entropy loss)
        dEn_da2 = softmax_backward(y2, l)

        # Back propagation(Fully connected layer)
        dEn_dX2 = np.dot(dEn_da2, self.w2.T)
        dEn_dw2 = np.dot(y1.T, dEn_da2)
        dEn_db2 = np.sum(dEn_da2 , axis=0)

        # Back propagation(sigmoid function)
        dEn_da1 = sigmoid_backward(dEn_dX2, y1)

        # Back propagation(Fully connected layer)
        dEn_dw1 = np.dot(x.T, dEn_da1)
        dEn_db1 = np.sum(dEn_da1 , axis=0)

        # Update weights and biases
        self.w1 -= lr * dEn_dw1
        self.w2 -= lr * dEn_dw2
        self.b1 -= lr * dEn_db1
        self.b2 -= lr * dEn_db2

        # Calculate cross entropy loss
        a = self.forward(x)
        En = cross_entropy_loss(a, l)
        return En

    def train(self):
        # Get the size of dataset
        image_size = self.x.shape[0]

        # Initialize the list of cross entropy loss
        En_average_list = np.zeros(0)

        for i in range(EPOCH_SIZE):
            # Print the number of the epoch
            print(f"Epoch: {i+1}")
            En = np.zeros(0)

            for j in range(int(image_size / BATCH_SIZE)):
                # Learning
                En_tmp = self.learn()
                En = np.append(En, En_tmp)

            # Print cross entropy loss average of the epoch
            En_average = np.average(En)
            print(f"Cross entropy loss: {En_average}")
            En_average_list = np.append(En_average_list, En_average)

        np.savez('parameter/kadai4', self.w1, self.w2, self.b1, self.b2)
        plt.plot(En_average_list)
        plt.show()

class NetworkA1():
    def __init__(self):
        # Download test images
        self.x = load_images("train-images-idx3-ubyte.gz")
        # Download test labels
        self.l = get_one_hot(download("train-labels-idx1-ubyte.gz"))
        # Initialize weights
        self.w1 = np.random.normal(0, DEV1, (D, M))
        self.w2 = np.random.normal(0, DEV2, (M, C))

        # Initialize biases
        self.b1 = np.random.normal(0, DEV1, M)
        self.b2 = np.random.normal(0, DEV2, C)

    # Forward propagation
    def forward(self, x):
        y1 = ReLU_forward(np.dot(x, self.w1) + self.b1)
        y2 = softmax_forward(np.dot(y1, self.w2) + self.b2)
        return y2

    # Learning
    def learn(self):
        # Input layer
        x = self.x
        l = self.l
        # Get mini batch
        random_index = np.random.choice(x.shape[0], BATCH_SIZE, replace = False)
        x = x[random_index]
        l = l[random_index]

        # Forward propagation
        y1 = ReLU_forward(np.dot(x, self.w1) + self.b1)
        y2 = softmax_forward(np.dot(y1, self.w2) + self.b2)

        # Back propagation(softmax function and Cross entropy loss)
        dEn_da2 = softmax_backward(y2, l)

        # Back propagation(Fully connected layer)
        dEn_dX2 = np.dot(dEn_da2, self.w2.T)
        dEn_dw2 = np.dot(y1.T, dEn_da2)
        dEn_db2 = np.sum(dEn_da2 , axis=0)

        # Back propagation(ReLU function)
        dEn_da1 = ReLU_backward(dEn_dX2, y1)

        # Back propagation(Fully connected layer)
        dEn_dw1 = np.dot(x.T, dEn_da1)
        dEn_db1 = np.sum(dEn_da1 , axis=0)

        # Update weights and biases
        self.w1 -= lr * dEn_dw1
        self.w2 -= lr * dEn_dw2
        self.b1 -= lr * dEn_db1
        self.b2 -= lr * dEn_db2

        # Calculate cross entropy loss
        a = self.forward(x)
        En = cross_entropy_loss(a, l)
        return En

    def train(self):
        # Get the size of dataset
        image_size = self.x.shape[0]

        # Initialize the list of cross entropy loss
        En_average_list = np.zeros(0)

        for i in range(EPOCH_SIZE):
            # Print the number of the epoch
            print(f"Epoch: {i+1}")
            En = np.zeros(0)

            for j in range(int(image_size / BATCH_SIZE)):
                # Learning
                En_tmp = self.learn()
                En = np.append(En, En_tmp)

            # Print cross entropy loss average of the epoch
            En_average = np.average(En)
            print(f"Cross entropy loss: {En_average}")
            En_average_list = np.append(En_average_list, En_average)

        np.savez('parameter/kadaia1', self.w1, self.w2, self.b1, self.b2)
        plt.plot(En_average_list)
        plt.show()

# Run task
if __name__ == "__main__":
    n = NetworkA1()
    n.train()