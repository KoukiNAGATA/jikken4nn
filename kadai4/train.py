import numpy as np
import mnist
import matplotlib.pyplot as plt

# Define const
D = 784 # input
M = 100 # middle
C = 10 # output labels
DEV1 = np.sqrt(1/D)
DEV2 = np.sqrt(1/M)
EPOCH_SIZE = 50 # number of epoch

TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
TRAIN_LABELS = "train-labels-idx1-ubyte.gz"

# Number of mini batch
BATCH_SIZE = 32

# Initialize learning rate
lr = 0.01

# Initialize dropout rate
dropout_rate = 0.5

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

# Sigmoid
class Network4():
    def __init__(self):
        # Download test images
        self.x = load_images(TRAIN_IMAGES)
        # Download test labels
        self.l = get_one_hot(download(TRAIN_LABELS))

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
        plt.plot(En_average_list, label="Sigmoid")

# ReLU
class NetworkA1():
    def __init__(self):
        # Download test images
        self.x = load_images(TRAIN_IMAGES)
        # Download test labels
        self.l = get_one_hot(download(TRAIN_LABELS))

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
        plt.plot(En_average_list, label="ReLU")

# ReLU, dropout
class NetworkA2():
    def __init__(self):
        # Download test images
        self.x = load_images(TRAIN_IMAGES)
        # Download test labels
        self.l = get_one_hot(download(TRAIN_LABELS))

        # Initialize weights
        self.w1 = np.random.normal(0, DEV1, (D, M))
        self.w2 = np.random.normal(0, DEV2, (M, C))

        # Initialize biases
        self.b1 = np.random.normal(0, DEV1, M)
        self.b2 = np.random.normal(0, DEV2, C)

        # Define drop out mask
        self.mask = None

    # Forward propagation
    def forward(self, x):
        y1 = ReLU_forward(np.dot(x, self.w1) + self.b1)
        y1 = self.dropout_forward(y1)
        y2 = softmax_forward(np.dot(y1, self.w2) + self.b2)
        return y2

    # Dropout function
    def dropout_forward(self, t):
        self.mask = np.random.rand(*t.shape) > (1 - dropout_rate)
        return t * self.mask

    def dropout_backward(self, dt):
        return dt * self.mask

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
        y1 = self.dropout_forward(y1)
        y2 = softmax_forward(np.dot(y1, self.w2) + self.b2)

        # Back propagation(softmax function and Cross entropy loss)
        dEn_da2 = softmax_backward(y2, l)

        # Back propagation(Fully connected layer)
        dEn_dX2 = np.dot(dEn_da2, self.w2.T)
        dEn_dw2 = np.dot(y1.T, dEn_da2)
        dEn_db2 = np.sum(dEn_da2 , axis=0)

        # Back propagation(ReLU function)
        dEn_da1 = self.dropout_backward(dEn_dX2)
        dEn_da1 = ReLU_backward(dEn_da1, y1)

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

        np.savez('parameter/kadaia2', self.w1, self.w2, self.b1, self.b2)
        plt.plot(En_average_list, label="SGD")

# ReLU, dropout, Momentum
class NetworkA4_1():
    def __init__(self):
        # Download test images
        self.x = load_images(TRAIN_IMAGES)
        # Download test labels
        self.l = get_one_hot(download(TRAIN_LABELS))

        # Initialize weights
        self.w1 = np.random.normal(0, DEV1, (D, M))
        self.w2 = np.random.normal(0, DEV2, (M, C))
        self.dw1 = 0
        self.dw2 = 0

        # Initialize biases
        self.b1 = np.random.normal(0, DEV1, M)
        self.b2 = np.random.normal(0, DEV2, C)
        self.db1 = 0
        self.db2 = 0

        # Initialize biases learning rate
        self.α = 0.9

        # Define drop out mask
        self.mask = None

    # Forward propagation
    def forward(self, x):
        y1 = ReLU_forward(np.dot(x, self.w1) + self.b1)
        y1 = self.dropout_forward(y1)
        y2 = softmax_forward(np.dot(y1, self.w2) + self.b2)
        return y2

    # Dropout function
    def dropout_forward(self, t):
        self.mask = np.random.rand(*t.shape) > (1 - dropout_rate)
        return t * self.mask

    def dropout_backward(self, dt):
        return dt * self.mask

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
        y1 = self.dropout_forward(y1)
        y2 = softmax_forward(np.dot(y1, self.w2) + self.b2)

        # Back propagation(softmax function and Cross entropy loss)
        dEn_da2 = softmax_backward(y2, l)

        # Back propagation(Fully connected layer)
        dEn_dX2 = np.dot(dEn_da2, self.w2.T)
        dEn_dw2 = np.dot(y1.T, dEn_da2)
        dEn_db2 = np.sum(dEn_da2 , axis=0)

        # Back propagation(ReLU function)
        dEn_da1 = self.dropout_backward(dEn_dX2)
        dEn_da1 = ReLU_backward(dEn_da1, y1)

        # Back propagation(Fully connected layer)
        dEn_dw1 = np.dot(x.T, dEn_da1)
        dEn_db1 = np.sum(dEn_da1 , axis=0)

        # Update weights and biases
        self.dw1 = self.α * self.dw1 - lr * dEn_dw1
        self.dw2 = self.α * self.dw2 - lr * dEn_dw2
        self.db1 = self.α * self.db1 - lr * dEn_db1
        self.db2 = self.α * self.db2 - lr * dEn_db2

        self.w1 += self.dw1
        self.w2 += self.dw2
        self.b1 += self.db1
        self.b2 += self.db2

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

        np.savez('parameter/kadaia4_1', self.w1, self.w2, self.b1, self.b2)
        plt.plot(En_average_list, label="Momentum")

# ReLU, dropout, AdaGrad
class NetworkA4_2():
    def __init__(self):
        # Download test images
        self.x = load_images(TRAIN_IMAGES)
        # Download test labels
        self.l = get_one_hot(download(TRAIN_LABELS))

        # Initialize weights
        self.w1 = np.random.normal(0, DEV1, (D, M))
        self.w2 = np.random.normal(0, DEV2, (M, C))

        # Initialize biases
        self.b1 = np.random.normal(0, DEV1, M)
        self.b2 = np.random.normal(0, DEV2, C)

        # Initialize learning rate
        self.lr = pow(10, -3)
        self.h_w1 = pow(10, -8)
        self.h_w2 = pow(10, -8)
        self.h_b1 = pow(10, -8)
        self.h_b2 = pow(10, -8)

        # Define drop out mask
        self.mask = None

    # Forward propagation
    def forward(self, x):
        y1 = ReLU_forward(np.dot(x, self.w1) + self.b1)
        y1 = self.dropout_forward(y1)
        y2 = softmax_forward(np.dot(y1, self.w2) + self.b2)
        return y2

    # Dropout function
    def dropout_forward(self, t):
        self.mask = np.random.rand(*t.shape) > (1 - dropout_rate)
        return t * self.mask

    def dropout_backward(self, dt):
        return dt * self.mask

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
        y1 = self.dropout_forward(y1)
        y2 = softmax_forward(np.dot(y1, self.w2) + self.b2)

        # Back propagation(softmax function and Cross entropy loss)
        dEn_da2 = softmax_backward(y2, l)

        # Back propagation(Fully connected layer)
        dEn_dX2 = np.dot(dEn_da2, self.w2.T)
        dEn_dw2 = np.dot(y1.T, dEn_da2)
        dEn_db2 = np.sum(dEn_da2 , axis=0)

        # Back propagation(ReLU function)
        dEn_da1 = self.dropout_backward(dEn_dX2)
        dEn_da1 = ReLU_backward(dEn_da1, y1)

        # Back propagation(Fully connected layer)
        dEn_dw1 = np.dot(x.T, dEn_da1)
        dEn_db1 = np.sum(dEn_da1 , axis=0)

        # Update weights and biases
        self.h_w1 += dEn_dw1 * dEn_dw1
        self.h_w2 += dEn_dw2 * dEn_dw2
        self.h_b1 += dEn_db1 * dEn_db1 
        self.h_b2 += dEn_db2 * dEn_db2 

        self.w1 -= self.lr / np.sqrt(self.h_w1) * dEn_dw1
        self.w2 -= self.lr / np.sqrt(self.h_w2) * dEn_dw2
        self.b1 -= self.lr / np.sqrt(self.h_b1) * dEn_db1
        self.b2 -= self.lr / np.sqrt(self.h_b2) * dEn_db2

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

        np.savez('parameter/kadaia4_2', self.w1, self.w2, self.b1, self.b2)
        plt.plot(En_average_list, label="AdaGrad")

# ReLU, dropout, RMSProp
class NetworkA4_3():
    def __init__(self):
        # Download test images
        self.x = load_images(TRAIN_IMAGES)
        # Download test labels
        self.l = get_one_hot(download(TRAIN_LABELS))

        # Initialize weights
        self.w1 = np.random.normal(0, DEV1, (D, M))
        self.w2 = np.random.normal(0, DEV2, (M, C))

        # Initialize biases
        self.b1 = np.random.normal(0, DEV1, M)
        self.b2 = np.random.normal(0, DEV2, C)

        # Initialize learning rate
        self.lr = pow(10, -3)
        self.h_w1 = 0
        self.h_w2 = 0
        self.h_b1 = 0
        self.h_b2 = 0
        self.ρ = 0.9
        self.ε = pow(10, -8)

        # Define drop out mask
        self.mask = None

    # Forward propagation
    def forward(self, x):
        y1 = ReLU_forward(np.dot(x, self.w1) + self.b1)
        y1 = self.dropout_forward(y1)
        y2 = softmax_forward(np.dot(y1, self.w2) + self.b2)
        return y2

    # Dropout function
    def dropout_forward(self, t):
        self.mask = np.random.rand(*t.shape) > (1 - dropout_rate)
        return t * self.mask

    def dropout_backward(self, dt):
        return dt * self.mask

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
        y1 = self.dropout_forward(y1)
        y2 = softmax_forward(np.dot(y1, self.w2) + self.b2)

        # Back propagation(softmax function and Cross entropy loss)
        dEn_da2 = softmax_backward(y2, l)

        # Back propagation(Fully connected layer)
        dEn_dX2 = np.dot(dEn_da2, self.w2.T)
        dEn_dw2 = np.dot(y1.T, dEn_da2)
        dEn_db2 = np.sum(dEn_da2 , axis=0)

        # Back propagation(ReLU function)
        dEn_da1 = self.dropout_backward(dEn_dX2)
        dEn_da1 = ReLU_backward(dEn_da1, y1)

        # Back propagation(Fully connected layer)
        dEn_dw1 = np.dot(x.T, dEn_da1)
        dEn_db1 = np.sum(dEn_da1 , axis=0)

        # Update weights and biases
        self.h_w1 = self.ρ * self.h_w1 + (1 - self.ρ) * dEn_dw1 * dEn_dw1
        self.h_w2 = self.ρ * self.h_w2 + (1 - self.ρ) * dEn_dw2 * dEn_dw2
        self.h_b1 = self.ρ * self.h_b1 + (1 - self.ρ) * dEn_db1 * dEn_db1 
        self.h_b2 = self.ρ * self.h_b2 + (1 - self.ρ) * dEn_db2 * dEn_db2 

        self.w1 -= self.lr / (np.sqrt(self.h_w1) + self.ε) * dEn_dw1
        self.w2 -= self.lr / (np.sqrt(self.h_w2) + self.ε) * dEn_dw2
        self.b1 -= self.lr / (np.sqrt(self.h_b1) + self.ε) * dEn_db1
        self.b2 -= self.lr / (np.sqrt(self.h_b2) + self.ε) * dEn_db2

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

        np.savez('parameter/kadaia4_3', self.w1, self.w2, self.b1, self.b2)
        plt.plot(En_average_list, label="RMSProp")

# ReLU, dropout, AdaDelta
class NetworkA4_4():
    def __init__(self):
        # Download test images
        self.x = load_images(TRAIN_IMAGES)
        # Download test labels
        self.l = get_one_hot(download(TRAIN_LABELS))

        # Initialize weights
        self.w1 = np.random.normal(0, DEV1, (D, M))
        self.w2 = np.random.normal(0, DEV2, (M, C))
        self.dw1 = 0
        self.dw2 = 0

        # Initialize biases
        self.b1 = np.random.normal(0, DEV1, M)
        self.b2 = np.random.normal(0, DEV2, C)
        self.db1 = 0
        self.db2 = 0

        # Initialize learning rate
        self.h_w1 = 0
        self.h_w2 = 0
        self.h_b1 = 0
        self.h_b2 = 0
        self.s_w1 = 0
        self.s_w2 = 0
        self.s_b1 = 0
        self.s_b2 = 0
        self.ρ = 0.95
        self.ε = pow(10, -6)

        # Define drop out mask
        self.mask = None

    # Forward propagation
    def forward(self, x):
        y1 = ReLU_forward(np.dot(x, self.w1) + self.b1)
        y1 = self.dropout_forward(y1)
        y2 = softmax_forward(np.dot(y1, self.w2) + self.b2)
        return y2

    # Dropout function
    def dropout_forward(self, t):
        self.mask = np.random.rand(*t.shape) > (1 - dropout_rate)
        return t * self.mask

    def dropout_backward(self, dt):
        return dt * self.mask

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
        y1 = self.dropout_forward(y1)
        y2 = softmax_forward(np.dot(y1, self.w2) + self.b2)

        # Back propagation(softmax function and Cross entropy loss)
        dEn_da2 = softmax_backward(y2, l)

        # Back propagation(Fully connected layer)
        dEn_dX2 = np.dot(dEn_da2, self.w2.T)
        dEn_dw2 = np.dot(y1.T, dEn_da2)
        dEn_db2 = np.sum(dEn_da2 , axis=0)

        # Back propagation(ReLU function)
        dEn_da1 = self.dropout_backward(dEn_dX2)
        dEn_da1 = ReLU_backward(dEn_da1, y1)

        # Back propagation(Fully connected layer)
        dEn_dw1 = np.dot(x.T, dEn_da1)
        dEn_db1 = np.sum(dEn_da1 , axis=0)

        # Update weights and biases
        self.h_w1 = self.ρ * self.h_w1 + (1 - self.ρ) * dEn_dw1 * dEn_dw1
        self.h_w2 = self.ρ * self.h_w2 + (1 - self.ρ) * dEn_dw2 * dEn_dw2
        self.h_b1 = self.ρ * self.h_b1 + (1 - self.ρ) * dEn_db1 * dEn_db1 
        self.h_b2 = self.ρ * self.h_b2 + (1 - self.ρ) * dEn_db2 * dEn_db2

        self.dw1 = - np.sqrt(self.s_w1 + self.ε) / np.sqrt(self.h_w1 + self.ε) * dEn_dw1
        self.dw2 = - np.sqrt(self.s_w2 + self.ε) / np.sqrt(self.h_w2 + self.ε) * dEn_dw2
        self.db1 = - np.sqrt(self.s_b1 + self.ε) / np.sqrt(self.h_b1 + self.ε) * dEn_db1
        self.db2 = - np.sqrt(self.s_b2 + self.ε) / np.sqrt(self.h_b2 + self.ε) * dEn_db2

        self.s_w1 = self.ρ * self.s_w1 + (1 - self.ρ) * self.dw1 * self.dw1
        self.s_w2 = self.ρ * self.s_w2 + (1 - self.ρ) * self.dw2 * self.dw2
        self.s_b1 = self.ρ * self.s_b1 + (1 - self.ρ) * self.db1 * self.db1 
        self.s_b2 = self.ρ * self.s_b2 + (1 - self.ρ) * self.db2 * self.db2 

        self.w1 += self.dw1
        self.w2 += self.dw2
        self.b1 += self.db1
        self.b2 += self.db2

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

        np.savez('parameter/kadaia4_4', self.w1, self.w2, self.b1, self.b2)
        plt.plot(En_average_list, label="AdaDelta")

# ReLU, dropout, Adam
class NetworkA4_5():
    def __init__(self):
        # Download test images
        self.x = load_images(TRAIN_IMAGES)
        # Download test labels
        self.l = get_one_hot(download(TRAIN_LABELS))

        # Initialize weights
        self.w1 = np.random.normal(0, DEV1, (D, M))
        self.w2 = np.random.normal(0, DEV2, (M, C))

        # Initialize biases
        self.b1 = np.random.normal(0, DEV1, M)
        self.b2 = np.random.normal(0, DEV2, C)

        # Initialize learning rate
        self.t = 0
        self.m_w1 = 0
        self.m_w2 = 0
        self.m_b1 = 0
        self.m_b2 = 0
        self.v_w1 = 0
        self.v_w2 = 0
        self.v_b1 = 0
        self.v_b2 = 0
        self.α = pow(10, -3)
        self.β1 = 0.9
        self.β2 = 0.999
        self.ε = pow(10, -8)

        # Define drop out mask
        self.mask = None

    # Forward propagation
    def forward(self, x):
        y1 = ReLU_forward(np.dot(x, self.w1) + self.b1)
        y1 = self.dropout_forward(y1)
        y2 = softmax_forward(np.dot(y1, self.w2) + self.b2)
        return y2

    # Dropout function
    def dropout_forward(self, t):
        self.mask = np.random.rand(*t.shape) > (1 - dropout_rate)
        return t * self.mask

    def dropout_backward(self, dt):
        return dt * self.mask

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
        y1 = self.dropout_forward(y1)
        y2 = softmax_forward(np.dot(y1, self.w2) + self.b2)

        # Back propagation(softmax function and Cross entropy loss)
        dEn_da2 = softmax_backward(y2, l)

        # Back propagation(Fully connected layer)
        dEn_dX2 = np.dot(dEn_da2, self.w2.T)
        dEn_dw2 = np.dot(y1.T, dEn_da2)
        dEn_db2 = np.sum(dEn_da2 , axis=0)

        # Back propagation(ReLU function)
        dEn_da1 = self.dropout_backward(dEn_dX2)
        dEn_da1 = ReLU_backward(dEn_da1, y1)

        # Back propagation(Fully connected layer)
        dEn_dw1 = np.dot(x.T, dEn_da1)
        dEn_db1 = np.sum(dEn_da1 , axis=0)

        # Update weights and biases
        self.t += 1

        self.m_w1 = self.β1 * self.m_w1 + (1 - self.β1) * dEn_dw1
        self.m_w2 = self.β1 * self.m_w2 + (1 - self.β1) * dEn_dw2
        self.m_b1 = self.β1 * self.m_b1 + (1 - self.β1) * dEn_db1
        self.m_b2 = self.β1 * self.m_b2 + (1 - self.β1) * dEn_db2

        self.v_w1 = self.β2 * self.v_w1 + (1 - self.β2) * dEn_dw1 * dEn_dw1
        self.v_w2 = self.β2 * self.v_w2 + (1 - self.β2) * dEn_dw2 * dEn_dw2
        self.v_b1 = self.β2 * self.v_b1 + (1 - self.β2) * dEn_db1 * dEn_db1 
        self.v_b2 = self.β2 * self.v_b2 + (1 - self.β2) * dEn_db2 * dEn_db2

        self.w1 -= self.α * (self.m_w1 / (1 - pow(self.β1, self.t))) / (np.sqrt(self.v_w1 / (1 - pow(self.β2, self.t))) + self.ε)
        self.w2 -= self.α * (self.m_w2 / (1 - pow(self.β1, self.t))) / (np.sqrt(self.v_w2 / (1 - pow(self.β2, self.t))) + self.ε)
        self.b1 -= self.α * (self.m_b1 / (1 - pow(self.β1, self.t))) / (np.sqrt(self.v_b1 / (1 - pow(self.β2, self.t))) + self.ε)
        self.b2 -= self.α * (self.m_b2 / (1 - pow(self.β1, self.t))) / (np.sqrt(self.v_b2 / (1 - pow(self.β2, self.t))) + self.ε)

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

        np.savez('parameter/kadaia4_5', self.w1, self.w2, self.b1, self.b2)
        plt.plot(En_average_list, label="Adam")

# Run task
if __name__ == "__main__":
    # print("Sigmoid")
    # n4 = Network4()
    # n4.train()

    # print("ReLU")
    # n_a1 = NetworkA1()
    # n_a1.train()

    print("SGD")
    n_a2 = NetworkA2()
    n_a2.train()

    print("Momentum")
    n_a4_1 = NetworkA4_1()
    n_a4_1.train()

    print("AdaGrad")
    n_a4_2 = NetworkA4_2()
    n_a4_2.train()

    print("RMSProp")
    n_a4_3 = NetworkA4_3()
    n_a4_3.train()

    print("AdaDelta")
    n_a4_4 = NetworkA4_4()
    n_a4_4.train()

    print("Adam")
    n_a4_5 = NetworkA4_5()
    n_a4_5.train()

    plt.legend()
    plt.show()