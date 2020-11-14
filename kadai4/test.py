import numpy as np
import mnist

# Define const
D = 784 # input
M = 100 # middle
C = 10 # output labels
DEV1 = np.sqrt(1/D)
DEV2 = np.sqrt(1/M)

#####################################################################

# Preprocessing
def preprocessing(x):
    x = normalize(x)
    x = x.reshape(x.shape[0], D)
    return x

def normalize(x):
    return x / 255.0

def download(data):
    # Download data
    y = mnist.download_and_parse_mnist_file(data)
    return y

def load_images(images):
    # Download images
    x = download(images)
    x = preprocessing(x)
    return x

#####################################################################

# Functions

# Sigmoid function
def sigmoid(t):
    return 1 / (1 + np.exp(-t))

# ReLU function
def ReLU(t):
    return t * (t > 0)

# Softmax function
def softmax(a):
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

def cross_entropy_loss(x, y):
    return  -np.sum(y * np.log(x+(1e-7))) / y.shape[0]

#####################################################################

# Test
class Test4():
    def __init__(self):
        # Download test images
        self.x = load_images("t10k-images-idx3-ubyte.gz")
        # Download test labels
        self.l = download("t10k-labels-idx1-ubyte.gz")
        # Download parameters
        parameters = np.load('parameter/kadai4.npz')

        self.w1 = parameters['arr_0']
        self.w2 = parameters['arr_1']
        self.b1 = parameters['arr_2']
        self.b2 = parameters['arr_3']

    # Forward propagation
    def forward(self, x):
        y1 = sigmoid(np.dot(x, self.w1) + self.b1)
        y2 = softmax(np.dot(y1,self. w2) + self.b2)
        return y2

    def test(self):
        image_size = len(self.x)
        correct_number = 0
        for i in range(image_size):
            y = self.forward(self.x[i])
            num = np.argmax(y)
            if self.l[i] == num:
                correct_number += 1
        correct_answer_rate = correct_number / image_size * 100
        print(f"Correct answer rate: {correct_answer_rate}%")

class TestA1():
    def __init__(self):
        # Download test images
        self.x = load_images("t10k-images-idx3-ubyte.gz")
        # Download test labels
        self.l = download("t10k-labels-idx1-ubyte.gz")
        # Download parameters
        parameters = np.load('parameter/kadaia1.npz')

        self.w1 = parameters['arr_0']
        self.w2 = parameters['arr_1']
        self.b1 = parameters['arr_2']
        self.b2 = parameters['arr_3']

    # Forward propagation
    def forward(self, x):
        y1 = ReLU(np.dot(x, self.w1) + self.b1)
        y2 = softmax(np.dot(y1,self. w2) + self.b2)
        return y2

    def test(self):
        image_size = len(self.x)
        correct_number = 0
        for i in range(image_size):
            y = self.forward(self.x[i])
            num = np.argmax(y)
            if self.l[i] == num:
                correct_number += 1
        correct_answer_rate = correct_number / image_size * 100
        print(f"Correct answer rate: {correct_answer_rate}%")

#####################################################################

# Run task
if __name__ == "__main__":
    t = TestA1()
    t.test()