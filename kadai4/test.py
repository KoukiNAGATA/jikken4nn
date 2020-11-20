import numpy as np
import mnist

# Define const
D = 784 # input
M = 100 # middle
C = 10 # output labels
DEV1 = np.sqrt(1/D)
DEV2 = np.sqrt(1/M)

TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
TEST_LABELS = "t10k-labels-idx1-ubyte.gz"

# Initialize dropout rate
dropout_rate = 0.5

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
        self.x = load_images(TEST_IMAGES)
        # Download test labels
        self.l = download(TEST_LABELS)
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
        print("Sigmoid")
        print(f"Correct answer rate: {correct_answer_rate}%")

# ReLU
class TestA1():
    def __init__(self):
        # Download test images
        self.x = load_images(TEST_IMAGES)
        # Download test labels
        self.l = download(TEST_LABELS)
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
        print("ReLU")
        print(f"Correct answer rate: {correct_answer_rate}%")

# ReLU, Dropout
class TestA2():
    def __init__(self):
        # Download test images
        self.x = load_images(TEST_IMAGES)
        # Download test labels
        self.l = download(TEST_LABELS)
        # Download parameters
        parameters = np.load('parameter/kadaia2.npz')

        self.w1 = parameters['arr_0']
        self.w2 = parameters['arr_1']
        self.b1 = parameters['arr_2']
        self.b2 = parameters['arr_3']

    # Forward propagation
    def forward(self, x):
        y1 = ReLU(np.dot(x, self.w1) + self.b1)
        y1 = self.dropout(y1)
        y2 = softmax(np.dot(y1,self. w2) + self.b2)
        return y2

    # Dropout function
    def dropout(self, t):
        return t * (1 - dropout_rate)

    def test(self):
        image_size = len(self.x)
        correct_number = 0
        for i in range(image_size):
            y = self.forward(self.x[i])
            num = np.argmax(y)
            if self.l[i] == num:
                correct_number += 1
        correct_answer_rate = correct_number / image_size * 100
        print("SGD")
        print(f"Correct answer rate: {correct_answer_rate}%")

# Momentum
class TestA4_1():
    def __init__(self):
        # Download test images
        self.x = load_images(TEST_IMAGES)
        # Download test labels
        self.l = download(TEST_LABELS)
        # Download parameters
        parameters = np.load('parameter/kadaia4_1.npz')

        self.w1 = parameters['arr_0']
        self.w2 = parameters['arr_1']
        self.b1 = parameters['arr_2']
        self.b2 = parameters['arr_3']

    # Forward propagation
    def forward(self, x):
        y1 = ReLU(np.dot(x, self.w1) + self.b1)
        y1 = self.dropout(y1)
        y2 = softmax(np.dot(y1,self. w2) + self.b2)
        return y2

    # Dropout function
    def dropout(self, t):
        return t * (1 - dropout_rate)

    def test(self):
        image_size = len(self.x)
        correct_number = 0
        for i in range(image_size):
            y = self.forward(self.x[i])
            num = np.argmax(y)
            if self.l[i] == num:
                correct_number += 1
        correct_answer_rate = correct_number / image_size * 100
        print("Momentum")
        print(f"Correct answer rate: {correct_answer_rate}%")

# AdaGrad
class TestA4_2():
    def __init__(self):
        # Download test images
        self.x = load_images(TEST_IMAGES)
        # Download test labels
        self.l = download(TEST_LABELS)
        # Download parameters
        parameters = np.load('parameter/kadaia4_2.npz')

        self.w1 = parameters['arr_0']
        self.w2 = parameters['arr_1']
        self.b1 = parameters['arr_2']
        self.b2 = parameters['arr_3']

    # Forward propagation
    def forward(self, x):
        y1 = ReLU(np.dot(x, self.w1) + self.b1)
        y1 = self.dropout(y1)
        y2 = softmax(np.dot(y1,self. w2) + self.b2)
        return y2

    # Dropout function
    def dropout(self, t):
        return t * (1 - dropout_rate)

    def test(self):
        image_size = len(self.x)
        correct_number = 0
        for i in range(image_size):
            y = self.forward(self.x[i])
            num = np.argmax(y)
            if self.l[i] == num:
                correct_number += 1
        correct_answer_rate = correct_number / image_size * 100
        print("AdaGrad")
        print(f"Correct answer rate: {correct_answer_rate}%")

# RMSProp
class TestA4_3():
    def __init__(self):
        # Download test images
        self.x = load_images(TEST_IMAGES)
        # Download test labels
        self.l = download(TEST_LABELS)
        # Download parameters
        parameters = np.load('parameter/kadaia4_3.npz')

        self.w1 = parameters['arr_0']
        self.w2 = parameters['arr_1']
        self.b1 = parameters['arr_2']
        self.b2 = parameters['arr_3']

    # Forward propagation
    def forward(self, x):
        y1 = ReLU(np.dot(x, self.w1) + self.b1)
        y1 = self.dropout(y1)
        y2 = softmax(np.dot(y1,self. w2) + self.b2)
        return y2

    # Dropout function
    def dropout(self, t):
        return t * (1 - dropout_rate)

    def test(self):
        image_size = len(self.x)
        correct_number = 0
        for i in range(image_size):
            y = self.forward(self.x[i])
            num = np.argmax(y)
            if self.l[i] == num:
                correct_number += 1
        correct_answer_rate = correct_number / image_size * 100
        print("RMSProp")
        print(f"Correct answer rate: {correct_answer_rate}%")

# AdaDelta
class TestA4_4():
    def __init__(self):
        # Download test images
        self.x = load_images(TEST_IMAGES)
        # Download test labels
        self.l = download(TEST_LABELS)
        # Download parameters
        parameters = np.load('parameter/kadaia4_4.npz')

        self.w1 = parameters['arr_0']
        self.w2 = parameters['arr_1']
        self.b1 = parameters['arr_2']
        self.b2 = parameters['arr_3']

    # Forward propagation
    def forward(self, x):
        y1 = ReLU(np.dot(x, self.w1) + self.b1)
        y1 = self.dropout(y1)
        y2 = softmax(np.dot(y1,self. w2) + self.b2)
        return y2

    # Dropout function
    def dropout(self, t):
        return t * (1 - dropout_rate)

    def test(self):
        image_size = len(self.x)
        correct_number = 0
        for i in range(image_size):
            y = self.forward(self.x[i])
            num = np.argmax(y)
            if self.l[i] == num:
                correct_number += 1
        correct_answer_rate = correct_number / image_size * 100
        print("AdaDelta")
        print(f"Correct answer rate: {correct_answer_rate}%")

# Adam
class TestA4_5():
    def __init__(self):
        # Download test images
        self.x = load_images(TEST_IMAGES)
        # Download test labels
        self.l = download(TEST_LABELS)
        # Download parameters
        parameters = np.load('parameter/kadaia4_5.npz')

        self.w1 = parameters['arr_0']
        self.w2 = parameters['arr_1']
        self.b1 = parameters['arr_2']
        self.b2 = parameters['arr_3']

    # Forward propagation
    def forward(self, x):
        y1 = ReLU(np.dot(x, self.w1) + self.b1)
        y1 = self.dropout(y1)
        y2 = softmax(np.dot(y1,self. w2) + self.b2)
        return y2

    # Dropout function
    def dropout(self, t):
        return t * (1 - dropout_rate)

    def test(self):
        image_size = len(self.x)
        correct_number = 0
        for i in range(image_size):
            y = self.forward(self.x[i])
            num = np.argmax(y)
            if self.l[i] == num:
                correct_number += 1
        correct_answer_rate = correct_number / image_size * 100
        print("Adam")
        print(f"Correct answer rate: {correct_answer_rate}%")

#####################################################################

# Run task
if __name__ == "__main__":
    t4 = Test4()
    t4.test()

    t_a1 = TestA1()
    t_a1.test()

    t_a2 = TestA2()
    t_a2.test()

    t_a4_1 = TestA4_1()
    t_a4_1.test()

    t_a4_2 = TestA4_2()
    t_a4_2.test()

    t_a4_3 = TestA4_3()
    t_a4_3.test()

    t_a4_4 = TestA4_4()
    t_a4_4.test()

    t_a4_5 = TestA4_5()
    t_a4_5.test()