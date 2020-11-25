import numpy as np
import mnist

# Define const
D = 784 # input
M = 100 # middle
C = 10 # output labels
DEV1 = np.sqrt(1/D)
DEV2 = np.sqrt(1/M)

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

# Contest
class Contest():
    def __init__(self):
        # Download test images
        self.x = np.loadtxt("le4MNIST_X.txt")
        self.x = normalize(self.x)
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
        path = 'le4MNIST_answer.txt'
        f = open(path, mode='w')
        for i in range(image_size):
            y = self.forward(self.x[i])
            num = np.argmax(y)
            f.write(str(num)+'\n')
        f.close()
        print('書き込みが完了しました')

#####################################################################

# Run task
if __name__ == "__main__":
    t = Contest()
    t.test()