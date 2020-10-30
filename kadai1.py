import numpy as np
import functions as func

# Download images
X = func.load_images("train-images-idx3-ubyte.gz")

# Input number
try:
    val = int(input('Enter a number 0 or more to 9999 or less: '))
    image = X[val]
    if(val > 9999):
        raise IndexError
except IndexError:
    # IndexError
    print('Invalid index.')
    raise
except:
    # Error
    print('Invalid string.')
    raise

# Run task
y = func.forward(image)
num = np.argmax(y)
print(num)