from max_pooling import MaxPool2
from cnn_filter import Conv3x3
from tensorflow.keras.datasets import mnist

(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0

conv = Conv3x3(8)
pool = MaxPool2()
output = conv.forward(train_images[0])
output = pool.forward(output)
print(output.shape)