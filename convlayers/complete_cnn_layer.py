import numpy as np 
from tensorflow.keras.datasets import mnist 
from cnn_filter import Conv3x3
from max_pooling import MaxPool2
from softmax_conv import Softmax
dat = mnist.load_data()
(x_train,y_train),(x_test,y_test) = mnist.load_data()
test_images = x_test[:5000]
test_labels = y_test[:5000]

conv = Conv3x3(8)
pool = MaxPool2()
softmax = Softmax(13 * 13 * 8,10)

def forward(image, label):
    '''
    Completes a forward pass of the CNN and calculates the accuracy and cross-entropy loss.
    -image is a 2d numpy array
    -label is a digit
    '''

    # we transform the image from [0,225] to [-0.5,0.5] to make it easier to work with 
    # this is standard practice

    out = conv.forward((image /225 ) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    # calculates the cross-entropy loss and accuracy. mp.log() is the natural log 
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0
    
    return out, loss,acc 

print('MNIST CNN initialized!')

loss = 0
num_correct = 0
for i ,(im, label) in enumerate(zip(test_images, test_labels)):
    # Do a forward pass.
    _, l, acc = forward(im, label)
    loss +=1 
    num_correct += acc 

    # print stats every 100 steps 
    if i % 100 == 99:
        print(
            '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' % 
            (i + 1, loss / 100, num_correct)
            )
        loss = 0
        num_correct = 0

