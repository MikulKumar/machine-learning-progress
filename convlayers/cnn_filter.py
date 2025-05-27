import numpy as np 

class Conv3x3:
    # A convolution layer using 3x3 filters.

    def __init__(self, num_filters):
        self.num_filters = num_filters

        # filters is a 3d array with dimensions (num_filters, 3, 3)
        # we divide by 9 to reduce the variance of our initial values
        self.filters = np.random.randn(num_filters, 3, 3) / 9
    def iterate_regions(self,image):
        '''
        Generates all possible 3x3 image regions using valid padding.
        - image is a 2d numpy array 
        '''

        h,w = image.shape 

        for i in range(h-2):
            for j in range(w-2):
                im_region = image[i:(i+3), j:(j+3)]
                yield im_region, i, j

    def forward(self, input):
        '''
        Performs a forward pass of the conv layer using the given input.
        Returns a 3d numpy array with dimensions (h,w, num_filters).
        - input is a 2d numpy array
        '''

        h,w = input.shape
        output = np.zeros((h-2, w-2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i,j] = np.sum(im_region * self.filters, axis=(1,2))
        
        return output
        
from tensorflow.keras.datasets import mnist 
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0

conv = Conv3x3(8)
output = conv.forward(train_images[0])
print(output.shape)
'''
iterate_regions() is a helper generator method that yields all valid 3x3 image regions for us.
This will be useful for implementing the backwards portion of this class later on.

The line of code that actully performs the convolutions is mentioned below.
    output[i,j] = np.sum(im_region * self.filters, axis=(1,2))

    We have im_region, a 3x3 array containg the relevant image region.

    We have self.filters, a 3d array.

    We do im_region * self.filters, which uses numpy's broadcasting feature to 
    element-wise multiply the two arrays. The result is a 3d array with the same dimensions as self.filters.

    We np.sum() the result of the previous step using axis = (1,2), which produces a 1d array of length 
    num_filters where each element contains the convolution result for the corresponding filter.

    We assign the result to output[i,j], which contains convolution results for pixel (i,j) in the output.
'''