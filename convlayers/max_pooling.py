# implementing Pooling from scratch using numpy
import numpy as np 

class MaxPool2:
    # A max pooling layer using a pool size of 2

    def iterate_regions(self, image):
        '''
        Generates non-overlapping 2x2 image regions to pool over.
        - image is a 2d numpy array (black and white)
        '''
        h,w, _ = image.shape
        new_h = h//2 
        new_w = w//2 

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i*2):(i * 2+2), (j*2):(j * 2+2)]
                yield im_region, i , j # Returns the 2×2 region and its coordinates
                '''
                For example, when i=1, j=2:
                We take pixels from rows 2-3 and columns 4-5
                (1*2):(1*2+2) means 2:4 (rows 2 and 3)
                (2*2):(2*2+2) means 4:6 (columns 4 and 5)
                '''


    def forward(self, input):
        '''
        performs a forward pass of the maxpool layer using the given input.
        Returns a 3d numpy array with dimension ( h / 2,w / 2, num_filters).
        - input is a 3d numpy array with dimensions (h,w, num_filters)
        '''

        h,w, num_filters = input.shape
        output = np.zeros((h // 2, w//2, num_filters))

        for im_region, i ,j in self.iterate_regions(input):
            output[i,j] = np.amax(im_region, axis=(0,1))

        return output

'''
This class works similarly to the Conv3x3 class we implemented previously. The critical is :
for im_region, i ,j in self.iterate_regions(input):
to find the max from a given image region, we use amax(), numpy's array ax method. We set axis= (0,1) because we only wan to maximize over 
the first two dimensions, height and width, and not the third, num_filters.
'''