import numpy as np 

class Softmax:
    # A standard fully-connected layer with softmax activation.

    def __init__(self, input_len, nodes):
        # we divide by input_len to reduce the variance of out intial value 
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        '''
        Performs a forward pass of the softmax layer using the given input. 
        Returns a 1d numpy array containing the respective probability value
        - input can be any array with any dimensions.
        '''
        self.last_input_shape = input.shape 
        input = input.flatten()
        self.last_input = input
        input_len, nodes = self.weights.shape 

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals 
        exp = np.exp(totals)
        return exp/ np.sum(exp, axis=0)

    def backprop():
        '''
        performs a backward pass of the softmax layer.
        Returns the loss gradient for this layer's inputs.
        - d_l_d_out is the loss gradient for this layer's outputs.
        '''

        # we know only 1 element of d_l_d_out will be nonzero
        for i, gradient in enumerate(d_l_d_out):
            if gradient == 0:
                continue 
            
            # e^totals 
            t_exp = np.exp(self.last_totals)

            # sum of all e^totals 
            s = np.sum(t_exp)

            # gradient of out[i] against totals
            d_out_d_t = -t_exp[i] * t_exp / (s ** 2)
            d_out_d_t[i] = t_exp[i] * (s-t_exp[i]) / (s ** 2)
'''
We flatten() the input to make it easier to work with, since we no longer need its shape.

np.dot() multiplies input and self.weights element-wise and then sums the results.

np.exp() calculates the exponentials used for softmax.
'''
