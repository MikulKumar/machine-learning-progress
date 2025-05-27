# making a neural network from scratch using numpy only
import numpy as np 
x = np.array([[0,0,1],
              [0,1,0],
              [1,0,0],
              [0,1,0]]) 

y = np.array([[0,1,0,1]]).T

#sigmoid function
def non_lin(x, deri=False):
    if deri == True:
        return x*(1-x) # used when we need to learn from errors / backward pass calculations
    return 1/(1+np.exp(-x)) # used to convert raw data to probability / used to make predictions


np.random.seed(1)# creates random numbers , which arent actually random but follow a sequence
# these numbers are used for the small tasks , weight initialization, Dropout, Data_splitting

syn0 = 2*np.random.random((3,1)) - 1 # these are the weights 
eta = [1,0.1,0.001,0.0001,0.00001,0.000001] # learning rates

for i in range(0,6,1):
    for o in range(10000):

        # forward propogation / predictions
        l0 = x 
        l1 = non_lin(np.dot(l0,syn0))

        # how much did i miss, how far was i from the actual output
        l1_error = y - l1

        # multiply how much we missed by the 
        # slope of the sigmoid at the values in l1
        l1_delta = l1_error * non_lin(l1,True)

        #update the weights 
        syn0 += eta[i] * np.dot(l0.T , l1_delta)  # shape: (n, m) ⋅ (m, 1) → (n, 1)
        #you move your weights in the direction that most reduces your error.
    
    print("Output after Training\n", l1)
    