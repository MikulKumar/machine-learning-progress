# code a Neuron
import numpy as np 

def sigmoid (x):
    return 1/(1+np.exp(-x))

class neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias 
    
    def feedforward(self, inputs):
        neu = np.dot(self.weights, inputs) + self.bias 
        return sigmoid(neu)
    

i = np.array([2,3]) # input
bias = 4
weigh = np.array([0,1]) # weights 
n  = neuron(weigh, bias)

print(n.feedforward(i))
# 0.9990889488055994


