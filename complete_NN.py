# A complete NN
import numpy as np 

def sigmoid(x,deri=False):
    if deri==True:
        fx = sigmoid(x,deri=False)
        return fx * (1 - fx)
    else:
        return 1/(1+np.exp(-x))

def mse_loss(y_true,y_pred):
    return ((y_true-y_pred)**2).mean()

class NeuralNetwork:
    '''
    - 2 inputs 
    - a hidden layer with 2 neurons (h1,h2)
    - an output layer with  1 nuron (o1)
    '''
    def __init__(self): # specifiying the variables that will be used 
        # weights
        # weights are the like the gears which you modify in order to get the solution that you need
        # they specify how important it is
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Biases 
        # it determines how sensitive it is to the data inputted
        self.b1 = np.random.random()
        self.b2 = np.random.random()
        self.b3 = np.random.random()

    def feedforward(self,x):
        # x is a numpy array with 2 elements.
        # formula:  f(sigmoid)((w1*x1) + (w2*x2) + b)
        h1 = sigmoid((self.w1*x[0] + self.w2*x[1]) + self.b1)#using the formula and converting it into a number between 0 and 1
        h2 = sigmoid((self.w3*x[0]+ self.w4*x[1]) + self.b2)
        o1 = sigmoid((self.w5*h1+ self.w6*h2) + self.b3)

        return o1
    
    def train(self, data, all_y_trues):
        '''
        - data is a (n x 2) numpy array , n = # of samples in the daataset.
        - all_y_trues is a numpy array with n elements.
          Elements in all_y_trues correspond to those in data
        '''
        learn_rate = 0.1 # by how much will it change in its next trial 
        epochs = 1000 # how many trials

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # -- Do a feedforward (we'll need these values later)
                # computing the activations for hidden layer
                sum_h1 = (self.w1*x[0]+ self.w2*x[1]) + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = (self.w3 *x[0] + self.w4 * x[1]) + self.b2 
                h2 = sigmoid(sum_h2)

                sum_o1 = (self.w5* h1 + self.w6* h2) + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1 

                # ---Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / parital w1"
                # It tells you how much the prediction pushes the loss up or dow
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                # back propogation through the output layer
                d_ypred_d_w5 = h1 * sigmoid(sum_o1, deri=True)
                d_ypred_d_w6 = h2 * sigmoid(sum_o1, deri=True)
                d_ypred_d_b3 =  sigmoid(sum_o1, deri=True)

                d_ypred_d_h1 = self.w5 * sigmoid(sum_o1, deri=True)
                d_ypred_d_h2 = self.w6 * sigmoid(sum_o1, deri=True)

                # neuron h1 
                # back propogation through the hidden layer 1
                d_h1_d_w1 = x[0] * sigmoid(sum_h1, deri=True)
                d_h1_d_w2 = x[1] * sigmoid(sum_h1, deri=True)
                d_h1_d_b1 = sigmoid(sum_o1, deri=True)

                # neuron h2 
                # back propogation through the hidden layer 2
                d_h2_d_w3 = x[0] * sigmoid(sum_h2, deri=True)
                d_h2_d_w4 = x[1] * sigmoid(sum_h2, deri=True)
                d_h2_d_b2 = sigmoid(sum_h2, deri=True)
                # --- Update weights and biases
                # Neuron h1 
                # changing the weights of hidden layer 1 
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1 

                # Neuron h2 
                # changing the weights of hidden layer 2 
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1 
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

                # --- Calculate total loss at the end of each epoch 
                # - Calculating the loss of each epoch and printing it every 10 epochs
                if epoch % 10 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1 , data)
                    loss = mse_loss(all_y_trues, y_preds)
                    print("Epoch %d loss: %.3f" % (epoch, loss))


# Define dataset
# this data is normalised by some constant or formula
# i got the data off of some guy on google 
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])

all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Train our neural network
network = NeuralNetwork()
network.train(data, all_y_trues)
np.random.seed(1)

# make some predictions 
emily = np.array([-7, -3]) # 128 pounds , 63 inches
frank = np.array([20,2]) # 155 pounds, 68 inches
print("Emily: %.3f"% network.feedforward(emily))

if network.feedforward(emily) >= 0.5:
    print("emily is a girl")
else:
    print("emily is a boy")

print("Frank: % .3f" % network.feedforward(frank))

if network.feedforward(frank) >= 0.5:
    print("Frank is a girl")
else:
    print("Frank is a boy")