# my version
import numpy as np 

def mse_loss(y_true,y_pred):
    n  = len(y_true)
    loss = 0
    for i in range(0,n):
        loss += 1/n*((y_true[i]-y_pred[i])**2)
    return loss


y_true = np.array([1,0,0,1])
y_pred = np.array([0,0,0,0])
print(mse_loss(y_true,y_pred))

# faster method 

def mse_LOSS(y_true,y_pred):
    return ((y_true-y_pred)**2).mean()

y_tru = np.array([1,0,0,1])
y_pre = np.array([0,0,0,0])

print(mse_LOSS(y_tru, y_pre))