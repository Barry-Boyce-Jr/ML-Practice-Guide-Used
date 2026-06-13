#Author: Barry Boyce Jr.
#Guide Followed/Credit: "Mr. P Solver" on Youtube

import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np
import matplotlib.pyplot as plt

x = torch.tensor([[6,2], [5,2], [1,3], [7,6]]).float() #independent variable 
y = torch.tensor([1,5,2,5]).float()                    #dependent variable 

# The goal for this file is to create a model/neural network that can figure out/predict what the algorithm is that 
# makes the independent variable go to the dependent variable. (for example, how does [6,2] go to 1, how does [5,2] go to 5, etc.)

#First take each element of x and multiply it by an 8x2 matrix. Then take the results and multiply each element by an 8x1 matrix (these are the layers of the neural network) (8 is an arbitrary choice, could be 9, 10, etc.)

M1 = nn.Linear(2,8,bias=False) #creates matrix 1. #takes in a 2 dimensional vector and returns an 8 dimensional vector. No bias. 
print(M1)
M1(x)
print(M1(x))

M2 = nn.Linear(8,1,bias=False) #creates matrix 2 which will be chained with matrix 1. #takes in a 8 dimensionla vector and returns a 1 dimensional vector. No bias. 
M2(M1(x))
print(M2(M1(x)))
print(M2(M1(x)).shape)

print(M2(M1(x)).squeeze()) #This removes the uneeded dimension so that the shape of the tensor better resembles the desired output of y (which is all in one line, not multiple)
print(M2(M1(x)).squeeze().shape)


# Now to train the network so that the values after going through the matrices resemble the values of y (what we want the model to predict)
class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(2,8,bias=False)
        self.Matrix2 = nn.Linear(8,1,bias=False)
    def forward(self,x):
        x = self.Matrix1(x)
        x = self.Matrix2(x)
        return x.squeeze()
    
# able to store parameters for the neural network
f = MyNeuralNet()
for par in f.parameters():
    print(par)

# givees prediciton from x data, hwoever, at the moment its not accurate. So we must adjust the weights, by using teh loss function
prediction = f(x)
print(prediction)

loss = nn.MSELoss()
loss(y,prediction)
print(loss(y,prediction))


optimizer = SGD(f.parameters(), lr=0.001)

#adjusts the parameters over and over
losses = []
for _ in range(50):
    optimizer.zero_grad()      #ensures gradient resets each run
    loss_value = loss(f(x), y) #computes loss
    loss_value.backward()      #compute gradient
    optimizer.step()           #perform iteration using gradient above
    losses.append(loss_value.item())
print("Plot:")

plt.plot(losses)
plt.ylabel('Loss')
plt.xlabel('Epochs/Iterations')
plt.show() #plot for the loss as the nueral network is ran each time 

#Now here are the predictions of the model
print(f(x))
# VS. the actual results
print(y)

# the results are not exact and are actually not too close, but they are as close as the model could predict for the simple data and neural netowrk given. As networks and data get more complicated the more accurate the model can predict.
