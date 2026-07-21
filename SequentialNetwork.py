#Author: Barry Boyce Jr.
#Guide Followed/Credit: Luke Polson aka "Mr. P Solver" on Youtube 

#This program used an activation function (ReLU) to better train the model to make predictions closer to actual values. 
#The program adds more layers to the Neural Network, and allows for bigger matrices, as well as more rounds of training.
#This model may suffer from "overfitting
# "
import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np
import matplotlib.pyplot as plt

#Data
x = torch.tensor([[6,2], [5,2], [1,3], [7,6]]).float() 
y = torch.tensor([1,5,2,5]).float()                    

#Creates Matrices
M1 = nn.Linear(2,8,bias=False) 
print(M1)
M1(x)
print(M1(x))

M2 = nn.Linear(8,1,bias=False) 
M2(M1(x))
print(M2(M1(x)))
print(M2(M1(x)).shape)

#shapes matrices
print(M2(M1(x)).squeeze()) 
print(M2(M1(x)).squeeze().shape)


#Neural Network
class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(2,80)   #Changed 8 to 80 because the ReLU function allows for bigger matrices
        self.Matrix2 = nn.Linear(80,80)  #added middle layer (another matrix)
        self.Matrix3 = nn.Linear(80,1)
        self.R = nn.ReLU()                         #Added activation function 
    def forward(self,x):
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()
    
# able to store parameters for the neural network
f = MyNeuralNet()
for par in f.parameters():
    print(par)

# gives prediction from x data, however, at the moment its not accurate. So we must adjust the weights, by using teh loss function
prediction = f(x)
print(prediction)

loss = nn.MSELoss()
loss(y,prediction)
print(loss(y,prediction))


optimizer = SGD(f.parameters(), lr=0.001)

#adjusts the parameters over and over
losses = []
for _ in range(5000):          #how many epochs/rounds the model will be trained
    optimizer.zero_grad()      #ensures gradient resets each run
    loss_value = loss(f(x), y) #computes loss
    loss_value.backward()      #compute gradient
    optimizer.step()           #perform iteration using gradient above
    losses.append(loss_value.item())
print("Plot:")

plt.plot(losses)
plt.ylabel('Loss')
plt.xlabel('Epochs/Iterations')
plt.show() #plot for the loss as the neural network is ran each time 

#Now here are the predictions of the model
print(f(x))
# VS. the actual results
print(y)

#The predictions are now near perfectly accurate to the actual results. 
