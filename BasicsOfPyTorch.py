import torch
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt
import time

print("numpy array between 0 & 1: ")
n = np.linspace(0,1,5)
print(n)
print("\n")
 
print("pytorch array between 0 & 1: ")
t = torch.linspace(0,1,5)
print(t)
print("\n")

n = np.arange(48)
print("numpy array of 48 variables: ")
print(n)
print("\n")

t = torch.arange(48)
print("torch array of 48 variables: ")
print(t)
print("\n")

n = np.arange(48).reshape(3,4,4)
print("numpy 3D array of 48 variables: ")
print(n)
print("\n")

t = torch.arange(48).reshape(3,4,4)
print("torch 3D array of 48 variables: ")
print(t)
print("\n")


print("multiplication of 2 numpy arrays that have equal dimensions: ")
print("a = [1 2]")
print("b = [3 4]")

a = np.array([1,2])
b = np.array([3,4])
print("product = ", a*b)

# Broadcasting Rules: 
# Arrays are able to be multiplied, divided, added, subtracted if their dimensions are equal, or of they follow the
# broadcasting rules which are 
# 1. They are equal 
# 2. One of them is 1
# Example that can be multiplied: 
# Shape 1: (3,1,8)
# Shape 2: (3,5,1)
# Because 3 = 3, There is a 1 in the second place with the 5, and there is a 1 in the third place with the 8