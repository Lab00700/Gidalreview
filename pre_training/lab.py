import math
import torch
import torch.optim as optim


w0=1
w1=2
w2=3

x0=2
x1=1

lr=0.1

f=1/(1+math.exp(-(w0*x0+w1*x1+w2)))

print(f**2)