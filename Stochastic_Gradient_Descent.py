import matplotlib.pyplot as plt
import numpy as np
from torch._C import layout

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


#Stochastic gradient descent(随机梯度下降)
def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2


def gradient(xs, ys):
    return 2 * x * (x * w - y)


print('Predict (before training)', 4, forward(4))

cost_list = []
epoch_list = []

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w-=0.01*grad
        print('\tgrad:',x,y,grad)
        l=loss(x,y)
    print('progress:',epoch,'w=',w,'loss=',layout)
print('predict (after training)',4,forward(4))

