import matplotlib.pyplot as plt
import numpy as np

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

def forward(x):
    return x * w


def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y)**2
        return cost / len(xs)

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


print('Predict (before training)', 4, forward(4))

cost_list=[]
epoch_list=[]

for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    print('Epoch:', epoch, 'w=', w, 'loss=', cost_val)
    cost_list.append(cost_val)
    epoch_list.append(epoch)
print('Predict (after training)', 4, forward(4))

#draw the graph
plt.plot(epoch_list,cost_list)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

#梯度下降的优点在于可以并行计算