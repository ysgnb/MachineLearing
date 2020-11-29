import torch
x_data = [1.0, 2.0, 3.0,5.0,6.0,7.0,8.0,9.0,10.0]
y_data = [2.0, 3.0, 4.0,10.0,12.0,14.0,16.0,18.0,20.0]
w1 = torch.Tensor([1.0])
w1.requires_grad = True
w2 = torch.Tensor([1.0])
w2.requires_grad = True
b = torch.Tensor([1.0])
b.requires_grad = True


def forward(x):
    y_pred = w1 * x * x + w2 * x + b
    return y_pred


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2


print('predict(before training)', 4, forward(4).item())
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, w1.grad.item(),w2.grad.item(),b.grad.item())

        w1.data = w1.data - 0.0001 * w1.grad.data
        w1.grad.data.zero_()
        w2.data = w2.data - 0.0001 * w2.grad.data
        w1.grad.data.zero_()
        b.data = w1.data - 0.0001 * b.grad.data
        b.grad.data.zero_()
        #0.046496473252773285 未清零b
        #0.022994259372353554 清零b

    print('progress:', epoch, l.item())
print('predict(after training)', 4, forward(4).item(),w1.item(),w2.item(),b.item())
