import torch
x_data = [1.0, 2.0, 3.0,5.0,6.0,7.0,8.0,9.0,10.0]
y_data = [2.0, 3.0, 4.0,10.0,12.0,14.0,16.0,18.0,20.0]
w = torch.Tensor([1.0])
w.requires_grad = True


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2

print('predict(before training)',4,forward(4).item())

for epoch in range(3000):
    for x,y in zip(x_data,y_data):
        l=loss(x,y)
        l.backward()
        print('\tgrad:',x,y,w.grad.item())
        w.data=w.data-0.001*w.grad.data
        w.grad.data.zero_()
    print('progress:',epoch,l.item())
print('predict(after training)',4 ,forward(4).item())
