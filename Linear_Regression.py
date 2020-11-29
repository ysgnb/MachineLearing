import torch
x_data = torch.Tensor([[1.0], [2.0], [3.0], [5.0], [6.0], [7.0], [8.0], [9.0],
                      [10.0]])
y_data = torch.Tensor([[2.0], [3.0], [4.0], [10.0], [12.0], [14.0], [16.0],
                      [18.0], [20.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()  #调用父类的构造函数
        self.linear = torch.nn.Linear(1,1,bias=False)  #构造对象，wx+b形式的
        #这个linear也是继承自Module的
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()

criterion = torch.nn.MSELoss(size_average=False)
#采用MSE损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(3000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()  #梯度清零
    loss.backward()  #后向传播
    optimizer.step()  #梯度更新

print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred=', y_test.data)
