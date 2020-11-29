import torch
import torch.nn as nn
import numpy as np
xy = np.loadtxt('C:\program_design\diabetes.csv',
                delimiter=',',
                dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])


# y_data=np.vstack(y_data).reshape(-1,1)
# y_data=torch.tensor(y_data)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(8, 6)
        self.linear2 = nn.Linear(6, 4)
        self.linear3 = nn.Linear(4, 1)
        self.activate = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()
criterion = nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# print(x_data.size())
# print(y_data.size())
for epoch in range(10000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    if (epoch + 1) % 100 == 0:
        print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()