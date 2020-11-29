import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
x_data=torch.Tensor([[1.0],[2.0],[3.0]])
y_data=torch.Tensor([[0],[0],[1]])#这边作为分类

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel,self).__init__()
        self.linear=torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred=F.sigmoid(self.linear(x))
        return y_pred

model=LogisticRegressionModel()

criterion=torch.nn.BCELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameters(),lr=1e-3)

for epoch in range(20000):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,'loss=',loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x=np.linspace(0,10,200)#0--10取200个点
x_t=torch.Tensor(x).view((200,1)) #转成200x1的形式
y_t=model(x_t)
y=y_t.data.numpy()
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')    #画一条中间线
plt.xlabel('hours')
plt.ylabel('possibility of pass')
plt.grid()  #显示网格线
plt.show()
    