import torch
import torchvision
import torchvision.transforms as trans
import torch.nn as nn
import torch.optim as optim 

#定义设备的选择
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: '+str(device))

#定义网络结构
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1=nn.Sequential(
            #输入层数1，输出层数6，卷积核大小5，步长1，padding2  (b, 1, 28, 28) => (b, 6, 28, 28)
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) #(b, 6, 28, 28) => (b, 6, 14, 14)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(6, 16, 5), #(b, 6, 14, 14) => (b, 16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) #(b, 16, 5, 5)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(16, 120, 5), #(b, 120, 1, 1)
            nn.ReLU()
        )
        self.fc1=nn.Sequential(
            nn.Linear(120, 84),#(b, 84)
            nn.ReLU()
        )
        self.fc2=nn.Sequential(
            nn.Linear(84, 10) #(b, 10)
        )
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=x.view(x.size()[0], -1) #变换维度，将x展成一维
        x=self.fc1(x)
        x=self.fc2(x)
        return x

#超参数的设置
EPOCHE=8
BATCH_SIZE=64
LR=0.001

#定义数据预处理模式
transform=trans.ToTensor()

#定义训练数据集
trainset=torchvision.datasets.MNIST(
    root='./data/',
    train=True,
    transform=transform,
    download=True,
)

# 定义训练批数据
trainloader=torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

#定义测试数据集
testset=torchvision.datasets.MNIST(
    root='./data/',
    train=False,
    transform=transform,
    download=True,
)

#定义测试批数据
testloader=torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

#转移设备，定义损失函数和优化方法
net=LeNet().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(), lr=LR, momentum=0.9)

#训练过程
if __name__ == "__main__":
    for epoch in range(1, EPOCHE+1):
        sum_loss=0.0
        #读取数据
        for i, data in enumerate(trainloader):
            inputs, lables=data
            inputs=inputs.to(device)
            lables=lables.to(device)
            #梯度清零
            optimizer.zero_grad()
            #前向和逆向传播
            outputs=net(inputs)
            loss=criterion(outputs, lables)
            loss.backward()
            optimizer.step()
            #100个batch输出一次loss
            sum_loss+=loss.item()
            if i%100==99:
                print('epoch: %d, batch: %d, loss: %05f' % (epoch, i+1, sum_loss/100))
                sum_loss=0.0
            #每跑完一个epoch进行一次测试
        with torch.no_grad():
            correct=0
            total=0
            for data in testloader:
                images, lables=data
                images=images.to(device)
                lables=lables.to(device)
                outputs=net(images)
                _, predicted=torch.max(outputs.data, 1)
                total+=lables.size(0)
                correct+=(predicted==lables).sum()
            print('correct: '+str(correct))
            print('total: '+str(total))
            print('epoch %d accuracy: %d%%' % (epoch, (100 * correct // total)))
    torch.save(net.state_dict(), 'para_data.pth')