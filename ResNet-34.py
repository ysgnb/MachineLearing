import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

#定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ' + str(device))

#定义ResNet的一个block
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if in_channel != out_channel or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride, 0),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
            out = self.left(x)
            x = self.shortcut(x)
            out += x
            out = F.relu(out)
            return out

#定义ResNet
class ResNet(nn.Module):
    def __init__(self, ResBLock):
        super(ResNet, self).__init__()
        #(b, 3, 32, 32) => (b, 64, 32, 32)
        self.ready = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        #(b, 64, 32, 32) => (b, 64, 32, 32)
        self.layer1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBLock(64, 64)
        )
        #(b, 64, 32, 32) => (b, 128, 32, 32)
        self.layer2 = nn.Sequential(
            ResBLock(64, 128),
            ResBLock(128, 128),
            ResBLock(128, 128), 
            ResBLock(128, 128)
        )
        #(b, 128, 32, 32) => (b, 256, 16, 16)
        self.layer3 = nn.Sequential(
            ResBLock(128, 256, 2),
            ResBLock(256, 256),
            ResBLock(256, 256),
            ResBLock(256, 256),
            ResBLock(256, 256),
            ResBLock(256, 256)
        )
        #(b, 256, 16, 16) => (b, 512, 16, 16)
        self.layer4 = nn.Sequential(
            ResBLock(256, 512),
            ResBLock(512, 512),
            ResBLock(512, 512)
        )
        #(b, 512, 16, 16) => (b, 512, 8, 8)
        self.pool = nn.Sequential(
            nn.AvgPool2d(2, 2)
        )
        #(b, 512 * 8 * 8) => (b, 10)
        self.fc = nn.Sequential(
            nn.Linear(512 * 8 * 8, 10)
        )

    def forward(self, x):
        x = self.ready(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#超参数的设置
EPOCHE = 120
BATCH_SIZE = 64
LR = 1e-2

#定义预处理
transform = transforms.ToTensor()

#定义训练数据集
train_set = torchvision.datasets.CIFAR10(
    root = './data/',
    train = True,
    transform = transform,
    download = True
)

#定义训练批数据
trainloader = torch.utils.data.DataLoader(
    train_set,
    batch_size = BATCH_SIZE,
    shuffle = True
)

#定义测试数据集
test_set = torchvision.datasets.CIFAR10(
    root = './data/',
    train = False,
    transform = transform,
    download = True
)

#定义测试批数据
testlaoder = torch.utils.data.DataLoader(
    test_set,
    batch_size = BATCH_SIZE,
    shuffle = False
)

#转移设备，定义损失函数和优化方法
net = ResNet(ResBlock).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = LR, momentum = 0.9)

#训练过程
if __name__ == "__main__":
    for epoch in range(1, EPOCHE + 1):
        sum_loss = 0.0
        # 读取数据
        for i, data in enumerate(trainloader):
            inputs, lables = data
            inputs = inputs.to(device)
            lables = lables.to(device)
            #梯度清零
            optimizer.zero_grad()
            #前向传播与反向传播
            outputs = net(inputs)
            loss = criterion(outputs, lables)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            #每100个batch输出loss
            if i % 1 == 0:
                print('epoch: %d, batch: %d, loss: %05f' % (epoch, i + 1, sum_loss / 1))
                sum_loss = 0.0
        #每跑完一个epoch进行一个test
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                inputs, lables = data
                inputs = inputs.to(device)
                lables = lables.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += lables.size(0)
                correct += (predicted == lables).sum()
            print('epoch %d accuracy: %d%%' % (epoch, (100 * correct // total)))
    torch.save(net.state_dict(), 'ResNet_para.pth')