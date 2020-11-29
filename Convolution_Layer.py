import torch
from torch import optim
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
# in_channels, out_channels = 5, 10
# width = 100, height = 100
# kernel_size = 3
# batch_size = 1

# input = torch.randn(batch_size, in_channels, width, height)
# conv_layer = torch.nn.Conv2d(in_channels,
#                              out_channels,
#                              kernel_size=kernel_size)
# output=conv_layer(input)




# # input=torch.tensor([1,1,1,1,1,
# #        1,1,1,1,1,
# #        1,1,1,1,1,
# #        1,1,1,1,1,
# #        1,1,1,1,1,])#5X5
# # input=torch.Tensor.view(1,1,5,5)
# input = torch.randn(1, 1, 5, 5)
# #（batch-size,in-channels,width,height)
# conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2, bias=True)
# kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)
#                          #out-channels,in-channels,width,height
# conv_layer.weight.data = kernel.data 
# output = conv_layer(input) 
# print(output.data)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,10,kernel_size=5)
        self.conv2=nn.Conv2d(10,20,kernel_size=5)
        self.pooling=nn.MaxPool2d(2)
        self.linear=nn.Linear(320,10)

    def forward(self,x):
        batch_size=x.size(0)
        x=F.relu(self.pooling(self.conv1(x)))
        x=F.relu(self.pooling(self.conv2(x)))
        x=x.view(x.size(0),-1)
        x=self.linear(x)
        return x

model=Net()
batch_size=64
device = torch.device("cuda:0" if torch.cuda. is_available() else"cpu")
model.to(device)

transform = transforms.Compose([  #归一化
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])

train_dataset = datasets.MNIST(root='C:\\Users\\ysgnb\\VSC\\data',
                               train=True,
                               download=True,
                               transform=transform)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='C:\\Users\\ysgnb\\VSC\\data',
                              train=False,
                              transform=transform,
                              download=True)

test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=.01, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs,target=inputs.to(device),target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            #一个batch的平均loss
            print('[%d,%5d] loss:%.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = .0

def test():
    correct = 0
    total = 0

    for data in test_loader:
            images, labels = data
            images,labels=images.to(device),labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()    
    print('Accuracy on test set is %d%%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()