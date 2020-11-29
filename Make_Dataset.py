import torch
import numpy as np 
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class anydataset(Dataset):
    def __init__(self):
        xy=np.loadtxt('filepath',delimiter=',',dtype=np.float32)
        self.len=xy.shape[0]
        self.x_data=torch.from_numpy(xy[:,:-1])
        self.y_data=torch.from_numpy(xy[:,:[-1]])
    
    def __getitem__(self, index: int):
        return self.x_data[index],self.y_data[index]

    def __len__(self) :
        return self.len


dataset=anydataset('filepath')
train_loader=DataLoader(dataset=dataset,batch_size=32,shuffle=True)

critirion=nn.BCELoss(size_average=True)
optimizer=torch.optim.SGD()