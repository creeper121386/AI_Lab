import torchvision.datasets as dataset
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as D
import torch.optim as optim
import torch.nn as nn
import numpy as np
in_chanl = out_chanl = 1    # 识别器的输入输出维数
batchSize = 8
size = 32
nz = 100
y1 = 10 # y中包含的离散变量的个数，假设表示one-hot编码的label
y2 = 2  # y中包含连续变量的个数，假设表示笔画的粗细和字体的倾角


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        layer1 = nn.Sequential(
            nn.Conv2d(in_chanl, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
       )
        layer2 = nn.Sequential(
            nn.Linear(128*(size//4)**2, 1024),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, out_chanl+y1+y2)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = x.view(-1, 128*(size//4)**2)
        x = self.layer2(x)
        a = F.sigmoid(x[:, out_chanl])
        b = x[:, out_chanl:out_chanl+y2]
        c = x[:, out_chanl+y2:]
        return a, b, c


class G(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        layer1 = nn.Sequential(
            nn.Linear(nz+y1+y2, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128*(size//4)**2),
            nn.BatchNorm2d(128*(size//4)**2, 128*(size//4)**2),
            nn.ReLU()
       )
        layer2 = nn.Sequential(
            nn.Linear(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_chanl, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, x):









class InfoGAN(object):
    def __init__(self, GNet, DNet):
        self.trainData = dataset.MNIST(root='/run/media/why/DATA/why的程序   测试/AI_Lab/DataSet', train=True,
                         transform=T.Compose([T.Resize(self.imgSize), T.ToTensor()]), download=True)      #     size = 28*28
        self.trainLoader = D.DataLoader(dataset=self.trainData, batch_size=self.batchSize, shuffle=True)
        self.device = torch.device("cuda")




print("test over.")
