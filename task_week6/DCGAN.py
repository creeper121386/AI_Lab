import torchvision.transforms as T
import torchvision.datasets as dSet
import torch.nn as nn
import torch
ndf = 64
ngf = 64
lr = 0.01
cuda = True
batchSize = 4
transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainData = dSet.MNIST(root='/run/media/why/DATA/why的程序测试/AI_Lab/DataSet',
                       train=True, transform=transform, download=True)      # size = 28*28
trainLoader = torch.utils.data.DataLoader(dataset=trainData,
                                          batch_size=batchSize,
                                          shuffle=True)


class D(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True))
        self.layer3 = nn.Sequential(
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True))


for x in trainLoader:
    pass