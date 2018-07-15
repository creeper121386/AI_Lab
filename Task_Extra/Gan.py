import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch
import torchvision
import matplotlib.pyplot as plt
from model import G, D
import PIL.Image as Image
import os
import numpy as np


epoch = 20
lr = 0.01
nz = 100        # 白噪声向量的长度
imgSize = 400
batchSize = 16
lossFunc = nn.BCELoss()
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
savePath = '/run/media/why/DATA/why的程序测试/AI_Lab/DataSet/sunset'
imgPath = '/run/media/why/DATA/why的程序测试/AI_Lab/DataSet/爬虫_landscape'
transform = T.Compose([T.Resize(400), T.RandomCrop(imgSize), T.ToTensor()])


class whyDataset(Dataset):
    def __init__(self, imgPath):
        self.path = imgPath
        self.fileList = os.listdir(self.path)

    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, ix):
        path = self.path + '/' + self.fileList[ix]
        img = Image.open(path)
        img = transform(img)
        return img, 1   # real label


def show():
    for i, (img, _) in enumerate(trainData, 0):
        img = img.numpy()
        # img = T.ToPILImage(img)
        plt.imshow(np.transpose(img, (1, 2, 0)))
        # plt.axis('off')
        plt.show()


Dnn = D()
Gnn = G()
trainData = whyDataset(imgPath)
trainLoader = DataLoader(dataset=trainData, batch_size=batchSize, shuffle=True)
if cuda:
    Dnn = Dnn.cuda()
    Gnn = Gnn.cuda()
D_optim = optim.Adam(Dnn.parameters(), lr=lr, betas=(0.9, 0.999))
G_optim = optim.Adam(Gnn.parameters(), lr=lr, betas=(0.9, 0.999))

z = torch.randn(batchSize, nz, 1, 1).to(device)
for j in range(epoch):
    for i, (img, _) in enumerate(trainData, 0):
        Dnn.zero_grad()
        Gnn.zero_grad()
        real = img.to(device)
        pred1 = Dnn(real)
        realLabel = torch.full((batchSize, ), 1, device=device)
        D_loss1 = lossFunc(pred1, realLabel)
        D_loss1.backward()

        noise = torch.randn(batchSize, nz, 1, 1).to(device)
        fakeLabel = torch.full((batchSize, ), 0, device=device)
        fake = Gnn(noise)
        pred2 = Dnn(fake.detach())
        D_loss2 = lossFunc(pred2, fakeLabel)
        D_loss2.backward()
        D_optim.step()

        D_loss = D_loss1 + D_loss2
        pred3 = Dnn(fake)
        G_loss = lossFunc(pred3, realLabel)
        G_loss.backward()
        G_optim.step()

        if not(j % 20):
        print('trainning[num{},epoch:{}] G_loss:{},D_loss:{}'.format(j, i, G_loss, D_loss))

test = Gnn(z)

