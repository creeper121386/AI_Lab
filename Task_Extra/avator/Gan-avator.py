import time
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch
from model2 import G, D
import PIL.Image as Image
import os
import numpy as np
# from torchvision.models import resnet18


epoch = 20
lr = 0.001
nz = 100        # 白噪声向量的长度
imgSize = 64
batchSize = 64
shotNum = 100
saveNum = 5
lossFunc = nn.BCELoss()
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
savePath = '/disk/unique/why/avator'
imgPath = '/disk/unique/why/faces'
# transform = T.Compose([T.Resize(400), T.RandomCrop(imgSize), T.ToTensor()])
transform = T.Compose([T.Resize(imgSize), T.ToTensor()])


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


def show(img):
    img = img.numpy()
    # img = T.ToPILImage(img)
    plt.imshow(np.transpose(img, (1, 2, 0)))
    # plt.axis('off')
    plt.show()


def train(Dnn, Gnn, trainLoader):
    z = torch.randn(batchSize, nz, 1, 1).to(device)
    for j in range(epoch):
        for i, (img, _) in enumerate(trainLoader, 0):
            Dnn.zero_grad()
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

            Gnn.zero_grad()
            D_loss = D_loss1 + D_loss2
            pred3 = Dnn(fake)
            G_loss = lossFunc(pred3, realLabel)
            G_loss.backward()
            G_optim.step()

            if not(i % shotNum):
                print('[{},epoch:{}]  G_loss:{},  D_loss:{}'.format(
                    i, j, G_loss, D_loss))
                torchvision.utils.save_image(
            	    Gnn(z).detach(), savePath+'/epoch{}-num{}.jpg'.format(j+1, i), normalize=True)
   		
                tmp = torch.randn(batchSize, nz, 1, 1).to(device)
                torchvision.utils.save_image(
                Gnn(tmp).detach(), savePath+'/rand-epoch{}-num{}.jpg'.format(j+1, i), normalize=True)
        if not(j % saveNum):
            torch.save(Gnn.state_dict(), "/disk/unique/why/model/Gnn-epoch{}.pkl".format(j+1))
            torch.save(Gnn.state_dict(), "/disk/unique/why/model/Dnn-epoch-{}.pkl".format(j+1))
    print('train finish')
    return Dnn, Gnn


Dnn = D()
Gnn = G()
Dnn.to(device)
Gnn.to(device)
trainData = whyDataset(imgPath)
trainLoader = DataLoader(dataset=trainData, batch_size=batchSize, shuffle=True, drop_last=True)
D_optim = optim.Adam(Dnn.parameters(), lr=lr, betas=(0.5, 0.999))
G_optim = optim.Adam(Gnn.parameters(), lr=lr, betas=(0.5, 0.999))
Dnn, Gnn = train(Dnn, Gnn, trainLoader)

torch.save(Gnn.state_dict(), "/disk/unique/why/model/Gnn-epoch{}.pkl".format(epoch))
torch.save(Gnn.state_dict(), "/disk/unique/why/model/Dnn-epoch-{}.pkl".format(epoch))
