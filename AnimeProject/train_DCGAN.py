'''
[LL trainning log] epoch==500, size==64*64
epoch == 420时效果下降。epoch 200~400之间效果基本一致。
'''
import time
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch
from model64 import G, D
import PIL.Image as Image
import os
import numpy as np
from config import *
#import matplotlib.pyplot as plt
# from torchvision.models import resnet18
################## Func ########################
save_all_model = False
lossFunc = nn.BCELoss()
device = torch.device(
    'cuda') if torch.cuda.is_available() and cuda else torch.device('cpu')
transform = T.Compose([T.Resize(imgSize), T.ToTensor()])
# transform = T.Compose([T.Resize(400), T.RandomCrop(imgSize), T.ToTensor()])


class whyDataset(Dataset):
    def __init__(self, imgPath):
        self.path = imgPath
        self.fileList = os.listdir(self.path)

    def __len__(self):
        return len(self.fileList) if not sampleNum else sampleNum

    def __getitem__(self, ix):
        path = self.path + '/' + self.fileList[ix]
        img = Image.open(path)
        bg = Image.new("RGB", img.size, (255,255,255))
        bg.paste(img,img)
        bg = transform(bg)
        return bg, 1   # real label


def show(data):
    for img in data:
        img = img[0]
        img = img.numpy()
        # img = T.ToPILImage(img)
        plt.imshow(np.transpose(img, (1, 2, 0)))
        # plt.axis('off')
        plt.show()


def saveModel(Dnn, Gnn, modelPath, num, save_all=False):
    if save_all:
        torch.save(Gnn, modelPath+"/Gnn-epoch{}.pkl".format(num))
        torch.save(Dnn, modelPath+"/Dnn-epoch{}.pkl".format(num))
    else:
        torch.save(Gnn.state_dict(), modelPath+"/Gnn-epoch{}.pkl".format(num))
        torch.save(Dnn.state_dict(), modelPath+"/Dnn-epoch{}.pkl".format(num))


def write(path, G, D):
    G = np.array(G)
    D = np.array(D)
    np.savetxt(path+'/Gdata.txt', G)
    np.savetxt(path+'/Ddata.txt', D)


def train(Dnn, Gnn):
    trainLoader = DataLoader(
        dataset=trainData, batch_size=batchSize, shuffle=True, drop_last=True)
    D_optim = optim.Adam(Dnn.parameters(), lr=lr, betas=(0.5, 0.999))
    G_optim = optim.Adam(Gnn.parameters(), lr=lr, betas=(0.5, 0.999))

    z = torch.randn(batchSize, nz, 1, 1).to(device)
    D_lossList_all = []
    G_lossList_all = []
    for j in range(epoch):
        D_lossList = []
        G_lossList = []
        for i, (img, _) in enumerate(trainLoader, 0):
            Dnn.zero_grad()
            real = img.to(device)
            pred1 = Dnn(real)
            # Edit here to change the REAL&FAKE LABEL!
            realLabel = torch.full((batchSize, ), 1, device=device)
            fakeLabel = torch.full((batchSize, ), 0, device=device)
            D_loss1 = lossFunc(pred1, realLabel)
            D_loss1.backward()

            noise = torch.randn(batchSize, nz, 1, 1).to(device)
            fake = Gnn(noise)
            pred2 = Dnn(fake.detach())
            D_loss2 = lossFunc(pred2, fakeLabel)
            D_loss2.backward()
            D_optim.step()
            D_loss = D_loss1 + D_loss2
            D_lossList.append(float(D_loss))

            Gnn.zero_grad()
            pred3 = Dnn(fake)
            G_loss = lossFunc(pred3, realLabel)
            G_lossList.append(float(G_loss))
            G_loss.backward()
            G_optim.step()

            if not(i % saveImgNum) and saveImg:
                torchvision.utils.save_image(
                    Gnn(z).detach(), savePath+'/epoch{}-num{}.jpg'.format(j+1, i), normalize=True)
                tmp = torch.randn(batchSize, nz, 1, 1).to(device)
                torchvision.utils.save_image(
                    Gnn(tmp).detach(), savePath+'/rand-epoch{}-num{}.jpg'.format(j+1, i), normalize=True)

                
            if not(i % shotNum):
                print('[epoch %03d]  G_loss:%.6f,  D_loss:%.6f' % (
                    j, G_loss, D_loss))


        D_lossList_all.append(D_lossList)
        G_lossList_all.append(G_lossList)
        if not(j % saveModelNum) and saveModel:
            torch.save(Gnn.state_dict(), modelPath +
                       "/Gnn-epoch{}.pkl".format(j+1))
            torch.save(Dnn.state_dict(), modelPath +
                       "/Dnn-epoch{}.pkl".format(j+1))
    if writeData:
        write(dataPath, G_lossList_all, D_lossList_all)
    #draw(list(range(len(G_lossList))), G_lossList, D_lossList)
    print('train finish!')
    return Dnn, Gnn


if __name__ == '__main__':
    Dnn = D()
    Gnn = G()
    Dnn.to(device)
    Gnn.to(device)
    trainData = whyDataset(imgPath)
    #show(trainData)
    
    Dnn, Gnn = train(Dnn, Gnn)
    if saveModel:
        saveModel()
        torch.save(Gnn.state_dict(),
                   modelPath+"/Gnn-epoch{}.pkl".format(epoch))
        torch.save(Dnn.state_dict(),
                   modelPath+"/Dnn-epoch-{}.pkl".format(epoch))
