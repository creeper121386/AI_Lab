import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as T
import torch.nn as nn
import torch
import PIL.Image as Image
import os
import numpy as np
from model2 import G, D
#import matplotlib.pyplot as plt
# from torchvision.models import resnet18

############### Hyper Param #################
saveModel = True
saveImg = False
cuda = True
sampleNum = False    # set False to use all samples
epoch = 20
lr = 0.0001
n_D = 5     # train D_net 5 times in a iteration.
nz = 100        # size of noise
imgSize = 64
batchSize = 64
shotNum = 100       # save loss info per 100 iterations
saveNum = 5     # save current model per 5 times

############### Path ######################
# workDir = '/disk/unique/why'
#imgPath = '/run/media/why/DATA/why的程序测试/AI_Lab/DataSet/faces'
workDir = os.getcwd()
imgPath = workDir + '/faces'
savePath = workDir + '/saveImg'
modelPath = workDir + '/model'
dataPath = workDir + '/data'

################## init Function ########################
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
        img = transform(img)
        return img, 1   # real label


class GP_loss(nn.Module):
    def __init__(self, lamda):
        super(GP_loss, self).__init__()
        self.lamda = lamda
        return

    def forward(self, D_output1, D_output2, D_grad):
        tmp = self.lamda * (torch.sqrt(D_grad) - 1)**2
        return torch.mean(D_output1+D_output2+tmp)


def show(img):
    img = img.numpy()
    # img = T.ToPILImage(img)
    plt.imshow(np.transpose(img, (1, 2, 0)))
    # plt.axis('off')
    plt.show()


def writeInfo(path, G, D):
    G = np.array(G)
    D = np.array(D)
    np.savetxt(path+'/Gdata-WGAN.txt', G)
    np.savetxt(path+'/Ddata-WGAN.txt', D)


class WGAN(object):
    def __init__(self):
        self.Dnn = D().to(device)
        self.Gnn = G().to(device)
        self.D_optim = optim.Adam(Dnn.parameters(), lr=lr, betas=(0.0, 0.9))
        self.G_optim = optim.Adam(Gnn.parameters(), lr=lr, betas=(0.0, 0.9))

    def train(self, trainLoader):
        z = torch.randn(batchSize, nz, 1, 1).to(device)
        D_lossList_all = []
        G_lossList_all = []
        for j in range(epoch):
            D_lossList = []
            G_lossList = []
            for i, (img, _) in enumerate(trainLoader, 0):
                for _ in range(n_D):
                    noise = torch.randn(batchSize, nz, 1, 1).to(device)
                    sigma = torch.rand(batchSize, 1)
                    real = img.to(device)
                    self.Dnn.zero_grad()
                    fake = self.Gnn(noise)
                    new = sigma*real + (1-sigma)*new
                    





                pred1 = self.Dnn(real)
                # Edit here to change the REAL&FAKE LABEL!
                realLabel = torch.full((batchSize, ), 1, device=device)
                fakeLabel = torch.full((batchSize, ), 0, device=device)
                D_loss1 = GP_loss(pred1, realLabel)
                D_loss1.backward()

                fake = self.Gnn(noise)
                pred2 = self.Dnn(fake.detach())
                D_loss2 = GP_loss(pred2, fakeLabel)
                D_loss2.backward()
                self.D_optim.step()

                self.Gnn.zero_grad()
                D_loss = D_loss1 + D_loss2
                D_lossList.append(float(D_loss))
                pred3 = self.Dnn(fake)
                G_loss = GP_loss(pred3, realLabel)
                G_lossList.append(float(G_loss))
                G_loss.backward()
                self.G_optim.step()

                if not(i % shotNum):
                    print('[{},epoch:{}]  G_loss:{},  D_loss:{}'.format(
                        i, j, G_loss, D_loss))
                    if saveImg:
                        save_image(self.Gnn(z).detach(), savePath +
                                   '/epoch{}-num{}.jpg'.format(j+1, i), normalize=True)
                        ############# save image from random noise: ###############
                        # tmp = torch.randn(batchSize, nz, 1, 1).to(device)
                        # save_image(self.Gnn(tmp).detach(
                        # ), savePath+'/rand-epoch{}-num{}.jpg'.format(j+1, i), normalize=True)

            ################ epoch end ####################
            D_lossList_all.append(D_lossList)
            G_lossList_all.append(G_lossList)
            if not(j % saveNum) and saveModel:
                torch.save(self.Gnn.state_dict(), modelPath +
                           "/WGAN_G-epoch{}.pkl".format(j+1))
                torch.save(self.Dnn.state_dict(), modelPath +
                           "/WGAN_D-epoch{}.pkl".format(j+1))
        writeInfo(dataPath, G_lossList_all, D_lossList_all)
        #draw(list(range(len(G_lossList))), G_lossList, D_lossList)
        print('############### train finish ! ################')


########################## Main: #############################
if __name__ == '__main__':
    trainLoader = DataLoader(
        dataset=whyDataset(imgPath), batch_size=batchSize, shuffle=True, drop_last=True)
    Gan = WGAN()
    Gan.train(trainLoader)
    if saveModel:
        torch.save(Gnn.state_dict(),
                   modelPath+"/WGAN_G-epoch{}.pkl".format(epoch))
        torch.save(Dnn.state_dict(),
                   modelPath+"/WGAN_D-epoch-{}.pkl".format(epoch))
