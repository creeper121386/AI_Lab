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
saveModel = False
saveImg = False
writeData = True
cuda = True
sampleNum = 10    # set False to use all samples
epoch = 10
batchSize = 8
lr = 0.0001
n_D = 5     # train D_net 5 times in a iteration.
nz = 100        # size of noise
imgSize = 64
shotNum = 100       # save loss info per 100 iterations
saveNum = 5     # save current model per 5 times
nc = 3      # num of channels

############### Path ######################
# workDir = '/disk/unique/why'
workDir = os.getcwd()
imgPath = workDir + '/faces'
imgPath = '/run/media/why/DATA/why的程序测试/AI_Lab/DataSet/faces'
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
    def __init__(self, lamda=10):
        super(GP_loss, self).__init__()
        self.lamda = lamda
        return

    def forward(self, D_output1, D_output2, x_grad):
        tmp = self.lamda * (torch.sqrt(x_grad) - 1)**2
        return torch.mean(D_output1-D_output2+tmp)
        # Loss = D(x_new)-D(x)+lambda(||grad(D(x_new))||-1)^2, for D(x_new)=D_output1, D(x)=D_output2.


def show(img):
    # img = T.ToPILImage(img)
    img = img.numpy()
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
        Dnn = D()
        Gnn = G()
        self.Dnn = Dnn.to(device)
        self.Gnn = Gnn.to(device)
        self.D_optim = optim.Adam(
            self.Dnn.parameters(), lr=lr, betas=(0.0, 0.9))
        self.G_optim = optim.Adam(
            self.Gnn.parameters(), lr=lr, betas=(0.0, 0.9))

    def step_D(self, real):
        gploss = GP_loss()
        noise = torch.randn(batchSize, nz, 1, 1).to(device)
        self.Dnn.zero_grad()
        fake = self.Gnn(noise)
        new = torch.zeros(real.size())      # batchSize * n * imgSize * imgSize
        for k in range(batchSize):
            sigma = float(torch.rand(1, 1))
            new[k] = sigma*real[k] + (1-sigma)*fake[k]
        pred = self.Dnn(new)
        x_grad = torch.ones((batchSize, 1))
        for k in range(batchSize):
            pred[k].backward()
            x_grad[k] = new.grad[k]
        D_loss = gploss(pred, self.Dnn(real), x_grad)
        self.Dnn.zero_grad()
        D_loss.backward()
        self.D_optim.setp()

    def train(self, trainLoader):
        z = torch.randn(batchSize, nz, 1, 1).to(device)
        D_lossList_all = []
        G_lossList_all = []
        for j in range(epoch):
            D_lossList = []
            G_lossList = []
            for i, (img, _) in enumerate(trainLoader, 0):
                real = img.to(device)
                for _ in range(n_D):
                    self.step_D(real)
                
                noise = torch.randn(batchSize, nz, 1, 1).to(device)
                self.Gnn.zerograd()
                G_loss = torch.mean(-self.Dnn(self.Gnn(noise)))
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
        if writeData:
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
