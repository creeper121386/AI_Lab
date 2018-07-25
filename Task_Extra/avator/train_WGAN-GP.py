import os
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn as nn
from torchvision.utils import save_image
import torchvision.transforms as T
import PIL.Image as Image
import numpy as np
from model2 import G, D
#import matplotlib.pyplot as plt
# from torchvision.models import resnet18

############### Hyper Param #################
saveModel = False
saveImg = False
writeData = False
cuda = False
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


# def show(img):
#     # img = T.ToPILImage(img)
#     img = img.numpy()
#     plt.imshow(np.transpose(img, (1, 2, 0)))
#     # plt.axis('off')
#     plt.show()


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
        self.G_loss = []
        self.D_loss = []

    def saveModel(self, No):
        torch.save(self.Gnn.state_dict(), modelPath +
                   "/WGAN_G-epoch{}.pkl".format(No))
        torch.save(self.Dnn.state_dict(), modelPath +
                   "/WGAN_D-epoch{}.pkl".format(No))

    def saveLoss(self, path):
        if writeData:        
            G = np.array(self.G)
            D = np.array(self.D)
            np.savetxt(path+'/Gdata-WGAN.txt', G)
            np.savetxt(path+'/Ddata-WGAN.txt', D)

    def step_D(self, real):
        gploss = GP_loss()
        noise = torch.randn(batchSize, nz, 1, 1).to(device)
        # TODO(Why): change sampling method to random mode in each n_D loop.
        self.Dnn.zero_grad()
        fake = self.Gnn(noise)
        new = torch.ones(real.size(), )      # batchSize * n * imgSize * imgSize
        for k in range(batchSize):
            sigma = float(torch.rand(1, 1))
            new[k] = sigma*real[k] + (1-sigma)*fake[k]        
        pred = self.Dnn(new)
        x_grad = torch.autograd.grad(pred, new, torch.ones(batchSize, 1), retain_graph=True)
        D_loss = gploss(pred, self.Dnn(real), x_grad[0])
        self.Dnn.zero_grad()
        D_loss.backward()
        self.D_optim.step()
        return D_loss

    def train(self, trainLoader):
        z = torch.randn(batchSize, nz, 1, 1).to(device)
        for j in range(epoch):
            D_lossList = []
            G_lossList = []
            for i, (img, _) in enumerate(trainLoader, 0):
                real = img.to(device)
                for _ in range(n_D):
                    D_loss = self.step_D(real)

                noise = torch.randn(batchSize, nz, 1, 1).to(device)
                self.Gnn.zero_grad()
                G_loss = torch.mean(-self.Dnn(self.Gnn(noise)))
                G_loss.backward()
                self.G_optim.step()

                if not i % shotNum:
                    print('[{},epoch:{}]  G_loss:{},  D_loss:{}'.format(
                        i, j, G_loss, D_loss))
                    if saveImg:
                        save_image(self.Gnn(z).detach(), savePath +
                                   '/epoch{}-num{}.jpg'.format(j+1, i), normalize=True)
                        ############# save image from rand noise: ###############
                        # tmp = torch.randn(batchSize, nz, 1, 1).to(device)
                        # save_image(self.Gnn(tmp).detach(
                        # ), savePath+'/rand-epoch{}-num{}.jpg'.format(j+1, i), normalize=True)

            ######### one epoch trainning end #########
            self.D_loss.append(D_lossList)
            self.G_loss.append(G_lossList)
            if not(j % saveNum) and saveModel:
                self.saveModel(j+1)
            self.saveLoss(dataPath)
        print('############## train finish ! ################')


########################## Main: #############################
if __name__ == '__main__':
    trainLoader = DataLoader(
        dataset=whyDataset(imgPath), batch_size=batchSize, shuffle=True, drop_last=True)
    Gan = WGAN()
    Gan.train(trainLoader)
    if saveModel:
        torch.save(Gan.Gnn.state_dict(),
                   modelPath+"/WGAN_G-epoch{}.pkl".format(epoch))
        torch.save(Gan.Dnn.state_dict(),
                   modelPath+"/WGAN_D-epoch-{}.pkl".format(epoch))
