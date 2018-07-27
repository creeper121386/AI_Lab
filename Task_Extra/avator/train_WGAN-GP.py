import os
from config import *
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn as nn
from torch import autograd
from torchvision.utils import save_image
import torchvision.transforms as T
import PIL.Image as Image
import numpy as np
from model2 import G, D
#import matplotlib.pyplot as plt

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
        grad = (x_grad**2).sum(1).sum(1).sum(1)
        tmp = self.lamda * (torch.sqrt(grad) - 1)**2
        return torch.mean(D_output1-D_output2+tmp)
        # Loss = D(x_new)-D(x)+lambda(||grad(D(x_new))||_2 -1)^2, for D(x_new)=D_output1, D(x)=D_output2.


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
        self.one = torch.FloatTensor([1]).to(device)
        self.n_one = self.one * -1

    def saveModel(self, No):
        torch.save(self.Gnn.state_dict(), modelPath +
                   "/WGAN_G-epoch{}.pkl".format(No))
        torch.save(self.Dnn.state_dict(), modelPath +
                   "/WGAN_D-epoch{}.pkl".format(No))
        print("[save] WGAN_G-epoch{}.pkl, WGAN_D-epoch{}.pkl saved.".format(No, No))

    def saveLoss(self, path):
        if writeData:
            G = np.array(self.Gnn)
            D = np.array(self.Dnn)
            np.savetxt(path+'/Gdata-WGAN.txt', G)
            np.savetxt(path+'/Ddata-WGAN.txt', D)

    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(batchSize, 1).to(device)
        alpha = alpha.expand(batchSize, real_data.nelement(
        )/batchSize).contiguous().view(batchSize, 3, imgSize, imgSize)

        new = alpha * real_data + ((1 - alpha) * fake_data)
        new = new.to(device)
        new = autograd.Variable(new, requires_grad=True)
        pred = self.Dnn(new)
        grad = autograd.grad(outputs=pred, inputs=new,
                                  grad_outputs=torch.ones(pred.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad = grad.view(grad.size(0), -1)
        loss = ((grad.norm(2, dim=1) - 1)** 2).mean() * lamda
        return loss

    def step_D(self, real):
        gploss = GP_loss()
        noise = torch.randn(batchSize, nz, 1, 1).to(device)
        fake = self.Gnn(noise)
        # batchSize * n * imgSize * imgSize
        new = torch.ones(real.size(), ).to(device)
        for k in range(batchSize):
            sigma = float(torch.rand(1, 1))
            new[k] = sigma*real[k] + (1-sigma)*fake[k]
        pred1 = self.Dnn(fake)
        pred2 = self.Dnn(new)
        x_grad = torch.autograd.grad(pred2, new, torch.ones(
            batchSize, 1).to(device), retain_graph=True)
        D_loss = gploss(pred1, self.Dnn(real), x_grad[0])
        self.Dnn.zero_grad()
        D_loss.backward()
        self.D_optim.step()
        return D_loss

    def step_D_new(self, real):
        noise = torch.randn(batchSize, nz, 1, 1).to(device)
        fake = self.Gnn(noise)      # batchSize * n * imgSize * imgSize
        pred_fake = self.Dnn(fake).mean()
        pred_real = self.Dnn(real).mean()
        pred_real.backward(self.n_one)
        pred_fake.backward(self.one)
        D_loss = self.calc_gradient_penalty(real, fake)
        D_loss.backward()
        self.D_optim.step()
        return pred_fake - pred_real + D_loss

    def train(self, trainLoader):
        z = torch.randn(batchSize, nz, 1, 1).to(device)
        for j in range(epoch):
            D_lossList = []
            G_lossList = []
            for i, (img, _) in enumerate(trainLoader, 0):
                real = img.to(device)
                self.Dnn.zero_grad()
                D_loss = self.step_D_new(real)
                if not i % n_D:
                    self.Gnn.zero_grad()
                    noise = torch.randn(batchSize, nz, 1, 1).to(device)
                    G_loss = torch.mean(self.Dnn(self.Gnn(noise)))
                    G_loss.backward(self.n_one)
                    G_loss = - G_loss
                    self.G_optim.step()

                if not i % shotNum:
                    print('[%03d,epoch%2d]  G_loss:%.6f,  D_loss:%.6f' % (i, j, G_loss, D_loss))
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
