import torchvision.datasets as dSet
import torchvision.transforms as T
import torch.optim as optim
import torch.nn as nn
import torchvision
import torch
cuda = True
epoch = 30
lr = 0.01
nz = 100        # 白噪声向量的长度
nc = 1         # channel数
ndf = 64      # 网络D的feature map数量
ngf = 64       # 网络G的feature map数量
imgSize = 28
batchSize = 4
device = torch.device("cuda:0" if cuda else "cpu")
transform = T.Compose(
    [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainData = dSet.MNIST(root='/run/media/why/DATA/why的程序测试/AI_Lab/DataSet',
                       train=True, transform=transform, download=True)      # size = 28*28
trainLoader = torch.utils.data.DataLoader(dataset=trainData,
                                          batch_size=batchSize,
                                          shuffle=True)

path = '/run/media/why/DATA/why的程序测试/AI_Lab/Task/task_week6/img'


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.layers(input).view(-1, 1).squeeze(1)


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, input):
        return self.layers(input)

gNet = G()
dNet = D()
if cuda:
    dNet = dNet.cuda()
    gNet = gNet.cuda()
D_optim = optim.Adam(dNet.parameters(), lr=lr, betas=(0.5, 0.999))
G_optim = optim.Adam(gNet.parameters(), lr=lr, betas=(0.5, 0.999))
fixed_noise = torch.randn(batchSize, nz, 1, 1).cuda()
criterion = nn.BCELoss()


for j in range(epoch):
    for i, data in enumerate(trainLoader, 0):
        dNet.zero_grad()
        real = data[0].to(device)
        label = torch.full((batchSize,), 1).cuda()
        output = dNet(real)
        err_D = criterion(output, label)
        err_D.backward()
        Dx = output.mean().item()

        noise = torch.randn(batchSize, nz, 1, 1).cuda()
        fake = gNet(noise)
        label.fill_(0)
        output = dNet(fake.detach())
        err_Dfake = criterion(output, label)
        err_Dfake.backward()
        z1 = output.mean().item()
        errD = err_Dfake + err_D
        D_optim.step()

        gNet.zero_grad()
        label.fill_(1)
        output = dNet(fake)
        errG = criterion(output, label)
        errG.backward()
        z2 = output.mean().item()
        G_optim.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (j, epoch, i, len(trainLoader),
                 errD.item(), errG.item(), Dx, z1, z2))

        if i % 100 == 0:
            torchvision.utils.save_image(real,
                    '%s/real_samples.png' % path,
                    normalize=True)
            fake = gNet(fixed_noise)
            torchvision.utils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (path, j),
                    normalize=True)



