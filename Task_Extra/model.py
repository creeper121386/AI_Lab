import torch.nn as nn

nz = 100        # 白噪声向量的长度
nc = 1         # channel数
ndf = 64      # 网络D的feature map数量
ngf = 64       # 网络G的feature map数量
imgSize = 400
batchSize = 10

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

            nn.Conv2d(ndf*4, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # return self.layers(input).view(-1, 1).squeeze(1)
        return self.layers(x)


class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*4, 4),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)