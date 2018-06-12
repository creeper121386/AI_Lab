import torchvision.datasets as dSet
import torchvision.transforms as T
import torch.optim as optim
from model import G, D
import torch.nn as nn
import torchvision
import torch
cuda = True
epoch = 10
lr = 0.01       # 学习率
nz = 100        # 白噪声向量的长度
imgSize = 32    
batchSize = 16
criterion = nn.BCELoss()
device = torch.device("cuda" if cuda else "cpu")
path = '/run/media/why/DATA/why的程序测试/AI_Lab/Task/task_week6/img'
trainData = dSet.MNIST(root='/run/media/why/DATA/why的程序测试/AI_Lab/DataSet',
                       train=True, transform=T.Compose([T.Resize(imgSize), T.ToTensor()]), download=True)      # size = 28*28
trainLoader = torch.utils.data.DataLoader(dataset=trainData,
                                          batch_size=batchSize,
                                          shuffle=True)


gNet = G()
dNet = D()
if cuda:
    dNet = dNet.cuda()
    gNet = gNet.cuda()
D_optim = optim.Adam(dNet.parameters(), lr=lr, betas=(0.5, 0.999))
G_optim = optim.Adam(gNet.parameters(), lr=lr, betas=(0.5, 0.999))
fixed_noise = torch.randn(batchSize, nz, 1, 1).cuda()               # 用来生成假图片的噪声

for j in range(epoch):
    for i, (img, _) in enumerate(trainLoader, 0):
        # 训练识别器：
        dNet.zero_grad()                        # 初始化所有梯度
        real = img.to(device)                   # 训练集中抽取的真图片
        label = torch.full((batchSize,), 1, device=device)          # 为所有真图片标记1
        output = dNet(real)
        err_D = criterion(output, label)        # 计算loss
        err_D.backward()                        # 反向计算所有梯度
        #　Dx = output.mean().item()               

        noise = torch.randn(batchSize, nz, 1, 1, device=device)     # 创建噪声向量
        fake = gNet(noise)                      # 开始造假！
        label.fill_(0)                          # 为所有假图片标记0
        output = dNet(fake.detach())            # 识别假图片（不计算G中参数的梯度）
        err_Dfake = criterion(output, label)    # 计算loss
        err_Dfake.backward()                    # 反向计算梯度
        # z1 = output.mean().item()              
        errD = err_Dfake + err_D                # 综合考虑loss
        D_optim.step()                          # 更新识别器的参数
        # 训练生成器：
        gNet.zero_grad()                        # 初始化梯度
        label.fill_(1)                          # 为所有label标记1（要使得造假后的图片输入D后，输出的结果更接近1）
        output = dNet(fake)                     # 识别假图片
        errG = criterion(output, label)         # 计算loss
        errG.backward()                         # 反向计算梯度
        # z2 = output.mean().item()               
        G_optim.step()                          # 更新生成器的参数
        # 输出、保存训练信息
        if not(i % 100):
            print('epoch:[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (j, epoch, i, len(trainLoader), errD.item(), errG.item()))
            torchvision.utils.save_image(real,'%s/realSamples.png' % path,normalize=True)
            fake = gNet(fixed_noise)
            torchvision.utils.save_image(fake.detach(),
                '%s/fakeSamples_epoch_%03d.png' % (path, j),normalize=True)
print('traning finished!')