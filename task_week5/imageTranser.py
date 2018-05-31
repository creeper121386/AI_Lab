import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
import copy


imgSize = 200
num_steps = 300
contentWeight = 1
styleWeight = 1e5
contentLayers = ['conv_4']
styleLayers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
loader = transforms.Compose(
    [transforms.Resize(imgSize), transforms.CenterCrop(imgSize), transforms.ToTensor()])
rootPath = "/run/media/why/DATA/why的程序测试/AI_Lab/Task/task_week5/"
cnn = (models.vgg19(pretrained=True).features).cuda().eval()
unloader = transforms.ToPILImage()


def loadImg(fname):
    img = Image.open(rootPath+fname)
    img = (loader(img).unsqueeze(0)).to("cuda", torch.float)
    return img


def show(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    plt.pause(0.1)  # pause a bit so that plots are updated


def calGram(featmap):
    bSize, mapNum, m, n = featmap.size()
    feat = featmap.view(bSize*mapNum, m*n)
    G = torch.mm(feat, feat.t())
    return G.div(bSize*mapNum*m*n)


class contentLoss(nn.Module):
    def __init__(self, content, weight):
        super().__init__()
        self.weight = weight
        self.func = nn.MSELoss()
        self.content = content.detach() * weight

    def forward(self, input):
        self.loss = self.func(input * self.weight, self.content)
        return input

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss


class styleLoss(nn.Module):
    def __init__(self, target, weight):
        super().__init__()
        self.weight = weight
        self.func = nn.MSELoss()
        self.target = calGram(target).detach() * weight

    def forward(self, input):
        self.output = input.clone()
        self.G = calGram(input)
        self.G.mul_(self.weight)
        self.loss = self.func(self.G, self.target)
        return self.output

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss


def calLoss(cnn, origin, target):
    cnn = copy.deepcopy(cnn)
    model = nn.Sequential().cuda()
    C_losses = []
    S_losses = []
    i = 0
    for L in cnn.children():
        if isinstance(L, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(L, nn.ReLU):
            name = 'relu_{}'.format(i)
            L = nn.ReLU(inplace=False)
        elif isinstance(L, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            model.add_module(name, L)
            continue
        model.add_module(name, L)

        if name in contentLayers:
            contentFeat = model(origin).detach()
            C_loss = contentLoss(contentFeat, contentWeight)
            model.add_module("contentLoss{}".format(i), C_loss)
            C_losses.append(C_loss)

        if name in styleLayers:
            styleFeat = model(target).detach()
            S_loss = styleLoss(styleFeat, styleWeight)
            model.add_module("styleLoss{}".format(i), S_loss)
            S_losses.append(S_loss)
    return model, C_losses, S_losses


def transfer(cnn, origin, target, inputImg):
    model, C_losses, S_losses = calLoss(cnn, origin, target)
    prama = nn.Parameter(inputImg.data)
    optimizer = optim.LBFGS([prama])
    run = [0]
    while run[0] <= num_steps:
        def closure():
            prama.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(prama)
            styleScore = 0
            contentScore = 0
            for sl in S_losses:
                styleScore += sl.backward()
            for cl in C_losses:
                contentScore += cl.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    styleScore.data[0], contentScore.data[0]))
                print()
            return styleScore+contentScore
        optimizer.step(closure)
    prama.data.clamp_(0, 1)
    return prama.data


plt.ion()
target = loadImg("target.jpg")
origin = loadImg("origin.jpg")
img = origin.clone()
out = transfer(cnn, origin, target, img)
plt.figure()
show(out)
plt.savefig('/run/media/why/DATA/why的程序测试/AI_Lab/Task/task_week5/result.jpg')
plt.ioff()
plt.show()
