import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import copy

imgSize = 512
loader = transforms.Compose(
    [transforms.Resize(imgSize), transforms.CenterCrop(imgSize), transforms.ToTensor()])
rootPath = "/run/media/why/DATA/why的程序测试/AI_Lab/Task/task_week5/"
cnn = (models.vgg19(pretrained=True).features).cuda()

def loadImg(fname):
    img = Image.open(rootPath+fname)
    img = (loader(img).unsqueeze(0)).to("cuda", torch.float)
    return img

class C_loss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
    def forward(self, img):
        self.loss = F.mse_loss(img, self.target)
        return img




target = loadImg("target.jpg")
origin = loadImg("origin.jpg")







unloader = transforms.ToPILImage()  # reconvert into PIL image
plt.ion()
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(5) # pause a bit so that plots are updated
plt.figure()
imshow(target, title='Style Image')
plt.figure()
imshow(origin, title='Content Image')