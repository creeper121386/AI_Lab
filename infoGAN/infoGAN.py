import torchvision.datasets as dataset
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as D
import torch.optim as optim
import torch.nn as nn
import numpy as np

# 超参数设定 & 数据导入：
cuda = True
imgSize = 28
batchSize = 8
trainData = dataset.MNIST(root='/run/media/why/DATA/why的程序测试/AI_Lab/DataSet', train=True,
                  transform=T.Compose([T.Resize(imgSize), T.ToTensor()]), download=True)      # size = 28*28
trainLoader = D.DataLoader(dataset=trainData, batch_size=batchSize, shuffle=True)
device = torch.device("cuda" if cuda else "cpu")










print("test over.")
