import numpy as np
import matplotlib.pyplot as plt
import os

#dataPath = os.getcwd()
workDir = '/run/media/why/DATA/why的程序测试/AI_Lab/Task/Task_Extra/avator'
dataPath = workDir + '/data'
plotPath = workDir + '/plot'
linewidth = 1
alpha = 0.8
dot = False


def load(path):
    G = np.loadtxt(path+'/Gdata.txt')
    D = np.loadtxt(path+'/Ddata.txt')
    return G, D


def draw(G, D, xlabel, name):
    x = list(range(len(G)))
    plt.figure()
    plt.ylabel('Loss')
    plt.xlabel(xlabel)
    plt.plot(x, G, linewidth=linewidth, label='G_loss', color='blue', alpha = alpha)
    plt.plot(x, D, linewidth=linewidth, label='D_loss', color='green', alpha = alpha)
    if dot:
        plt.scatter(x, G, c='blue', edgecolors='black', alpha = 0.6)
        plt.scatter(x, D, c='green', edgecolors='black', alpha = 0.6)
    plt.legend()
    plt.xlim((0,len(x)))
    plt.savefig(plotPath+'/{}.png'.format(name))


def plot():
    G, D = load(dataPath)
    G_all = []
    D_all = []
    for i in range(len(G)):
        draw(G[i], D[i], 'num of samples in epoch {}'.format(i), 'epoch{}'.format(i))    
        G_all.append(np.average(G[i]))
        D_all.append(np.average(D[i]))
    draw(G_all, D_all, 'training epoch', 'all.png')
    print('/033[1;36;40m plot over! /033[0m')

if __name__ == '__main__':
    plot()
    