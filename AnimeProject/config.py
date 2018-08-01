import os

############### Hyper Param #################
saveModel = False
saveImg = True
writeData = False
cuda = False
sampleNum = 10    # set False to use all samples
epoch = 10
batchSize = 8
lr = 0.0001
n_D = 5     # train D_net 5 times in a iteration.
nz = 100        # size of noise
imgSize = 64
shotNum = 10      # save loss info per 100 iterations
saveImgNum = 10
saveModelNum = 10     # save current model per 5 times
ndf =64
ngf =64
nc = 3      # num of channels
lamda = 10

############### Path ######################
# workDir = '/disk/unique/why'
workDir = os.getcwd()
imgPath = workDir + '/faces'
imgPath = '/run/media/why/DATA/why的程序测试/AI_Lab/DataSet/AnimeProject/LL'
savePath = workDir + '/img'
modelPath = workDir + '/model'
dataPath = workDir + '/data'