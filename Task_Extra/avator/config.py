import os

############### Hyper Param #################
saveModel = False
saveImg = False
writeData = False
cuda = True
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
lamda = 10

############### Path ######################
# workDir = '/disk/unique/why'
workDir = os.getcwd()
imgPath = workDir + '/faces'
imgPath = '/run/media/why/DATA/why的程序测试/AI_Lab/DataSet/faces'
savePath = workDir + '/saveImg'
modelPath = workDir + '/model'
dataPath = workDir + '/data'