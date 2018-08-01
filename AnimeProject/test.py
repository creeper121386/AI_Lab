from model2 import G, D
import torch
import torchvision
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
randZ = False 
nz = 100
batchSize = 64
batchNum = 1
# modelNum = [x*10+1 for x in range(10)]+[100]
modelNum = [1, 6, 10, 11, 16, 20]
workDir = '/run/media/why/DATA/why的程序测试/AI_Lab/Task/Task_Extra/avator'
savePath = workDir + '/img'
modelPath = workDir + '/model'
#################################################

class TestNet(object):
    def __init__(self):
        self.G = G().to(device)
        self.batchNum = batchNum
################ init noise code z: #################

    def init(self):
        if not randZ:
            self.batchNum = 1
            tmp = torch.randn(nz, 1, 1)
            initZ = torch.randn(batchSize, nz, 1, 1)
            for i in range(nz-5):
                for j in range(batchSize):
                    initZ[j][i] = tmp[i]
            self.initZ = initZ.to(device)

###################### test:#########################
    def test(self):
        for n in modelNum:
            GmodelPath = modelPath + '/Gnn-epoch{}.pkl'.format(n)
            self.G.load_state_dict(torch.load(GmodelPath))
            for i in range(batchNum):
                if randZ:
                    z = torch.randn(batchSize, nz, 1, 1).to(device)
                else:
                    z = self.initZ
                torchvision.utils.save_image(self.G(z).detach(
                ), savePath+'/epoch{}-test{}.jpg'.format(n, i), normalize=True)
        print('\033[1;36;40m  generate over! \033[0m')

############## load model: ###############
#Dnn = D()
# Dnn.load_state_dict(torch.load(modelPath))


if __name__ == '__main__':
    testNet = TestNet()
    testNet.init()
    testNet.test()
