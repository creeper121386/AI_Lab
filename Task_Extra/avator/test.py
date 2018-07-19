from model2 import G, D
import torch
import torchvision
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
nz = 100
batchSize = 64
batchNum = 1
# modelNum = [x*10+1 for x in range(10)]+[100]
modelNum = [91, ]
randZ = True
tmp = torch.randn(nz, 1, 1)
initZ = torch.randn(batchSize, nz, 1, 1)
savePath = '/run/media/why/DATA/why的程序测试/AI_Lab/Task/Task_Extra/avator/img'


for i in range(nz-1):
    for j in range(batchSize):
        initZ[j][i] = tmp[i]
initZ = initZ.to(device)


#Dnn = D()
# Dnn.load_state_dict(torch.load(modelPath))
Gnn = G().to(device)

for n in modelNum:
    GmodelPath = '/run/media/why/DATA/why的程序测试/AI_Lab/Task/Task_Extra/avator/model/Gnn-epoch{}.pkl'.format(
        n)
    Gnn.load_state_dict(torch.load(GmodelPath))
    for i in range(batchNum):
        if randZ:
            z = torch.randn(batchSize, nz, 1, 1).to(device)
        else:
            z = initZ
        out = Gnn(z)
        torchvision.utils.save_image(Gnn(z).detach(
        ), savePath+'/epoch{}-test{}.jpg'.format(n, i), normalize=True)
