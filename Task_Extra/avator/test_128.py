from model_128 import G, D
import torch
import torchvision
nz = 100
batchSize = 64
batchNum = 2
modelNum = (1, 6, 10)
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
savePath = '/run/media/why/DATA/why的程序测试/AI_Lab/Task/Task_Extra/avator/img128'


#Dnn = D()
# Dnn.load_state_dict(torch.load(modelPath))
Gnn = G().to(device)

for n in modelNum:
    GmodelPath = '/run/media/why/DATA/why的程序测试/AI_Lab/Task/Task_Extra/avator/model_128/S-Gnn-epoch{}.pkl'.format(n)
    Gnn.load_state_dict(torch.load(GmodelPath))
    for i in range(batchNum):
        z = torch.randn(batchSize, nz, 1, 1).to(device)
        out = Gnn(z)
        torchvision.utils.save_image(Gnn(z).detach(
        ), savePath+'/img128_epoch{}-test{}.jpg'.format(n, i), normalize=True)
