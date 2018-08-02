from PIL import Image
import random
import torch
import torchvision.utils as vutils
import torchvision.transforms
import models
import re
CUDA = True
model_path = '/run/media/why/DATA/why的程序测试/AI_Lab/Task/AnimeProject/animeGAN-master/netG.pth'
save_path = '/run/media/why/DATA/why的程序测试/AI_Lab/Task/AnimeProject/animeGAN-master/img'
device = torch.device('cuda') if CUDA else torch.device('cpu')


def convert_img(img_tensor, nrow):
    img_tensor = img_tensor.to(device)
    grid = vutils.make_grid(img_tensor, nrow=nrow, padding=2)
    grid = grid.cpu()
    ndarr = grid.mul(0.5).add(0.5).mul(255).byte().transpose(0, 2).transpose(0, 1).numpy()
    im = Image.fromarray(ndarr)
    return im

def load():
    netG = models._netG_1(1, 100, 3, 64, 1) # ngpu, nz, nc, ngf, n_extra_layers
    netG = netG.to(device)
    if CUDA:
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    self_state = netG.state_dict()
    for name, param in state_dict.items():
        param = param.to(device)
        if name not in self_state and isinstance(param, torch.Tensor):
            for x, y in self_state.items():
                res = re.match(name, x)
                if res:
                    self_state[x].copy_(param)
                    break
    return netG


netG = load()
noise_batch = torch.FloatTensor(64, 100, 1, 1).normal_(0,1).to(device)
fake_batch, _ = netG(noise_batch)
im = convert_img(fake_batch.data, 8)
im.save('{}/{}.jpg'.format(save_path, random.randint(0, 1e5)))