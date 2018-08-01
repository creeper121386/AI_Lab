import matplotlib.pyplot as plt
from PIL import Image
import os
IMG_DIR = '/run/media/why/DATA/why的程序测试/AI_Lab/DataSet/AnimeProject/LL' 

def test():
    err = []
    dir = os.listdir(IMG_DIR)
    for x in dir:
        try:
            img = Image.open(IMG_DIR + '/{}'.format(x))
        except :
            err.append(x)
    print(err)
    print(len(err), 'files were broken.')
    return err
    # plt.figure()
    # plt.imshow(img)
    # plt.colorbar()
    # plt.show()


def remove(err):
    if not len(err):
        print('no file was broken.')
        return 0
    else:
        x = input("remove broken files? (y/n):")
        if x == 'y':
            for x in err:
                os.remove(IMG_DIR + '/{}'.format(x))
            print('removed %d files' % len(err))
        else:
            print('cancel removing.')


if __name__ == '__main__':
    err = test()
    remove(err)
    