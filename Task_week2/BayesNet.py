import numpy as np
oriNet = {0: [[], [1,2]], 1: [[0], [3]], 2: [[0], [5]], 3: [
    [1], [4]], 4: [[3,5], []], 5: [[2,6], [4]], 6: [[], [5]]}

def load():
    data = np.loadtxt("/media/why/DATA/why的程序测试/AI_Lab/Task/Task_week2/melon2.0.txt",
                      dtype=np.str, delimiter=' ')
    data = data[1:, :]
    #label = data[:, -1:]
    #data = data[:, :-1]
    return data


def calParam(var,net,data):
    for i in net.keys():
        for prtNo in net[i][0]:
            slice = data[:, prtNo:prtNo+1]
            prtValue=np.unique(slice)
            for value in prtValue: 
                index=np.where(slice==value)
                for i in index:
                    pass
                

data = load()
for x in range(7):
    tmp=data[:,x:x+1]
    a=np.unique(tmp)
    print(a)