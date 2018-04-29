import numpy as np
varNum = 7
oriNet = np.array([
    [0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0],  # 对应隐变量
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0]   # 对应标签
])  # 某行代表某结点指向的结点；某列代表某结点的父节点


class Node(object):
    def __init__(self, no, prt, chld, value):
        self.no = no
        self.value = value
        self.prt = prt
        self.chld = chld
        self.prob = [0]*len(prt)


class Net(object):
    def __init__(self, matrix):
        self.matrix = matrix

    def isDAG(self):
        if (np.zeros(varNum) == np.diag(self.matrix)).all():
            mat1 = (self.matrix-1)*3
            mat2 = (np.transpose(self.matrix)-1)*2
            if not (mat1 == mat2).any():
                return True
            else:
                return False
        else:
            return False

    def init(self, data):
        nodeList = []
        length = len(self.matrix[0])
        for i in range(length):
            slice = [x[0] for x in self.matrix[:, i:i+1]]
            chldList = [x for x in range(length) if self.matrix[i][x] != 0]
            prtList = [x for x in range(length) if slice[x] != 0]
            nodeSlice = data[:, i:i+1]
            nodeValue,count = np.unique(nodeSlice,return_counts=True)
            node = Node(i, prtList, chldList, nodeValue)
            j = 0
            if len(prtList)==0:
                node.prob=count/len(data)
            else:
                for prtNo in prtList:
                    prob = []
                    prtSlice = data[:, prtNo:prtNo+1]
                    prtValue = np.unique(prtSlice)
                    for prt in prtValue:
                        tmp = []
                        prtData = [x for x in data if x[prtNo] == prt]
                        for chld in nodeValue:
                            chldData = [x for x in prtData if x[i] == chld]
                            p = len(chldData)/len(prtData)
                            tmp.append(p)
                        prob.append(tmp)
                    prob = np.array(prob)
                    node.prob[j] = prob
                    j += 1
            nodeList.append(node)
        self.nodes = nodeList

    def optim(self, data):
        pass

    def calProb(self, feat):
        prob=1
        for i in range(varNum):
            if len(self.nodes[i].prt)==0:
            for prt in :
                pass



def load(divide):
    data = np.loadtxt("/media/why/DATA/why的程序测试/AI_Lab/Task/Task_week2/melon2.0_real.txt",
                      dtype=np.str, delimiter=' ')
    data = data[1:, :]
    if divide:
        label = data[:, -1:]
        data = data[:, :-1]
        return data, label
    else:
        return data


data = load(divide=False)
while 1:
    net = Net(np.random.randint(0, 2, size=(varNum, varNum)))
    if net.isDAG():
        break
net.init(data)
testData, testLabel = load(divide=True)
