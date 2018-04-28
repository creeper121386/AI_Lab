import numpy as np

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
    def __init__(self, no, prt, chld):
        self.no = no
        self.prt = prt
        self.chld = chld
        self.prob = [0]*len(prt)


class Net(object):
    def __init__(self, matrix):
        self.matrix = matrix
    def init(self,data):
        nodeList = []
        length = len(self.matrix[0])
        for i in range(length):
            slice = [x[0] for x in self.matrix[:, i:i+1]]
            chldList = [x for x in range(length) if self.matrix[i][x] != 0]
            prtList = [x for x in range(length) if slice[x] != 0]
            node = Node(i, prtList, chldList)
            j = 0
            for prtNo in prtList:
                prob = []
                prtSlice = data[:, prtNo:prtNo+1]
                nodeSlice = data[:, i:i+1]
                prtValue = np.unique(prtSlice)
                nodeValue = np.unique(nodeSlice)
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
            self.nodes=nodeList




    def isDAG(self):
        pass






def load():
    data = np.loadtxt("/media/why/DATA/why的程序测试/AI_Lab/Task/Task_week2/melon2.0_real.txt",
                      dtype=np.str, delimiter=' ')
    data = data[1:, :]
    return data



'''
# 一开始没想到用矩阵来表示...
oriNet = {0: [[], [1, 2]], 1: [[0], [3]], 2: [[0], [4]], 3: [
    [1, 4], [5]], 4: [[2, 6], [3]], 5: [[3], []], 6: [[], [4]]}
def initialize(oriNet, data):
    net = []
    for i in oriNet.keys():
        j = 0
        node = Node(i, oriNet[i][0], oriNet[i][1])
        for prtNo in oriNet[i][0]:
            prob = []
            prtSlice = data[:, prtNo:prtNo+1]
            chldSlice = data[:, i:i+1]
            prtValue = np.unique(prtSlice)
            chldValue = np.unique(chldSlice)
            for prt in prtValue:
                tmp = []
                prtData = [x for x in data if x[prtNo] == prt]
                for chld in chldValue:
                    chldData = [x for x in prtData if x[i] == chld]
                    p = len(chldData)/len(prtData)
                    tmp.append(p)
                prob.append(tmp)
            prob = np.array(prob)
            node.prob[j] = prob
            j += 1
        net.append(node)
    return net'''
'''def netInit(matrix, data):
    nodeList=[]
    length=len(matrix[0])
    for i in range(length):
        slice = [x[0] for x in matrix[:, i:i+1]]
        chldList=[x for x in range(length) if matrix[i][x]!=0]
        prtList = [x for x in range(length) if slice[x] != 0]
        node=Node(i,prtList,chldList)
        j=0
        for prtNo in prtList:
            prob = []
            prtSlice = data[:, prtNo:prtNo+1]
            nodeSlice = data[:, i:i+1]
            prtValue = np.unique(prtSlice)
            nodeValue = np.unique(nodeSlice)
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
            j+=1
        nodeList.append(node)
        net=Net(nodeList,matrix)
    return net'''


data = load()
net = Net(oriNet)
net.init(data)