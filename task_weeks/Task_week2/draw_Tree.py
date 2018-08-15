import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

node = dict(boxstyle = 'sawtooth', fc = '0.8')
leaf = dict(boxstyle = 'round4', fc = '0.8')
arrow = dict(arrowstyle = '<-')

def plotNode(nodeTxt, centrePt, parentPt, nodeType):
    creatPlot.ax1.annotate(nodeTxt, xy=parentPt,xycoords="axes fraction",xytext=centrePt,  textcoords='axes fraction',
    va='center', ha='center', bbox=nodeType,
    arrowprops=arrow)

def creatPlot():
    fig = plt.figure(1, facecolor='white')
    creatPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode(u'决策节点', (0.5, 0.1), (0.1, 0.5), node)
    plotNode(u'叶节点', (0.8, 0.1), (0.3, 0.8), leaf)
    plt.show()

creatPlot()
