import networkx as nx

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

import matplotlib.pyplot as plt
#%matplotlib inline


# 建立一个简单贝叶斯模型骨架
model = BayesianModel([('D', 'G'), ('I', 'G'), ('G', 'L'), ('I', 'S')])

# 最顶层的两个父节点的概率分布表
cpd_d = TabularCPD(variable='D', variable_card=2,
                   values=[[0.6, 0.4]])  # D: 课程难度(0,1)
cpd_i = TabularCPD(variable='I', variable_card=2,
                   values=[[0.7, 0.3]])  # I: 学生智商(0,1)

# 其它各节点的条件概率分布表（行对应当前节点索引，列对应父节点索引）
cpd_g = TabularCPD(variable='G', variable_card=3,       # G: 考试成绩(0,1,2)
                   values=[[0.3, 0.05, 0.9,  0.5],
                           [0.4, 0.25, 0.08, 0.3],
                           [0.3, 0.7,  0.02, 0.2]],
                   evidence=['I', 'D'],
                   evidence_card=[2, 2])
cpd_s = TabularCPD(variable='S', variable_card=2,       # S: SAT成绩(0,1)
                   values=[[0.95, 0.2],
                           [0.05, 0.8]],
                   evidence=['I'],
                   evidence_card=[2])
cpd_l = TabularCPD(variable='L', variable_card=2,       # L: 推荐质量(0,1)
                   values=[[0.1, 0.4, 0.99],
                           [0.9, 0.6, 0.01]],
                   evidence=['G'],
                   evidence_card=[3])

# 将各节点的概率分布表加入网络
model.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)

# 验证模型数据的正确性
print(u"验证模型数据的正确性:", model.check_model())

# 绘制贝叶斯图(节点+依赖关系)
nx.draw(model, with_labels=True, node_size=1000, font_weight='bold', node_color='y',
        pos={"L": [4, 3], "G": [4, 5], "S": [8, 5], "D": [2, 7], "I": [6, 7]})
plt.text(2, 7, model.get_cpds("D"), fontsize=10, color='b')
plt.text(5, 6, model.get_cpds("I"), fontsize=10, color='b')
plt.text(1, 4, model.get_cpds("G"), fontsize=10, color='b')
plt.text(4.2, 2, model.get_cpds("L"), fontsize=10, color='b')
plt.text(7, 3.4, model.get_cpds("S"), fontsize=10, color='b')
plt.title(u"带有条件概率分布的学生示例的贝叶斯网")
plt.show()
