import numpy as np
import jieba

fo = open("/run/media/why/DATA/why的程序测试/AI_Lab/Task/task_week5/code.txt", "r+")
str = fo.read()
fo.close()
# code = [1 if x =='-' else 0 for x in str]

jieba.load_userdict('/run/media/why/DATA/why的程序测试/AI_Lab/Task/task_week5/mydict.txt')
seg_list = jieba.cut(str, cut_all=True)
print(", ".join(seg_list))