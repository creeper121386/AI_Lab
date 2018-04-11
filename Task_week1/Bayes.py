import numpy as np
import os
import re
root = "/media/why/DATA/why的程序测试/AI_Lab/NLP/"
train_path1 = root+"train/pos/"
train_path2 = root+"train/neg/"
test_path1 = root+"test/pos/"
test_path2 = root+"test/neg/"
path = (train_path1, train_path2, test_path1, test_path2)
#trash = ('<br />', '.', '(', ')', '!', ';', '"', ':', '/ ', '*')


def load():
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    # 统计数据及文件数目：
    f_num = []
    for i in range(1):      # <-这里最后记得改成4！
        ls = os.listdir(path[i])
        num = 0
        for j in ls:
            if os.path.isfile(os.path.join(path[i], j)):
                num += 1
        f_num.append(num)
        for k in range(50):    # <-这里最后记得改成num!
            with open(path[i]+'%d.txt' % k) as f:
                tmp = f.read()
                # for x in trash:
                #    tmp = tmp.replace(x, ' ')
                regEX = re.compile('\\W*')
                tmp = tmp.lower()
                tmp = regEX.split(tmp)
                print(tmp)
                if i == 1 or i == 0:
                    train_data.append(tmp)
                    train_label.append(1-i)
                elif i == 2 or i == 3:
                    test_data.append(tmp)
                    test_label.append((i+1) % 2)
    return train_data, train_label, test_data, test_label, f_num


def word_to_index(train_data):
    index = {}
    for sentence in train_data:
        for word in sentence:
            if word not in index:
                index[word] = len(index)
    return index


train_data, train_label, test_data, test_label, f_num = load()
index = word_to_index(train_data)
print(f_num)
