import numpy as np
import os
root = "/media/why/DATA/why的程序测试/AI_Lab/NLP/"
train_path1 = root+"train/pos/"
train_path2 = root+"train/neg/"
test_path1 = root+"test/pos/"
test_path2 = root+"test/neg/"
path = (train_path1, train_path2, test_path1, test_path2)
trash = ('<br />', '.', '(', ')', '!', ';', '"', ':', '/ ', '*', '-')
p_c = 0.5  # 文档是pos或者neg的概率


def load():
    print("Loading data...")
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    # 统计数据及文件数目：
    f_num = []
    for i in range(4):      # <-这里最后记得改成4！
        ls = os.listdir(path[i])
        num = 0
        for j in ls:
            if os.path.isfile(os.path.join(path[i], j)):
                num += 1
        # num = 500        # <-这里最后记得改!
        f_num.append(num)
        for k in range(num):
            with open(path[i]+'%d.txt' % k) as f:
                tmp = f.read()
                for x in trash:
                    tmp = tmp.replace(x, ' ')
                tmp = tmp.lower()
                tmp = tmp.split()
                if i == 1 or i == 0:
                    train_data.append(tmp)
                    train_label.append(1-i)
                elif i == 2 or i == 3:
                    test_data.append(tmp)
                    test_label.append((i+1) % 2)
    return train_data, train_label, test_data, test_label, f_num


def word_to_list(train_data):  # 建立词集
    word_list = {}
    for sentence in train_data:
        for word in sentence:
            if word not in word_list:
                word_list[word] = len(word_list)
    return word_list


def data_to_vector(word_list, data, num):  # 将每一篇文本转化为词向量
    vector = []
    for y in data:
        tmp = [0]*len(word_list)
        for x in y:
            if x in word_list:
                index = word_list[x]
                tmp[index] += 1
        vector.append(tmp)
    print("Done.")
    return vector


def train(train_vector, train_label, doc_num):
    print("Now training...")
    doc_len = len(train_vector[0])
    pos_count = np.ones(doc_len)  # 计算pos/neg训练集中各个词条分别出现的次数
    neg_count = np.ones(doc_len)
    pos_num = 1
    neg_num = 1  # 计算pos/neg训练集中的总单词数
    for i in range(doc_num):
        if train_label[i] == 1:
            pos_count += train_vector[i]
            pos_num += sum(train_vector[i])
        elif train_label[i] == 0:
            neg_count += train_vector[i]
            neg_num += sum(train_vector[i])
    p1_vec = pos_count/pos_num
    p0_vec = neg_count/neg_num
    both_num = pos_num+neg_num
    both_count = pos_count+neg_count
    pw_vec = both_count/both_num  # 由于词集包含了测试集数据，计算训练集中的概率时，会有元素为0
    return p1_vec, p0_vec, pw_vec


def cal(doc_vec, p_vec, pw_vec, doc_len):
    tmp = p_c
    for i in range(doc_len):
        if doc_vec[i] != 0:
            tmp *= (p_vec[i]/pw_vec[i])
    return tmp


def test(test_vector, test_label, p1_vec, p0_vec, pw_vec, doc_num):
    print("Now testing...")
    error = 0
    doc_len = len(test_vector[0])
    for i in range(doc_num):
        p1 = cal(test_vector[i], p1_vec, pw_vec, doc_len)
        p0 = cal(test_vector[i], p0_vec, pw_vec, doc_len)
        result = 1 if p1 > p0 else 0
        if result != test_label[i]:
            error += 1
    return error/doc_num


train_data, train_label, test_data, test_label, f_num = load()
word_list = word_to_list(train_data+test_data)
train_vector = data_to_vector(word_list, train_data, 2*f_num[0])
test_vector = data_to_vector(word_list, test_data, 2*f_num[2])
train_data = np.array(train_data)
test_data = np.array(test_data)
p1_vec, p0_vec, pw_vec = train(train_vector, train_label, 2*f_num[0])
error = test(test_vector, test_label, p1_vec, p0_vec, pw_vec, 2*f_num[2])
print('error rate=', error)
