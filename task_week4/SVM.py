import function as F
import kernel as k
import numpy as np
normalize = True
loopTimes = 4  # 更新alpha的循环次数
kernel = 0  # 使用哪个核(0代表不使用，核函数编号见列表K)
C = 1e5
K = [None, k.liner, k.multi, k.Gauss, k.Laplace, k.sigmoid]
# 核函数相关参数在kernel.py中进行调节


def cal_w(feat, alpha, trainData, trainLabel):
    return np.dot((alpha*trainLabel).T, trainData)


def pred(feat, b, alpha, kernel, trainData, trainLabel):
    if kernel:
        return b + np.dot((trainLabel*alpha).T, np.array([K[kernel](x, feat.T) for x in trainData]))
    else:
        return b + np.dot((trainLabel*alpha).T, np.dot(trainData, feat))


def chooseAlpha(trainData, trainLabel, b, alpha, index):
    E = 0
    num = len(index)
    predLst = [pred(trainData[x], b, alpha, kernel,
                    trainData, trainLabel) for x in index]
    for x in range(num):
        ix = index[x]
        if alpha[ix] > 0 and alpha[ix] < C:
            preE = E
            E = max(E, np.abs(1-predLst[x]*trainLabel[ix]))
            if preE != E:
                i = x
    if E == 0:
        for x in range(num):
            ix = index[x]
            preE = E
            if alpha[ix] == 0:
                E = max(E, 1-predLst[x]*trainLabel[ix])
            elif alpha[ix] == C:
                E = max(E, predLst[x]*trainLabel[ix]-1)
            if preE != E:
                i = x
    if E == 0:
        i = 0
    E = [predLst[x]-trainLabel[index[x]] for x in range(num)]
    j = np.argmax(np.abs(np.array(E) - E[i]))
    i = index[i]
    j = index[j]
    return i, j


def train(trainData, trainLabel, num, kernel):
    # alpha = np.random.randint(0, C, (num, ))
    print('*** data training start ***')
    func = K[kernel] if kernel else np.dot
    alpha = np.zeros(num)
    for t in range(loopTimes):
        index = [x for x in range(num)]
        b = 0
        while len(index):
            i, j = chooseAlpha(trainData, trainLabel, b, alpha, index)
            Ei = pred(trainData[i], b, alpha, kernel,
                      trainData, trainLabel)-trainLabel[i]
            Ej = pred(trainData[j], b, alpha, kernel,
                      trainData, trainLabel)-trainLabel[j]
            preI = alpha[i].copy()
            preJ = alpha[j].copy()
            yi = trainLabel[i]
            yj = trainLabel[j]
            xi = trainData[i]
            xj = trainData[j]
            if yi == yj:
                L = max(0, alpha[j]+alpha[i]-C)
                H = min(C, alpha[j]+alpha[i])
            else:
                L = max(0, alpha[j]-alpha[i])
                H = min(C, C+alpha[j]-alpha[i])
            # c = -sum([trainLabel[k]*alpha[k]
            #          for k in range(num) if k != i and k != j])
            eta = 2*func(xi, xj.T)-func(xi, xi.T)-func(xj, xj.T)
            if L != H and eta < 0:
                alpha[j] -= yj * (Ei-Ej)/eta
                if alpha[j] > H:
                    alpha[j] = H
                if alpha[j] < L:
                    alpha[j] = L
                if np.abs(alpha[j]-preJ) < 1e-3:
                    index.remove(i)
                    continue
                alpha[i] += yi*yj*(preJ-alpha[j])
                b1 = b-Ei-yi*(alpha[i]-preI)*func(xi, xi.T) -\
                    yj*(alpha[j]-preJ)*func(xi, xj.T)
                b2 = b-Ej-yi*(alpha[i]-preI)*func(xi, xj.T) - \
                    yj*(alpha[j]-preJ)*func(xj, xj.T)
                if alpha[i] > 0 and alpha[i] < C:
                    b = b1
                elif alpha[j] > 0 and alpha[j] < C:
                    b = b2
                else:
                    b = (b1+b2)/2
                index.remove(i)
            else:
                index.remove(i)
                continue
        print('training: epoch No.', t)
    print('*** data training finish ***')
    return alpha, b


def test(testData, testLabel, trainData, trainLabel, alpha, b, kernel):
    print('testing...')
    count = 0
    for x, y in zip(testData, testLabel):
        if not kernel:
            w = cal_w(x, alpha, trainData, trainLabel)
            predLabel = np.dot(w, x.T)+b
        else:
            predLabel = pred(x, b, alpha, kernel, trainData, trainLabel)
        count += 1 if y*predLabel > 0 else 0
    return count/len(testLabel)


trainData, trainLabel, num = F.load(0, normalize)
testData, testLabel, _ = F.load(1, normalize)
alpha, b = train(trainData, trainLabel, num, kernel)
acc = test(testData, testLabel, trainData, trainLabel, alpha, b, kernel)
print(acc)
