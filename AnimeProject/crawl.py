import requests
from bs4 import BeautifulSoup
import re
from urllib.request import urlretrieve
import multiprocessing as mp
import os
ROOT_URL = 'https://db.loveliv.es/card/number/'
SAVE_PATH = '/run/media/why/DATA/why的程序测试/AI_Lab/DataSet/AnimeProject/LL'
SEEN_PATH = '/run/media/why/DATA/why的程序测试/AI_Lab/DataSet/AnimeProject/LL_old'
MP = True
SAVE_METHOD = 0

class Spider(object):
    def __init__(self):
        self.numList = set([str(x) for x in range(1, 1660)])
        fs = os.listdir(SEEN_PATH)
        self.seen = set([x[:-4] for x in fs])
        self.unseen = self.numList - self.seen
        self.avatorFinder = re.compile(
            r'//r\.llsif\.win/assets/image/units/u_rankup_icon_[0-9]*\.png')

    def getUrl(self, num):
        # num = str(num)
        if num not in self.unseen:
            return None
        html = requests.get(ROOT_URL + num)
        html = str(html.content)
        match = self.avatorFinder.findall(html)
        if len(match):
            return 'https:' + match[0]
        else:
            return None

    def getImg(self, imgUrl, num):
        # num = str(num)
        if SAVE_METHOD == 1:
            r = requests.get(imgUrl, stream=False)
            with open('{}/{}.jpg'.format(SAVE_PATH, num), 'wb') as f:
                f.write(r.content)
        elif SAVE_METHOD == 0:
            urlretrieve(imgUrl, '{}/{}.jpg'.format(SAVE_PATH, num))

        # self.unseen.remove(num)
        # self.seen.add(num)
        print('[Download] image num: %04d\n from URL: %s' % (int(num), imgUrl))

    def crawl(self, num):
        res = self.getUrl(num)
        if res:
            self.getImg(res, num)

    def crawl_all(self):
        if MP:
            pool = mp.Pool()
            pool.map(self.crawl, self.unseen)
        else:
            for num in self.unseen:
                self.crawl(num)

if __name__ == '__main__':
    spider = Spider()
    spider.crawl_all()

