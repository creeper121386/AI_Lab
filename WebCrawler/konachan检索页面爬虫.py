# 用于爬取konachan任意搜索页面下的全部图片，使用tag作为关键词，也可以爬取该tag下的所有图片。

from urllib.request import urlopen, urlretrieve
import re
from bs4 import BeautifulSoup
import requests

num = 1
pageNum = 30
localPath = '/run/media/why/DATA/why的程序测试/AI_Lab/DataSet/爬虫_landscape/'
rootURL = 'http://konachan.net/'
url = 'http://konachan.net/post?tags=landscape+sunset'


def craw(soup, pageNO):
    global num
    imgs = soup.find_all('span', {'class': 'plid'})
    for img in imgs:
        string = str(img.next)
        imgURL = re.split(' ', string)[-1]
        subSoup = BeautifulSoup(urlopen(imgURL), features='lxml')
        img = subSoup.find('img', {'class': 'image'})
        print('[downloading {}, page {}]\nfrom url: '.format(
            num, pageNO)+img['src'])
        urlretrieve(img['src'], localPath +
                    'picture{}(p{}).jpg'.format(num, pageNO))
        num += 1


def findNext(soup):
    next = soup.find('a', {'class': 'next_page'})
    return BeautifulSoup(urlopen(rootURL + next['href']), features='lxml')


soup = BeautifulSoup(urlopen(url), features='lxml')
for i in range(pageNum):
    craw(soup, i+1)
    soup = findNext(soup)
