#%%
from urllib.request import urlopen
import re
from bs4 import BeautifulSoup

pageNum = 30
localPath = '/run/media/why/DATA/why的程序测试/AI_Lab/DataSet/爬虫_landscape/'

rootURL = 'http://konachan.net/'
url = 'http://konachan.net/post?tags=clouds+landscape'

imgFinder = re.compile(r'.*\.jpg')

#%%
html = urlopen(url).read()
soup = BeautifulSoup(html, features='lxml')
content = soup.find_all('img')
for tag in content:
    match = imgFinder.match(tag['src'])
    print(type(tag))
    if match:
        print(match.group())