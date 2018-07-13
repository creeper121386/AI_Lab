from urllib.request import urlopen, urlretrieve
from bs4 import BeautifulSoup
import re 


finder = re.compile(r'//gel.*view&id=.*')

url = "https://gelbooru.com/index.php?page=post&s=list&tags=landscape+sunset"
response = BeautifulSoup(urlopen(url), features='lxml')
imgs = response.find_all('a')

for x in imgs:
    img = finder.match(x['href'])
    if img:
        imgURL = img.group()
        soup = BeautifulSoup(urlopen('https:'+imgURL), features='lxml')
        data = soup.find('img')
        print(data['src'])
        
        