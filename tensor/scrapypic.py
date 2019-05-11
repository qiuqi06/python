import re
import urllib
def getHtml(url):
    page= urllib.urlopen(url)
    html=page.read()
    return html
def getImg(html):
    reg=r'src="(.*?\.jpg" width'
    imgre=re.compile(reg)
    imglist=re.findall(imgre,html)
    x=0
    for imgurl in imglist:
        urllib.urlretrieve(imgurl,"%s.jpg"%x)
        x+=1
import urllib.request
def funcc():
    weburl = "http://www.douban.com/"
    webheader = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
    req = urllib.request.Request(url=weburl, headers=webheader)
    webPage = urllib.request.urlopen(req)
    data = webPage.read()
    data = data.decode('UTF-8')
    # print(data)
    # print(type(webPage))
    # print(webPage.geturl())
    # print(webPage.info())
    print(webPage.getcode())

