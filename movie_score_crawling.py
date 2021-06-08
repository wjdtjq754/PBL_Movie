from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import time
url_base='https://movie.naver.com/'
url_sub="movie/sdb/rank/rmovie.nhn?sel=pnt&tg=0&date=20210325&page=1"
page=urlopen(url_base+url_sub)

soup=BeautifulSoup(page,'html.parser')

import urllib
from tqdm import tqdm_notebook
movie_name=[]
movie_point=[]
page =range(1,41)
for p in page:
    html='https://movie.naver.com/movie/sdb/rank/rmovie.nhn?sel=pnt&tg=0&date=20210325&page={page}'
    response = urlopen(html.format(page=p))
    soup = BeautifulSoup(response,'html.parser')

    movie_name.extend([i.a.string for i in soup.find_all('div','tit5')])
    movie_point.extend([i.string for i in soup.find_all('td','point')])

    time.sleep(0.5)

movie = pd.DataFrame({'name':movie_name, 'point':movie_point})
movie['point'] = movie['point'].astype(float)
movie.to_csv('20210325_movie_point.csv',sep=',',encoding='utf-8-sig')
