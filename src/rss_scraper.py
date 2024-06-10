import requests
from bs4 import BeautifulSoup
import pandas as pd

urls = ['https://www.focus-news.net/rss.php',
'https://www.dnes.bg/rss.php?today', 
'https://www.dnes.bg/rss.php?cat=1', 
'https://www.dnes.bg/rss.php?cat=142', 
'https://www.dnes.bg/rss.php?cat=150',
'https://www.dnes.bg/rss.php?cat=4',
'https://www.dnes.bg/rss.php?cat=6',
'https://www.dnes.bg/rss.php?cat=2',
'https://fakti.bg/feed',
'https://fakti.bg/feed/life',
'https://fakti.bg/feed/technozone',
'https://fakti.bg/feed/avto',
'https://fakti.bg/feed/biznes',
'https://fakti.bg/feed/bulgaria',
'https://fakti.bg/feed/mnenia',
'https://fakti.bg/feed/krimi',
'https://fakti.bg/feed/sport',
'https://fakti.bg/feed/video',
'https://fakti.bg/feed/world',
'https://fakti.bg/feed/imoti',
'https://fakti.bg/feed/zdrave',
'https://fakti.bg/feed/kultura-art',
'https://fakti.bg/feed/razsledvania'
]

output = []
for feed in urls:
    resp = requests.get(feed, timeout=20)
    soup = BeautifulSoup(resp.content, features='xml')
    for entry in soup.find_all('item'):
        title = entry.find('title').text
        output.append({'news_title': title, 'label': 1})

df = pd.DataFrame(output)
df.to_csv('../data/raw/results_feed.csv', index=False)
