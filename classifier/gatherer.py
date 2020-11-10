import os
import sys
#import glob
sys.path.append(os.path.abspath('../knowledge-graph-builder/'))
from fake_news.process import process_news
from fake_news.process import process_text
import datetime


def process_urllist(list):
    procesed_news = []
    for news in list:
        procesed_news.append( process_news(news) )
    return procesed_news

def process_textlist(list,fecha):
    procesed_news = []
    for i in list:
        news = dict()
        news['SOURCE'] = "Texto plano"
        news['DATE'] = fecha
        news['NEWS'] = {'uri': "twitter", 'title': "twitter"}
        print(i)
        news['ENs'], news['RWORDS_COUNT'] = process_text(i)
        procesed_news.append( news)
    return procesed_news

if __name__ == "__main__":

    a = ["https://abcnews.go.com/International/guide-biggest-political-players-ongoing-brexit-saga/story?id=65428846","https://abcnews.go.com/International/guide-biggest-political-players-ongoing-brexit-saga/story?id=65428846"]
    b = process_urllist(a)
    print(b)
