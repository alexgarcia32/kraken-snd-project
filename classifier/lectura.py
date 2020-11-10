import os
import sys

import spacy
nlp = spacy.load('en_core_web_sm')
from fake_news.process import process_text

from senticnet.senticnet import SenticNet
from spacy.symbols import ADJ, NOUN, PRON, VERB


def previous_filter(df,rwords_news_min,rwords_en_min):
    # d: news dictionary
    # rwords_news_min: min number of words for news to be admitted
    # rwords_en_min: min number of RW for EN to be admitted
    en_to_del = [] #  EN to delete
    mistake_EN = ['http://dbpedia.org/resource/JavaScript','http://dbpedia.org/resource/Getty_Images','http://dbpedia.org/resource/HTML5']
    for news in list(df.keys()): # for every news
        # To delete news with less than rwords_news_min words
        if df[news]['RWORDS_COUNT']<rwords_news_min:
            del df[news]
        else:
            for EN in df[news]['ENs']: # for EN in news
                # if JavaScript, Getty_Images or HTML5 is unterstood as EN: delete
                if EN in mistake_EN:
                    en_to_del.append([news, EN])

                a = list(df[news]['ENs'][EN]['RWORDS'])
                b = nlp(" ".join(a))  # to string (nlp specifications)
                # To delete stop words and words with len(words)<3
                stop_b = [str(x) for x in b if x.is_stop or len(str(x)) < 3]
                if len(stop_b)>0:
                    for k in stop_b:
                        if k in df[news]['ENs'][EN]['RWORDS']:
                            del df[news]['ENs'][EN]['RWORDS'][k]
                # Once stop words have been removed, we delete EN with less than rwords_en_min RW
                a1 = list(df[news]['ENs'][EN]['RWORDS'])
                if len(a1) < rwords_en_min:
                    en_to_del.append([news, EN])
    for i in en_to_del:
        del df[i[0]]['ENs'][i[1]]

    return(df)


def read_fake(filepath):
    news = dict()
    news['SOURCE'] = "fake news 1"
    lines = []
    with open(filepath, encoding="utf8") as fp:
        cnt = 1
        line = fp.readline()
        while line:
            if line != "\n":
                lines.append(line)
            cnt += 1
            line = fp.readline()
    news['ENs'], news['RWORDS_COUNT'] = process_text(" ".join(lines))

    for k, v in news['ENs'].items():
        news["ENs"][k]["FREQ"] = news["ENs"][k]["COUNT"]
        del news["ENs"][k]["COUNT"]

    for k, v in news['ENs'].items():
        for k2, v2 in news['ENs'][k]["RWORDS"].items():
            lenlist = len(news['ENs'][k]["RWORDS"][k2]["sent"])
            news['ENs'][k]["RWORDS"][k2] = [sum(news['ENs'][k]["RWORDS"][k2]["sent"]) / lenlist, lenlist,
                                            news['ENs'][k]["RWORDS"][k2]["_POS"]]
    return news

