from collections import Counter, defaultdict
import datetime
import sys

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from spacy.parts_of_speech import ADJ, ADV, NOUN, VERB
import spacy

from fake_news.annotate import annotate_text
from fake_news.parser import ArticleParser


nlp = spacy.load("en_core_web_sm")
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer, before="parser")


def _avg(l):
    return sum(l) / len(l) if len(l) else 0


def _filter_tokens_by_pos(doc, FILTERED_POS=[]):
    tokens = [
        t for t in doc
        if t.pos in FILTERED_POS and t.is_alpha and len(t.text) > 1
    ]
    return tokens


FILTERED_POS = [ADJ, ADV, NOUN, VERB]


def tokens4words(doc):
    return _filter_tokens_by_pos(doc, FILTERED_POS)


def split_paragraphs(text):
    return [p for p in text.split('\n')]


def text_sentiment(doc):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(doc.text)['compound']


def process_sentence(sent):
    return tokens4words(sent), text_sentiment(sent)


def filter_contained_ents(sent, ents):
    return [e for e in ents if e in sent]


def build_knowledge_dict(ents, tokens, sent):
    v = {t.lemma_: {'sent': sent, '_POS': t.pos_} for t in tokens}
    kd = {e: v for e in ents}
    return kd


def init_knowledge_dict():
    return defaultdict(lambda: defaultdict(list))


def update_knowledge_dict(KD, kd):
    for k, v in kd.items():
        for i, j in v.items():
            KD[k][i].append(j)

    return KD


def simplify_knowledge_dict(kd, annotated_ents):
    aux = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for k, v in kd.items():
        aux[k]['COUNT'] = annotated_ents[k]
        for i, j in v.items():
            aux[k]['RWORDS'][i]['sent'] = [w['sent'] for w in j]
            aux[k]['RWORDS'][i]['_POS'] = j[0]['_POS']

    return aux




def process_text(text):
    annotated_text, ents = annotate_text(text)
    annotated_ents = Counter(list(uri)[0] for _, uri in ents)
    knowledge_dict = init_knowledge_dict()

    doc = nlp(annotated_text)
    for s in doc.sents:
        tokens, sent = process_sentence(s)
        ents = filter_contained_ents(s.text, annotated_ents)
        kd = build_knowledge_dict(ents, tokens, sent)
        knowledge_dict = update_knowledge_dict(knowledge_dict, kd)

    return (
        simplify_knowledge_dict(knowledge_dict, annotated_ents),
        len(tokens4words(doc))
    )


def process_news(url,k, date=datetime.date.today()):
    article_parser = ArticleParser(url)
    if not article_parser._article.publish_date:
        article_parser.date = date
    article_parser.save()
    news = dict()
    news['SOURCE'] = article_parser.source_url
    news['DATE'] = article_parser.date
    news['NEWS'] = {'uri': url, 'title': article_parser.title}
    news['ENs'], news['RWORDS_COUNT'] = process_text(article_parser.text)
    return news


if __name__ == "__main__":
    url = sys.argv[1]
    a = process_news(url)
    print(a)

    d = datetime.date(2019, 10, 3)
    k=1
    aa = []
    for i in aux:
        print(i)
        if i[0] not in aa:
            try:
                print("si")
                process_news(i[0],k,d)
                k=k+1
                print(k)
                aa.append(i[0])
            except:
                pass


