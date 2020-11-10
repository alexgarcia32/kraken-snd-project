import os
import sys
sys.path.append(os.path.abspath('../knowledge-graph-builder/'))
from fake_news.neo4j_conn import KnowledgeGraph
import gatherer

# Init
NEO4J_HOST = os.getenv('NEO4J_HOST', default='bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', default='neo4j')
NEO4J_PASS = os.getenv('NEO4J_PASS', default='')

neo = KnowledgeGraph(NEO4J_HOST, NEO4J_USER, NEO4J_PASS)

def get_related_news_by_date(nlist):
    related_news = []
    for i in nlist:
        related_news.append(neo.get_sentiment_by_en((i['DATE'])))
    return related_news



if __name__ == '__main__':

    a = ["https://abcnews.go.com/International/guide-biggest-political-players-ongoing-brexit-saga/story?id=65428846"]
    print(get_related_news_by_date(gatherer.process_urllist(a)))
