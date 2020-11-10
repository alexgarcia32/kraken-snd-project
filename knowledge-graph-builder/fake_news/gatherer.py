import argparse
import datetime
import dateutil.parser
import multiprocessing
import os
import sys

from fake_news.neo4j_conn import KnowledgeGraph
from fake_news.process import process_news
from newsapi import NewsApiClient

# Init
NEO4J_HOST = os.getenv('NEO4J_HOST', default='bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', default='neo4j')
NEO4J_PASS = os.getenv('NEO4J_PASS', default='')

newsapi = NewsApiClient(api_key='c438ac7b7042471294fdb21c35237fb0')


def _process_news(url, date):
    try:
        neo = KnowledgeGraph(NEO4J_HOST, NEO4J_USER, NEO4J_PASS)
        date = datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ')
        news = process_news(url, date=date)
        #
        neo.update_kg(news)
        print(url)
        return news
    except Exception:
        pass


def main(arguments):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '-s', "--start_date",
        help="The Start Date - format YYYY-MM-DD. Default: 1 week ago.",
        type=dateutil.parser.isoparse,
        default=datetime.date.today() + datetime.timedelta(days=-7)
    )
    parser.add_argument(
        '-e', "--end_date",
        help="The End Date format YYYY-MM-DD (Inclusive). Default: Today",
        type=dateutil.parser.isoparse,
        default=datetime.date.today() + datetime.timedelta(days=1)
    )
    parser.add_argument(
        '-q', "--query",
        help="Query. Default: brexit",
        default='+brexit'
    )

    args = parser.parse_args(arguments)

    sources = [
        'abc-news',
        'abc-news-au',
        'associated-press',
        'bbc-news',
        'breitbart-news',
        'cbc-news',
        'cbs-news',
        'cnn',
        'fox-news',
        'google-news',
        'google-news-uk',
        'independent',
        'metro',
        'mirror',
        'national-review',
        'nbc-news',
        'news24',
        'news-com-au',
        'newsweek',
        'new-york-magazine',
        'reuters',
        'rte',
        'the-american-conservative',
        'the-huffington-post',
        'the-new-york-times',
        'the-telegraph',
        'the-washington-post',
        'the-washington-times',
        'time',
        'usa-today',
    ]
    aux = []
    for j, s in enumerate(sources):
        all_articles = newsapi.get_everything(
            q="+brexit",
            sources=",".join(sources),
            from_param='2019-09-27',
            to='2019-09-27',
            language='en',
            sort_by='relevancy',
            page_size=100,
            page=1
        )
        aux += [
            (a['url'], a['publishedAt'])
            for i, a in enumerate(all_articles['articles'])
        ]

    with multiprocessing.Pool(processes=20) as pool:
        pool.starmap(_process_news, aux)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
