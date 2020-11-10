from neo4j import GraphDatabase


class KnowledgeGraph(object):
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def update_kg(self, kg):
        with self._driver.session() as session:
            session.write_transaction(
                self._create_news, kg['SOURCE'], kg['NEWS'],
                kg['DATE'].strftime('%Y-%m-%d'), kg['RWORDS_COUNT']
            )
            for en, v in kg['ENs'].items():
                session.write_transaction(
                    self._create_en, kg['NEWS'], en, v['COUNT']
                )
                for w, j in v['RWORDS'].items():
                    session.write_transaction(
                        self._create_word, en, w, j['_POS'],
                        kg['NEWS']['uri'], j['sent']
                    )

    def get_en_sentiment(self, en):
        with self._driver.session() as session:
            return session.write_transaction(self._get_en_sentiment, en)

    def get_news(self):
        with self._driver.session() as session:
            return session.write_transaction(self._get_news)

    def get_sentiment_by_en(self, date=None):
        with self._driver.session() as session:
            return session.write_transaction(self._get_sentiment_by_en,
                                             date=date)

    def get_related_news_by_date(self, date):
        with self._driver.session() as session:
            return session.write_transaction(self._get_related_news_by_date,
                                             date=date)

    def get_sentiments_magic_method(self, date):
        with self._driver.session() as session:
            return session.write_transaction(self._get_sentiments_magic_method,
                                             date=date)

    def get_data_by_source(self, date):
        with self._driver.session() as session:
            return session.write_transaction(self._get_data_by_source,
                                             date=date)

    @staticmethod
    def _get_news(tx):
        cmd = (
            'MATCH (n:NEWS)'
            'RETURN n'
        )
        return tx.run(cmd).value()

    @staticmethod
    def _create_news(tx, source, news, date, rw_c):
        cmd = (
            'MERGE (s:SOURCE {uri: $source })'
            'MERGE (n:NEWS {uri: $n_uri , title: $n_title, rw_count: $rw_c})'
            'MERGE (d:DATE {value: $date })'
            'CREATE (s)-[r:PUBLISHES]->(n)-[:ON]->(d)'
            'return r'
        )
        return tx.run(
            cmd, source=source, n_uri=news['uri'], n_title=news['title'],
            date=date, rw_c=rw_c)

    @staticmethod
    def _create_en(tx, news, en, count):
        cmd = """
            MERGE (n:NEWS {uri: $n_uri })
            MERGE (e:EN {uri: $en })
            CREATE (n)-[r:MENTIONS {count: $count}]->(e)
            return r
        """
        return tx.run(cmd, n_uri=news['uri'], en=en, count=count)

    @staticmethod
    def _create_word(tx, en, word, _POS, news_id, sent):
        cmd = """
            MERGE (e:EN {uri: $en })
            MERGE (w:WORD {lemma: $word, _POS: $_POS})
            CREATE (e)-[r:SAYS {news_id: $news_id, sent: $sent } ]->(w)
            return r
        """
        return tx.run(cmd, en=en, word=word, _POS=_POS, news_id=news_id,
                      sent=sent)

    @staticmethod
    def _get_sentiment_by_en(tx, date):
        if date:
            cmd = (
                "MATCH (d:DATE {value: $date})-[:ON]-(n:NEWS)"
                "WITH collect(n.uri) as uris"
                "MATCH p=(e:EN)-[r:SAYS]-(w:WORD)"
                "WHERE r.news_id in uris"
                "RETURN e.uri as en, avg(reduce(totalSent = 1, "
                "    s IN r.sent| totalSent + s) / size(r.sent)) as sent"
            )
            return tx.run(cmd, date=date).data()
        cmd = (
            "MATCH p=(e:EN)-[r:SAYS]-(w:WORD)"
            "RETURN e.uri as en, avg(reduce(totalSent = 1, "
            "    s IN r.sent| totalSent + s) / size(r.sent)) as sent"
        )
        return tx.run(cmd).data()

    @staticmethod
    def _get_related_news_by_date(tx, date):
        cmd = (
            "MATCH (d:DATE {value: $date})-[:ON]-(n:NEWS)"
            "RETURN n"
        )
        return tx.run(cmd, date=date).data()

    @staticmethod
    def _get_sentiments_magic_method(tx, date):
        cmd = (
            "MATCH (d:DATE {value: $date})-[:ON]-(n:NEWS)-"
            "[:PUBLISHES]-(s:SOURCE) "
            "WITH n, s.uri as s "
            "MATCH (n)-[p:MENTIONS]-(e:EN)-[m:SAYS {news_id: n.uri}]-(w:WORD) "
            "WITH n, s, p, e.uri as e, w as w, "
            "   reduce(t = 0, n IN m.sent | t + n) as sents, "
            "   size(m.sent) as s_c "
            "WITH n, s, e, {FREQ: p.count, RWORDS: apoc.map.fromLists("
            "   collect(w.lemma), collect([sents / s_c, s_c, w._POS]))} as x "
            "WITH n, s, apoc.map.fromLists(collect(e), collect(x)) as ents "
            "WITH n.uri as n, collect({SOURCE: s, ENs: ents, "
            "   RWORDS_COUNT: n.rw_count})[0] as x "
            "RETURN apoc.map.fromLists(collect(n), collect(x)) as s "
        )
        return tx.run(cmd, date=date).data()[0]['s']

    @staticmethod
    def _get_data_by_source(tx, date):
        cmd = (
            "MATCH (d:DATE {value: $date})-[:ON]-(n:NEWS)- "
            "[:PUBLISHES]-(s:SOURCE)  "
            "WITH n, s.uri as s  "
            "MATCH (n)-[p:MENTIONS]-(e:EN)-[m:SAYS {news_id: n.uri}]-(w:WORD) "
            "WITH s, p, e.uri as e, w as w,  "
            "   reduce(t = 0, n IN m.sent | t + n) as sents,  "
            "   size(m.sent) as s_c  "
            "WITH s, e, {RWORDS: apoc.map.fromLists( "
            "   collect(w.lemma), collect([sents / s_c, s_c, w._POS]))} as x  "
            "WITH s, apoc.map.fromLists(collect(e), collect(x)) as ents  "
            "RETURN apoc.map.fromLists(collect(s), collect(ents)) as s"
        )
        return tx.run(cmd, date=date).data()[0]['s']
