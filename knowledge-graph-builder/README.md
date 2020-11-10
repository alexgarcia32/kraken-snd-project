Run the neo4j server using docker:

```bash
docker pull neo4j
docker run --publish=7474:7474 --publish=7687:7687 --volume=$HOME/workspace/Fake_News/neo4j/data:/data --name=FN_neo4j neo4j
```

Use neo4j_conn to recover all the sentiments:

```python
from fake_news.neo4j_conn import KnowledgeGraph

kg = KnowledgeGraph('bolt://193.147.61.144:7777', 'neo4j', 'dslab2019')
d = kg.get_sentiments_magic_method('2019-09-11')
```
