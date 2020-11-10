from newspaper import Article
import justext


class ArticleParser:
    def __init__(self, url):
        self._article = Article(url)
        self._article.download()
        self._article.parse()
        self._text = None

    @property
    def title(self):
        return self._article.title

    @property
    def text(self):
        if not self._text:
            if self._article.is_valid_body():
                self._text = self._article.text
            else:
                self._text = '\n'.join(
                    p.text for p in justext.justext(
                        self._article.html, justext.get_stoplist("English")
                    )
                )

        return self._text

    @property
    def source_url(self):
        return self._article.source_url

    @property
    def date(self):
        return self._article.publish_date

    @date.setter
    def date(self, value):
        self._article.publish_date = value

    def save(self):
        f = open("../news/"+self.title.replace(" ", "_"), "w")
        f.write(self.date.isoformat() + "\n")
        f.write(self.title + "\n")
        f.write(self.source_url + "\n")
        f.write(self.text + "\n")
        f.close()
