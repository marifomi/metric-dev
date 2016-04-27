import os.path
import codecs


class ExtendedStopwords(object):

    stopwords_list = []

    def __init__(self, language):
        with codecs.open(os.path.expanduser('src/lex_resources/extended_stopwords/' + language + '.words'), 'r', 'utf-8') as f:
            for line in f:
                self.stopwords_list.append(line.strip())
