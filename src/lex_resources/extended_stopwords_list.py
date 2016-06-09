import os.path
import codecs


class ExtendedStopwordsList(object):

    stopwords_list = []

    def __init__(self, language):
        with codecs.open('src/lex_resources/extended_stopwords/' + language + '.words', 'r', 'utf-8') as f:
            for line in f:
                self.stopwords_list.append(line.strip())

def main():
    stopwords = ExtendedStopwordsList('english')
    for word in stopwords.stopwords_list:
        print(word)

if __name__ == '__main__':
    main()
