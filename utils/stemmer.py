from nltk import SnowballStemmer


class Stemmer(object):

    __internal_stemmer__ = None
    __stemmed_words__ = dict()

    def __init__(self, language):
        self.__internal_stemmer__ = SnowballStemmer(language)

    def stem(self, word):
        if word in self.__stemmed_words__:
            return self.__stemmed_words__[word]

        stem = self.__internal_stemmer__.stem(word)
        self.__stemmed_words__[word] = stem

        return stem
