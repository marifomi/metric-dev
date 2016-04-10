from nltk.corpus import stopwords
from nltk import SnowballStemmer

from src.lex_resources.synonym_dictionary import SynonymDictionary
from src.lex_resources.contraction_dictionary import ContractionDictionary


ppdb_dict = {}
word_vector = {}
pos_vector = {}

stemmer = SnowballStemmer('english')
synonymDictionary = SynonymDictionary('english')
contractionDictionary = ContractionDictionary('english')

punctuations = ['%', '(', '-lrb-', '.', ',', '-', '?', '!', ';', '_', ':', '{', '}', '[', '/', ']', '...', '"', '\'', ')', '-rrb-']
punctuations = map(lambda x: x.encode('UTF-8'), punctuations)

stopwords = stopwords.words('english')
stopwords = map(lambda x: x.encode('UTF-8'), stopwords)
