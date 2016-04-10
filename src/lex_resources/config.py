from nltk.corpus import stopwords

from src.lex_resources.synonym_dictionary import SynonymDictionary
from src.lex_resources.contraction_dictionary import ContractionDictionary
from src.utils.stemmer import Stemmer


ppdb_dict = {}
word_vector = {}
pos_vector = {}

stemmer = Stemmer('english')
synonymDictionary = SynonymDictionary('english')
contractionDictionary = ContractionDictionary('english')

punctuations = ['%', '(', '-lrb-', '.', ',', '-', '?', '!', ';', '_', ':', '{', '}', '[', '/', ']', '...', '"', '\'', ')', '-rrb-']
punctuations = list(map(lambda x: x.encode('UTF-8'), punctuations))

stopwords = stopwords.words('english')
stopwords = list(map(lambda x: x.encode('UTF-8'), stopwords))
