from nltk.corpus import stopwords

from lex_resources.synonym_dictionary import SynonymDictionary
from lex_resources.contraction_dictionary import ContractionDictionary
from lex_resources.extended_stopwords_list import ExtendedStopwordsList
from utils.stemmer import Stemmer


ppdb_dict = {}
word_vector = {}
pos_vector = {}

stemmer = Stemmer('english')
synonymDictionary = SynonymDictionary('english')
contractionDictionary = ContractionDictionary('english')
extended_stopwords = ExtendedStopwordsList('english')

punctuations = ['%', '(', '-lrb-', '.', ',', '-', '?', '!', ';', '_', ':', '{', '}', '[', '/', ']', '...', '"', '\'', ')', '-rrb-']
cobalt_stopwords = stopwords.words('english')
