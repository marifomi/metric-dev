from nltk.corpus import stopwords

from src.lex_resources.synonym_dictionary import SynonymDictionary
from src.lex_resources.contraction_dictionary import ContractionDictionary
from src.lex_resources.extended_stopwords import ExtendedStopwords
from src.utils.stemmer import Stemmer


ppdb_dict = {}
word_vector = {}
pos_vector = {}

stemmer = Stemmer('english')
synonymDictionary = SynonymDictionary('english')
contractionDictionary = ContractionDictionary('english')
extended_stopwords = ExtendedStopwords('english')

punctuations = ['%', '(', '-lrb-', '.', ',', '-', '?', '!', ';', '_', ':', '{', '}', '[', '/', ']', '...', '"', '\'', ')', '-rrb-']
cobalt_stopwords = stopwords.words('english')
