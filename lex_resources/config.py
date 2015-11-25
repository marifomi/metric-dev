from lex_resources.synonym_dictionary import SynonymDictionary
from lex_resources.contraction_dictionary import ContractionDictionary

ppdbDict = {}
word_vector = {}
posVector = {}

from nltk.corpus import stopwords
from nltk import SnowballStemmer


stemmer = SnowballStemmer('english')
synonymDictionary = SynonymDictionary('english')
contractionDictionary = ContractionDictionary('english')

punctuations = ['(','-lrb-','.',',','-','?','!',';','_',':','{','}','[','/',']','...','"','\'',')', '-rrb-']
punctuations = map(lambda x: x.encode('UTF-8'), punctuations)

stopwords = stopwords.words('english')
stopwords = map(lambda x: x.encode('UTF-8'), stopwords)
