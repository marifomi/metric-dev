import math

from src.lex_resources.config import *
from numpy import dot
from gensim import matutils

global stemmer
global punctuations
global ppdb_dict


__word_relatedness_alignment__ = dict()
__word_relatedness_scoring__ = dict()


def word_relatedness_alignment(word1, word2, config):

    if word1.form + '__' + word2.form in __word_relatedness_alignment__:
        return __word_relatedness_alignment__[word1.form + '__' + word2.form]

    canonical_word1 = canonize_word(word1.form)
    canonical_word2 = canonize_word(word2.form)

    similarity = None

    if canonical_word1.isdigit() and canonical_word2.isdigit() and canonical_word1 != canonical_word2:
        similarity = 0

    if similarity is not None and word1.pos.lower() == 'cd' and word2.pos.lower() == 'cd' and (not canonical_word1.isdigit() and not canonical_word2.isdigit()) and canonical_word1 != canonical_word2:
        similarity = 0

    if similarity is not None and contractionDictionary.check_contraction(canonical_word1, canonical_word2):
        similarity = config.exact

    # stopwords can be similar to only stopwords
    if similarity is not None and (canonical_word1 in stopwords and canonical_word2 not in stopwords) or (canonical_word1 not in stopwords and canonical_word2 in stopwords):
        similarity = 0

    # punctuations can only be either identical or totally dissimilar
    if similarity is not None and (canonical_word1 in punctuations or canonical_word2 in punctuations) and (not canonical_word1 == canonical_word2):
        similarity = 0

    if similarity is not None and canonical_word1 == canonical_word2:
        similarity = config.exact

    elif stemmer.stem(canonical_word1) == stemmer.stem(canonical_word2):
        similarity = config.stem

    elif word1.lemma == word2.lemma:
        similarity = config.stem

    elif synonymDictionary.checkSynonymByLemma(word1.lemma, word2.lemma) and 'synonyms' in config.selected_lexical_resources:
        similarity = config.synonym

    elif presentInPPDB(canonical_word1, canonical_word2) and 'paraphrases' in config.selected_lexical_resources:
        similarity = config.paraphrase

    elif ((not function_word(word1.form) and not function_word(word2.form)) or word1.pos[0] == word2.pos[0]) and cosine_similarity(word1.form, word2.form) > config.related_threshold and 'distributional' in config.selected_lexical_resources:

        if word1.form not in punctuations and word2.form not in punctuations:
            similarity = config.related
        else:
            similarity = 0.0

    else:
        similarity = 0.0

    __word_relatedness_alignment__[word1.form + '__' + word2.form] = similarity

    return similarity


def word_relatedness_scoring(word1, word2, scorer):

    if word1.form + '__' + word2.form in __word_relatedness_scoring__:
        return __word_relatedness_scoring__[word1.form + '__' + word2.form]

    canonical_word1 = canonize_word(word1.form)
    canonical_word2 = canonize_word(word2.form)

    if canonical_word1 == canonical_word2:
        similarity = scorer.exact

    elif contractionDictionary.check_contraction(canonical_word1, canonical_word2):
        similarity = scorer.exact

    elif word1.lemma == word2.lemma:
        similarity = scorer.stem

    elif stemmer.stem(canonical_word1) == stemmer.stem(canonical_word2):
        similarity = scorer.stem

    elif synonymDictionary.checkSynonymByLemma(word1.lemma, word2.lemma):
        similarity = scorer.synonym

    elif presentInPPDB(canonical_word1, canonical_word2):
        similarity = scorer.paraphrase

    else:
        similarity = scorer.related

    __word_relatedness_scoring__[word1.form + '__' + word2.form] = similarity

    return similarity


def word_relatedness_feature(word1, word2):

    canonical_word1 = canonize_word(word1.form)
    canonical_word2 = canonize_word(word2.form)

    if canonical_word1 == canonical_word2:
        lexSim = 'Exact'

    elif contractionDictionary.check_contraction(canonical_word1, canonical_word2):
        lexSim = 'Exact'

    elif word1.lemma == word2.lemma:
        lexSim = 'Exact'

    elif stemmer.stem(canonical_word1) == stemmer.stem(canonical_word2):
        lexSim = 'Exact'

    elif synonymDictionary.checkSynonymByLemma(word1.lemma, word2.lemma):
        lexSim = 'Synonym'

    elif presentInPPDB(canonical_word1, canonical_word2):
        lexSim = 'Paraphrase'

    else:
        lexSim = 'Distributional'

    return lexSim

def cosine_similarity(word1, word2):

    global word_vector

    if word1.lower() in word_vector.keys() and word2.lower() in word_vector.keys():
        return dot(matutils.unitvec(word_vector[word1.lower()]), matutils.unitvec(word_vector[word2.lower()]))
    else:
        return 0

def presentInPPDB(word1, word2):
    if (word1.lower(), word2.lower()) in ppdb_dict:
        return True
    if (word2.lower(), word1.lower()) in ppdb_dict:
        return True

def function_word(word):
    return (word.lower() in stopwords) or (word.lower() in punctuations) or (word.lower().isdigit())


def canonize_word(word):
    if len(word) > 1:
        canonical_form = word.replace('.', '')
        canonical_form = canonical_form.replace('-', '')
        canonical_form = canonical_form.replace(',', '').lower()
    else:
        canonical_form = word.lower()

    return canonical_form


def comparePos(pos1, pos2):

    if pos1 == pos2:
        posSim = 'Exact'
    elif pos1[0] == pos2[0]:
        posSim = 'Coarse'
    else:
        posSim = 'None'

    return posSim