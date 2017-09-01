from lex_resources.config import *
from numpy import dot
from gensim import matutils

global stemmer
global punctuations
global ppdb_dict
global cobalt_stopwords

__word_relatedness_alignment__ = dict()
__word_relatedness_scoring__ = dict()


def word_relatedness_alignment(word1, word2, config):

    double_check = 0

    if word1.form + '__' + word2.form in __word_relatedness_alignment__:
        return __word_relatedness_alignment__[word1.form + '__' + word2.form]

    canonical_word1 = canonize_word(word1.form)
    canonical_word2 = canonize_word(word2.form)

    similarity = None

    # First check the cases where the words do not match or do match for sure

    # Digits can be aligned only if they are identical

    if contractionDictionary.check_contraction(canonical_word1, canonical_word2):
        similarity = config.exact

    if canonical_word1.isdigit() and canonical_word2.isdigit() and canonical_word1 != canonical_word2:
        similarity = 0

    # stopwords can be similar to only stopwords
    if similarity is None and ((function_word(canonical_word1) and not function_word(canonical_word2)) or (function_word(canonical_word2) and not function_word(canonical_word1))):
        similarity = 0

    # punctuations can only be either identical or totally dissimilar
    if similarity is None and (canonical_word1 in punctuations or canonical_word2 in punctuations) and (not canonical_word1 == canonical_word2):
        similarity = 0

    if similarity is not None:
        __word_relatedness_alignment__[word1.form + '__' + word2.form] = similarity
        return similarity

    if canonical_word1 == canonical_word2:
        similarity = config.exact

    elif stemmer.stem(canonical_word1) == stemmer.stem(canonical_word2):
        similarity = config.stem

    elif word1.lemma == word2.lemma:
        similarity = config.stem

    elif synonymDictionary.checkSynonymByLemma(word1.lemma, word2.lemma) and 'synonyms' in config.selected_lexical_resources:
        similarity = config.synonym

    elif presentInPPDB(canonical_word1, canonical_word2) and 'paraphrases' in config.selected_lexical_resources:
        similarity = config.paraphrase

    elif cosine_similarity(word1.form, word2.form) > config.related_threshold and 'distributional' in config.selected_lexical_resources:
        double_check = 1

        if (word1.form not in punctuations and word2.form not in punctuations) and ((not function_word(word1.form) and not function_word(word2.form)) or word1.pos[0] == word2.pos[0]):
            similarity = config.related
        else:
            similarity = 0.0

    else:
        similarity = 0.0

    if double_check == 0:
        __word_relatedness_alignment__[word1.form + '__' + word2.form] = similarity

    return similarity


def word_relatedness_alignment_stanford(word1, word2, config):

    double_check = 0

    if word1.form + '__' + word2.form in __word_relatedness_alignment__:
        return __word_relatedness_alignment__[word1.form + '__' + word2.form]

    canonical_word1 = canonize_word(word1.form)
    canonical_word2 = canonize_word(word2.form)

    similarity = None
    similarity_type = None

    # First check the cases where the words do not match or do match for sure

    # Digits can be aligned only if they are identical

    if contractionDictionary.check_contraction(canonical_word1, canonical_word2):
        similarity = config.exact
        similarity_type = 'Exact'

    if canonical_word1.isdigit() and canonical_word2.isdigit() and canonical_word1 != canonical_word2:
        similarity = 0

    if similarity is None and word1.pos.lower() == 'cd' and word2.pos.lower() == 'cd' and (not canonical_word1.isdigit() and not canonical_word2.isdigit()) and canonical_word1 != canonical_word2:
        similarity = 0

    # stopwords can be similar to only stopwords
    if similarity is None and ((word1.is_function_word() and not word2.is_function_word()) or (word2.is_function_word() and not word2.is_function_word())):
        similarity = 0

    # punctuations can only be either identical or totally dissimilar
    if similarity is None and (word1.is_punctuation() or word2.is_punctuation()) and (not canonical_word1 == canonical_word2):
        similarity = 0

    if similarity is not None:
        __word_relatedness_alignment__[word1.form + '__' + word2.form] = (similarity, similarity_type)
        return similarity, similarity_type

    if canonical_word1 == canonical_word2:
        similarity = config.exact
        similarity_type = 'Exact'

    elif stemmer.stem(canonical_word1) == stemmer.stem(canonical_word2):
        similarity = config.stem
        similarity_type = 'Stem'

    elif word1.lemma == word2.lemma:
        similarity = config.stem
        similarity_type = 'Stem'

    elif synonymDictionary.checkSynonymByLemma(word1.lemma, word2.lemma) and 'synonyms' in config.selected_lexical_resources:
        similarity = config.synonym
        similarity_type = 'Synonym'

    elif presentInPPDB(canonical_word1, canonical_word2) and 'paraphrases' in config.selected_lexical_resources:
        similarity = config.paraphrase
        similarity_type = 'Paraphrase'

    elif (word1.index != 0 and word2.index != 0) and cosine_similarity(word1.form, word2.form) > config.related_threshold and 'distributional' in config.selected_lexical_resources:
        double_check = 1

        if (not word1.is_function_word() and not word2.is_function_word()) or word1.pos[0] == word2.pos[0]:
            similarity = config.related
            similarity_type = 'Distributional'
        else:
            similarity = 0.0

    else:
        similarity = 0.0

    if double_check == 0:
        __word_relatedness_alignment__[word1.form + '__' + word2.form] = (similarity, similarity_type)

    return similarity, similarity_type


def get_similarity_type_score(similarity_type, scorer):
    if similarity_type == 'Exact':
        return scorer.exact
    if similarity_type == 'Stem':
        return scorer.stem
    if similarity_type == 'Synonym':
        return scorer.synonym
    if similarity_type == 'Paraphrase':
        return scorer.paraphrase

    return scorer.related


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
    return (word.lower() in cobalt_stopwords) or (word.lower() in punctuations) or (contractionDictionary.is_contraction(word.lower()))


def ispunct(word):
    return word.lower() in punctuations


def function_word_extended(word):
    return (word.lower() in extended_stopwords.stopwords_list) or (word.lower() in punctuations) or (word.lower().isdigit())


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