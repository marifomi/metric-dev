from itertools import islice


class SynonymDictionary(object):

    wordSynsetTable = {}

    def __init__(self, language):
        with open('lex_resources/synonyms/' + language + '.synsets') as f:
            while True:
                synsetBatch = list(islice(f, 2))
                if not synsetBatch:
                    break

                synsetList = []
                for synset in synsetBatch[1].split():
                    synsetList.append(int(synset))

                self.wordSynsetTable[synsetBatch[0].strip()] = synsetList

    # Checks if two lemmas are in one synset
    def checkSynonymByLemma(self, lemma1, lemma2):
        if lemma1 not in self.wordSynsetTable or lemma2 not in self.wordSynsetTable:
            return False

        synsetList1 = self.wordSynsetTable[lemma1]
        synsetList2 = self.wordSynsetTable[lemma2]

        synonym = False

        for synset in synsetList1:
            if synset in synsetList2:
                synonym = True
                break

        return synonym