import codecs

from utils.parsed_sentences_loader import ParsedSentencesLoader
from utils.word import Word


class StanfordParseLoader(object):

    @staticmethod
    def parsed_sentences(input_path):
        with codecs.open(input_path, 'r', 'utf8') as f:
            text = f.read()

        loader = ParsedSentencesLoader()
        sentences = loader.load(text)
        parsed = []
        for sentence in sentences['sentences']:
            parsed.append(StanfordParseLoader._process_parse_result(sentence))
        return parsed


    @staticmethod
    def _get_words(raw_sentence):
        words = []
        for i, item in enumerate(raw_sentence['words']):
            word = Word(i + 1, item[0])
            word.lemma = item[1]['Lemma']
            word.pos = item[1]['PartOfSpeech']
            word.ner = item[1]['NamedEntityTag']
            words.append(word)
        return words

    @staticmethod
    def _get_dependencies(raw_sentence):
        result = {}
        for item in raw_sentence['dependencies']:
            relation = item[0]
            head_index = item[1][item[1].rindex("-") + 1:]
            child_index = item[2][item[2].rindex("-") + 1:]
            if not head_index.isdigit() or not child_index.isdigit():
                continue
            result[int(child_index)] = (relation, int(head_index))
        return result

    @staticmethod
    def _dependents_for_collapsed(words, dependencies):

        complete = [word for word in words if word.dep != '']
        child_word = None
        head_word = None
        for w in range(len(words)):
            if words[w].dep != '':
                continue
            for i in range(len(complete)):
                if complete[i].index > words[w].index:
                    child_word = complete[i]
                    break
            if child_word is None:
                continue
            for i in range(len(complete)):
                if child_word.head == complete[i].index:
                    head_word = complete[i]
            if head_word is None:
                continue
            for i in range(head_word.index, len(complete)):
                try:
                    if '_' in complete[i].dep and words[w].form in complete[i].dep:
                        words[w].collapsed = True
                        words[w].dep = complete[i].dep
                        words[w].head = head_word.index
                        words[w].set_children_nodes([child_word.index])
                except:
                    break
            if not words[w].collapsed: # head of punctuation marks - previous word
                words[w].head = words[w].index - 1
                words[w].dep = 'punct'

        return words

    @staticmethod
    def _words_with_dependenies(words, dependencies):
        for word in words:
            if word.index in dependencies.keys():
                word.dep = dependencies[word.index][0]
                word.head = dependencies[word.index][1]
        return words

    @staticmethod
    def _process_parse_result(raw_sentence):
        words = StanfordParseLoader._get_words(raw_sentence)
        dependencies = StanfordParseLoader._get_dependencies(raw_sentence)
        words = StanfordParseLoader._words_with_dependenies(words, dependencies)
        words = StanfordParseLoader._dependents_for_collapsed(words, dependencies)
        return words
