

class Word(object):

    index = -1

    form = None

    lemma = None

    pos = None

    ner = None

    dep = None

    head = -1

    collapsed = False

    _children = []
    _category = None

    def __init__(self, index, form, lemma, pos, dep):
        self.index = index
        self.form = form
        self.lemma = lemma
        self.pos = pos
        self.dep = dep

    def find_children_nodes(self, sentence_parse):
        if len(self._children) == 0:
            for i, word in enumerate(sentence_parse):
                if word.head == self.index:
                    self._children.append(word.index)

        return self._children

    def set_children_nodes(self, nodes):
        if len(self._children) == 0:
            self._children = nodes

    def get_category(self):
        pos_codes = {'n': 'noun', 'j': 'adjective', 'v': 'verb', 'r': 'adverb'}
        if self._category is None:
            if self.pos[0] in pos_codes.keys():
                self._category = pos_codes[self.pos[0]]
            else:
                self._category = ''
        return self._category

