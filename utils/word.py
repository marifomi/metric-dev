

class Word(object):

    index = -1

    form = None

    lemma = None

    pos = None

    ner = None

    dep = None

    head = -1

    child = -1 # for collapsed words

    collapsed = False

    def __init__(self, index, form, lemma, pos, dep):
        self.index = index
        self.form = form
        self.lemma = lemma
        self.pos = pos
        self.dep = dep
