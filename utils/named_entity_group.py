class NamedEntityGroup(object):

    def __init__(self, indicies, words, ner):
        self.indicies = indicies
        self.words = words
        self.ner = ner
        self.forms = []
        for word in words:
            self.forms.append(word.form)

    @staticmethod
    def sort_key(group):
        if hasattr(group, 'indicies'):
            return len(group.indicies)
        else:
            return 0