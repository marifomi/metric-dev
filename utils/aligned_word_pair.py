

class AlignedWordPair(object):

    similarity = None
    context_difference = None

    def __init__(self, left_word, right_word):
        self.left_word = left_word
        self.right_word = right_word
