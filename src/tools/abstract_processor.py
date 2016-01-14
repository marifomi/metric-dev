__author__ = 'MarinaFomicheva'


class AbstractProcessor(object):

    def __init__(self):
        self.name = str

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_result(self, result):
        self.result = result

    def get_result(self):
        return self.result