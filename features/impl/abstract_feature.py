__author__ = 'MarinaFomicheva'

class AbstractFeature(object):

    def __init__(self):
        self.computable = bool
        self.value = float
        self.description = str
        self.name = str

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def set_description(self, d):
        self.description = d

    def get_description(self):
        return self.description

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name
