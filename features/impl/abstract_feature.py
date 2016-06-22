

class AbstractFeature(object):

    def __init__(self):
        self.computable = None
        self.value = None
        self.description = None
        self.name = 'abstract_feature'
        self.group = None

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

    def set_group(self, group):
        self.group = group

    def get_group(self):
        return self.group

    def __str__(self):
        return self.name


class AbstractChunkFeature(AbstractFeature):

    chunk_number = 10

    def __init__(self):
        AbstractFeature.__init__(self)
        AbstractFeature.set_name(self, 'abstract_chunk_feature')