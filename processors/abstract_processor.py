

class AbstractProcessor(object):

    def __init__(self):
        self.name = str
        self.result_tgt = []
        self.result_ref = []

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_output(self, output):
        self.output = output

    def get_output(self):
        return self.output

    def set_result_tgt(self, result_tgt):
        self.result_tgt = result_tgt

    def set_result_ref(self, result_ref):
        self.result_ref = result_ref

    def get_result_tgt(self):
        return self.result_tgt

    def get_result_ref(self):
        return self.result_ref