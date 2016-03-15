__author__ = 'MarinaFomicheva'

from json import loads
import inspect
from src.tools import processors
from src.utils.sentence import Sentence

class RunTools(object):

    def __init__(self, config):
        self.config = config
        self.outputs = {}

    def run_tools(self, sample):

        select_names = loads(self.config.get('Resources', 'processors'))
        select_procs = []
        exist_procs = {}

        for name, my_class in inspect.getmembers(processors):
            exist_procs[name] = my_class

        for proc in select_names:
            name_class = (proc, exist_procs[proc])
            select_procs.append(name_class)

        for name, my_class in select_procs:

            instance = my_class()

            print instance.get_name()
            instance.run(self.config, sample)
            instance.get(self.config, sample)

            self.outputs['tgt', instance.get_name()] = instance.get_result_tgt()
            self.outputs['ref', instance.get_name()] = instance.get_result_ref()

    def assign_data(self, sample):

        self.run_tools(sample)

        sample_length = get_len(self.config.get('Data', 'tgt') + sample)

        sentences_tgt = []
        sentences_ref = []

        for i, sentence in enumerate(range(sample_length)):

            my_sentence_tgt = Sentence()
            my_sentence_ref = Sentence()

            for method in set([x[1] for x in self.outputs.keys()]):
                my_sentence_tgt.add_data(method, self.outputs['tgt', method][i])
                my_sentence_ref.add_data(method, self.outputs['ref', method][i])

            sentences_tgt.append(my_sentence_tgt)
            sentences_ref.append(my_sentence_ref)

        self.outputs = {}
        return [sentences_tgt, sentences_ref]

def get_len(my_file):
    return sum(1 for line in open(my_file))