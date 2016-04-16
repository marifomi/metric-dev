__author__ = 'MarinaFomicheva'

from json import loads
import inspect
from src.processors import processors
from src.utils.sentence import Sentence


class RunProcessors(object):

    def __init__(self, config):
        self.config = config
        self.outputs = {}

    def run_processors(self):

        results_target = []
        results_reference = []

        sentences_target = []
        sentences_reference = []

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

            print('Running ' + instance.get_name())
            instance.run(self.config)

            print('Getting ' + instance.get_name())
            instance.get(self.config)

            print(instance.get_name() + ' ' + 'finished!')

            results_target.append(instance.get_result_tgt())
            results_reference.append(instance.get_result_ref())

        for i in range(len(results_target[0])):

            my_sentence_tgt = Sentence()
            my_sentence_ref = Sentence()

            for k, (name, my_class) in enumerate(select_procs):
                instance = my_class()

                if instance.get_output() is not None:
                    my_sentence_tgt.add_data(instance.get_name(), results_target[k][i])
                    my_sentence_ref.add_data(instance.get_name(), results_reference[k][i])

            sentences_target.append(my_sentence_tgt)
            sentences_reference.append(my_sentence_ref)

        return [sentences_target, sentences_reference]


def get_len(my_file):
    return sum(1 for line in open(my_file))