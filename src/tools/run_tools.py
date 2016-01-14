__author__ = 'MarinaFomicheva'

from json import loads
import inspect
from src.tools import processors
import os
from ConfigParser import ConfigParser

class RunTools(object):

    def __init__(self, config):
        self.config = config
        self.outputs = {}

    def run_tools(self):

        my_processors = loads(self.config.get('Resources', 'processors'))

        for name, my_class in sorted(inspect.getmembers(processors)):

            if 'Abstract' in name:
                continue

            if not inspect.isclass(my_class):
                continue

            instance = my_class()

            if instance.get_name() not in my_processors:
                continue

            instance.run(self.config)

            self.outputs[name] = instance.get_result()

def main():

    cfg = ConfigParser()
    cfg.readfp(open(os.getcwd() + '/config/system.cfg'))

    tools = RunTools(cfg)
    tools.run_tools()

if __name__ == '__main__':
    main()