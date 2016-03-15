__author__ = 'MarinaFomicheva'

import urllib2
import os

es_dir = '/Users/MarinaFomicheva/Dropbox/workspace/questplusplus/lang_resources/spanish/wmt15_baseline'
en_dir = '/Users/MarinaFomicheva/Dropbox/workspace/questplusplus/lang_resources/english/wmt15_baseline'

url = 'http://www.quest.dcs.shef.ac.uk/quest_files/truecase-model.es'
response = urllib2.urlopen(url)
with open(es_dir + '/' + 'truecase-model.es', 'w') as f:
    f.write(response.read())
