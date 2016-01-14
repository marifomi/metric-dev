__author__ = 'MarinaFomicheva'

import urllib2
import os

es_dir = '/Users/MarinaFomicheva/Dropbox/workspace/questplusplus/lang_resources/spanish/wmt15_baseline'
en_dir = '/Users/MarinaFomicheva/Dropbox/workspace/questplusplus/lang_resources/english/wmt15_baseline'

url = 'http://www.quest.dcs.shef.ac.uk/quest_files/lm.europarl-interpolated-nc.es'
response = urllib2.urlopen(url)
with open(es_dir + '/' + 'lm.europarl-interpolated-nc.es', 'w') as f:
    f.write(response.read())
