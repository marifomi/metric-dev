import urllib3

es_dir = '/Users/MarinaFomicheva/Dropbox/workspace/questplusplus/lang_resources/spanish/wmt15_baseline'
en_dir = '/Users/MarinaFomicheva/Dropbox/workspace/questplusplus/lang_resources/english/wmt15_baseline'

url = 'http://www.quest.dcs.shef.ac.uk/quest_files/truecase-model.es'
response = urllib3.urlopen(url)
with open(es_dir + '/' + 'truecase-model.es', 'w') as f:
    f.write(response.read())
