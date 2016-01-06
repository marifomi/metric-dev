__author__ = 'MarinaFomicheva'

import re
import math

meta_info = open('/Users/MarinaFomicheva/Dropbox/workspace/weka/meta_test_set.newstest2015.cs-en', 'r')
predictions = open('/Users/MarinaFomicheva/Dropbox/workspace/weka/test.wmt15.cs-en.out', 'r')
detailed_predictions = open('/Users/MarinaFomicheva/Dropbox/workspace/weka/test.wmt15.cs-en.detailed.out', 'r')

meta_data = meta_info.readlines()
predictions_data = predictions.readlines()
detailed_predictions_data = detailed_predictions.readlines()

#output = open('/Users/MarinaFomicheva/Dropbox/workspace/weka/cobalt_tuned.cs-en.seg.score', 'w')
output2 = open('/Users/MarinaFomicheva/Dropbox/workspace/weka/cobalt_tuned_detailed.cs-en.seg.score', 'w')

predictions = []
scores = []
weights = [-0.1109,-0.3584,-0.116,0.1338,0.0428,0.0315,-0.0174,-0.0085,1.4248,-3.7486,-2.9053,-3.1312,0.792,1.5971,-1.4248,-1.8497,1.134,-1.6884,5.5126,-0.5998,-4.9015,-2.423,0.054,-0.0627,-1.2734,-0.1619,1.0241,0.2476,-0.2315,0.0904,-0.0844]


for line in detailed_predictions_data:
    attrs_clean = []

    if not re.match('\s+[0-9]+', line):
        continue
    attrs = re.sub(r'^.+\s\((.+)\)$', r'\1', line).strip().split(',')
    for i, attr in enumerate(attrs):
        if i == 5 or i == 7 or i == 9:
            continue
        attrs_clean.append(attr)

    score = 0.0
    for i, attr in enumerate(attrs_clean):
        score += float(attr) * weights[i]
    scores.append(score)

# for i, score in enumerate(scores):
#     prob = 1/(1 + math.exp(-score))
#     phrase = str(i+1)
#     print phrase + '\t' + str(score) + '\t' + str(prob)


# for line in predictions_data:
#
#     if not re.match('\s+[0-9]+', line):
#         continue
#     score = re.sub(r'^.+\s(.+) $', r'\1', line)
#     predictions.append(score.strip())

for i, line in enumerate(meta_data):
    direction = line.split(',')[0]
    system = line.split(',')[1]
    phrase = line.split(',')[2]
    #score1 = predictions[i]
    score2 = scores[i]

    #print >>output, 'cobalt-tuned\t' + direction + '\tnewstest2015\t' + system + '\t' + phrase + '\t' + str(score1)
    print >>output2, 'cobalt-tuned\t' + direction + '\tnewstest2015\t' + system + '\t' + phrase + '\t' + str(score2)

#output.close()
output2.close()



