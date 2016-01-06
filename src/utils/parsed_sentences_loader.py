#!/usr/bin/env python
#
#  This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import re

WORD_PATTERN = re.compile('\[([^\]]+)\]')

CR_PATTERN = re.compile(r"\((\d*),(\d)*,\[(\d*),(\d*)\]\) -> \((\d*),(\d)*,\[(\d*),(\d*)\]\), that is: \"(.*)\" -> \"(.*)\"")

STATE_START, STATE_TEXT, STATE_WORDS, STATE_TREE, STATE_DEPENDENCY, STATE_COREFERENCE = 0, 1, 2, 3, 4, 5


class ParsedSentencesLoader(object):

    def parse_bracketed(self, s):
        '''Parse word features [abc=... def = ...]
        Also manages to parse out features that have XML within them
        '''
        word = None
        attrs = {}
        temp = {}
        # Substitute XML tags, to replace them later
        for i, tag in enumerate(re.findall(r"(<[^<>]+>.*<\/[^<>]+>)", s)):
            if '</s>' in tag or '</S>' in tag:
                continue
            temp["^^^%d^^^" % i] = tag
            s = s.replace(tag, "^^^%d^^^" % i)
        # Load key-value pairs, substituting as necessary
        for attr, val in re.findall(r"([^=\s]*)=([^=\s]*)", s):
            if val in temp:
                val = temp[val]
            if attr == 'Text':
                word = val
            else:
                attrs[attr] = val
        return (word, attrs)


    def __parse_dependency_entry(self, line):
        split_entry = re.split("\(|, ", line[:-1])
        if len(split_entry) == 3:
            rel, left, right = split_entry
            return tuple([rel,left,right])
        else:
            return None

    def parse_parser_results(self, text):
        results = {"sentences": []}
        state = STATE_START
        for line in text.split("\n"):
            line = line.strip()
            if len(line) == 0:
                continue

            if line.startswith("Sentence #"):
                sentence = {'words': [], 'parsetree': [], 'dependencies': []}
                results["sentences"].append(sentence)
                state = STATE_TEXT

            elif state == STATE_TEXT:
                sentence['text'] = line
                state = STATE_WORDS

            elif state == STATE_WORDS:
                for s in WORD_PATTERN.findall(line):
                    sentence['words'].append(self.parse_bracketed(s))
                state = STATE_TREE

            elif state == STATE_TREE:
                if line.startswith('root'):
                    state = STATE_DEPENDENCY
                    sentence['parsetree'] = " ".join(sentence['parsetree'])
                    sentence['dependencies'].append(self.__parse_dependency_entry(line))
                else:
                    sentence['parsetree'].append(line)

            elif state == STATE_DEPENDENCY:
                entry = self.__parse_dependency_entry(line)
                if entry == None:
                    state = STATE_COREFERENCE
                else:
                    sentence['dependencies'].append(entry)

            elif state == STATE_COREFERENCE:
                if "Coreference set" in line:
                    if 'coref' not in results:
                        results['coref'] = []
                    coref_set = []
                    results['coref'].append(coref_set)
                else:
                    for src_i, src_pos, src_l, src_r, sink_i, sink_pos, sink_l, sink_r, src_word, sink_word in CR_PATTERN.findall(line):
                        src_i, src_pos, src_l, src_r = int(src_i)-1, int(src_pos)-1, int(src_l)-1, int(src_r)-1
                        sink_i, sink_pos, sink_l, sink_r = int(sink_i)-1, int(sink_pos)-1, int(sink_l)-1, int(sink_r)-1
                        coref_set.append(((src_word, src_i, src_pos, src_l, src_r), (sink_word, sink_i, sink_pos, sink_l, sink_r)))

        return results

    def load(self, sentence):
        return self.parse_parser_results(sentence)