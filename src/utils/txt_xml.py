__author__ = 'MarinaFomicheva'

import re
import codecs

def run(src_path, ref_path, tgt_path):

    src = codecs.open(src_path, 'r', 'utf-8')
    ref = codecs.open(ref_path, 'r', 'utf-8')
    tgt = codecs.open(tgt_path, 'r', 'utf-8')

    src_o = codecs.open(src_path + '.xml', 'w', 'utf-8')
    ref_o = codecs.open(ref_path + '.xml', 'w', 'utf-8')
    tgt_o = codecs.open(tgt_path + '.xml', 'w', 'utf-8')

    for my_file in [src_o, ref_o, tgt_o]:
        start_eval(my_file)

    start_refset(ref_o, ref_id='ref')
    start_doc(ref_o, doc_id='doc')
    for i, line in enumerate(ref.readlines()):
        if len(line) == 0:
            continue
        ref_o.write(phrase(line, i))
    end_doc(ref_o)
    end_set(ref_o, 'refset')

    start_srcset(src_o)
    start_doc(src_o, doc_id='doc')
    for i, line in enumerate(src.readlines()):
        src_o.write(phrase(line, i))
    end_doc(src_o)
    end_set(src_o, 'srcset')

    start_tstset(tgt_o, sys_id='system')
    start_doc(tgt_o, doc_id='doc')
    for i, line in enumerate(tgt.readlines()):
        tgt_o.write(phrase(line, i))
    end_doc(tgt_o)
    end_set(tgt_o, 'tstset')

    for my_file in [src_o, ref_o, tgt_o]:
        end_eval(my_file)
        my_file.close()

    for my_file in [src, ref, tgt]:
        my_file.close()

def phrase(line, counter):
    line = re.sub("gt;y", "", line)
    line = re.sub("&", "and", line)
    return "<seg id=\"" + str(counter + 1) + "\">" + line.strip() + "</seg>\n"

def start_refset(o, **kwargs):
    o.write("<refset setid=\"mtc\" srclang=\"Chinese\" trglang=\"English\" refid=\"" + kwargs['ref_id'] + "\">\n")

def start_srcset(o):
        o.write("<srcset setid=\"mtc\" srclang=\"Chinese\">\n")

def start_tstset(o, **kwargs):
    o.write("<tstset setid=\"mtc\" srclang=\"Chinese\" trglang=\"English\" sysid=\"" + kwargs['sys_id'] + "\">\n")

def end_set(o, my_set):
    o.write("</" + my_set + ">\n")

def start_doc(o, **kwargs):
    o.write("<doc docid=\"" + kwargs['doc_id'] + "\" genre=\"nw\">\n<p>\n")

def end_doc(o):
    o.write("</p>\n</doc>\n")

def start_eval(o):

    line1 = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    line2 = "<!DOCTYPE mteval SYSTEM \"ftp://jaguar.ncsl.nist.gov/mt/resources/mteval-xml-v1.3.dtd\">\n<mteval>\n"
    o.write(line1 + line2)

def end_eval(o):
    o.write("</mteval>\n")







