# -*-coding: utf-8 -*-
import re
import time
import io
import sys
import argparse
from collections import defaultdict

# parse/validate arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-b", "--brown_filename")
argParser.add_argument("-o", "--output_filename")
argParser.add_argument("-hk", "--haghighi_klein", action='store_true', help='when set, only use features used by Haghighi and Klein 2006, and later by Berg-Kirkpatrick et al 2010. By default, use the "full" set of features, as detailed in Ammar et al. 2014')
argParser.add_argument("-lang", "--lang", default='')
args = argParser.parse_args()

brown_file = io.open(args.brown_filename, encoding='utf8', mode='r')
output_file = io.open(args.output_filename, encoding='utf8', mode='w')

digit_regex = re.compile('\d')
hyphen_regex = re.compile('-')

suffix_counts = defaultdict(int)
prefix_counts = defaultdict(int)
counter = 0
for line in brown_file:
  counter += 1
  splits = line.split('\t')
  if len(splits) != 3:
    print 'len(splits) = ', len(splits), ' at line ', counter
    print splits
    assert False
  cluster, word, frequency = splits
  for suffix_length in range(1,4):
    if len(word) > suffix_length:
      suffix_counts[word[-suffix_length:]] += 1
      prefix_counts[word[0:suffix_length]] += 1

## This is the value "f" in Table 2 of Ammar et al. 2014
min_affix_count = counter / 1000.0

brown_file.close()
brown_file = io.open(args.brown_filename, encoding='utf8', mode='r')
for line in brown_file:
  cluster, word, frequency = line.split('\t')
  frequency = int(frequency)
  
  features = {}

  # hypthen
  if hyphen_regex.search(word):
    features[u'contain_hyphen'] = 1

  # digit
  if digit_regex.search(word):
    features[u'contain_digit'] = 1

  # affixes
  for affix_length in range(1,4):
    if len(word) > affix_length and suffix_counts[word[-affix_length:]] > min_affix_count and prefix_counts[word[0:affix_length]] > min_affix_count:
      features[u'{}-pref-{}-suff-{}'.format(affix_length, word[-affix_length:], word[0:affix_length])]=1
    if len(word) > affix_length and suffix_counts[word[-affix_length:]] > min_affix_count:
      features[u'{}-suff-{}'.format(affix_length, word[-affix_length:])]=1
    if len(word) > affix_length and prefix_counts[word[0:affix_length]] > min_affix_count:
      features[u'{}-pref-{}'.format(affix_length, word[0:affix_length])]=1

  # only fire an emission for frequent words
  min_word_frequency=100
  if frequency >= min_word_frequency:
    features[word.lower().replace(u'=', u'eq')] = 1

  # word shape
  shape=[u'^']
  for c in word:
    if c.isdigit():
      if shape[-1] != u'0': shape.append(u'0')
    elif c.isalpha() and c.isupper():
      if shape[-1] != u'A': shape.append(u'A')
    elif c.isalpha():
      if shape[-1] != u'a': shape.append(u'a')
    else: 
      if shape[-1] != u'#': shape.append(u'#')
  features[u''.join(shape)] = 1

  if len(features) == 0: 
    continue

  output_file.write(u'{} ||| {} |||'.format(word, word))
  for featureId in features.keys():
    output_file.write(u' {}={}'.format( featureId, features[featureId] ) )
  output_file.write(u'\n')

output_file.close()
brown_file.close()
