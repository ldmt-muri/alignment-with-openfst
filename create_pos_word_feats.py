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
args = argParser.parse_args()

brown_file = io.open(args.brown_filename, encoding='utf8', mode='r')
output_file = io.open(args.output_filename, encoding='utf8', mode='w')

digit_regex = re.compile('\d')
hyphen_regex = re.compile('-')

suffix_counts = defaultdict(int)
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
min_suffix_count = 10

brown_file.close()
brown_file = io.open(args.brown_filename, encoding='utf8', mode='r')
for line in brown_file:
  cluster, word, frequency = line.split('\t')
  frequency = int(frequency)
  
  features = {
    u'cluster-{}'.format(cluster):1, 
    u'clus-{}'.format(cluster[0:4]):1}
  
  for suffix_length in range(1,4):
    if len(word) > suffix_length and suffix_counts[word[-suffix_length:]] > min_suffix_count:
      features[u'{}-suff-{}'.format(suffix_length, word[-suffix_length:])]=1

  if digit_regex.search(word):
    features[u'contains-digit'] = 1
  elif hyphen_regex.search(word):
    features[u'contains-hyphen'] = 1
  #if len(word) > 1 and word[0] == word[0].upper() and word[1] != word[1].upper():
  #  features[u'capital-initial-only'] = 1
  #if word == word.upper() and word != word.lower():
  #  features[u'all-caps'] = 1
  if word[0] == word[0].upper() and word[0] != word[0].lower():
    features[u'capital-initial'] = 1
  features[u'lower-{}'.format(word.lower())] = 1

  if len(features) == 0: 
    continue

  output_file.write(u'{} ||| {} |||'.format(word, word))
  for featureId in features.keys():
    output_file.write(u' {}={}'.format( featureId, features[featureId] ) )
  output_file.write(u'\n')

output_file.close()
brown_file.close()
