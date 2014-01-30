import re
import time
import io
import sys
import argparse
from collections import defaultdict

# parse/validate arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--input_tag_dict_filename") 
argParser.add_argument("-o", "--output_tag_dict_filename") 
argParser.add_argument("-t", "--text_filename")
args = argParser.parse_args()

# first, create a map from lowercase to uppercase in this text
lower_to_upper = defaultdict(set)
with io.open(args.text_filename, encoding='utf8', mode='r') as text_file:
  for line in text_file:
    for token in line.split(' '):
      if token.lower() != token:
        lower_to_upper[token.lower()].add(token)

# now, for each lowercased word type in the tag dictionary, create additional cased wordtypes to cover all occurences in the text
input_tag_dict_file = io.open(args.input_tag_dict_filename, encoding='utf8', mode='r')
with io.open(args.output_tag_dict_filename, encoding='utf8', mode='w') as output_tag_dict_file:
  for line in input_tag_dict_file:
    output_tag_dict_file.write(line)
    for cased_type in lower_to_upper[line.split(' ')[0]]:
      output_tag_dict_file.write(u'{} {}'.format(cased_type, ' '.join(line.split(' ')[1:])))
