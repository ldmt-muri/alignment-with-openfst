import re
import time
import io
import sys
import argparse
from collections import defaultdict
import math

# parse/validate arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("-v", "--vectors_filename", help="word embeddings")
argparser.add_argument("-i", "--input_filename", help="conll formatted input for specifying relevant word pairs")
argparser.add_argument("-o", "--output_filename", help="the output precomputed wordpair features file")
args = argparser.parse_args()

# first, load the word embeddings in memory
word_embeddings = {}
with io.open(args.vectors_filename, encoding='utf8') as vectors_file:
  for line in vectors_file:
    splits = line.split()
    word, embeddings = splits[0], splits[1:]
    vector_norm = 0
    for i in xrange(len(embeddings)): 
      embeddings[i] = float(embeddings[i])
      vector_norm += embeddings[i] * embeddings[i]
    for i in xrange(len(embeddings)):
      embeddings[i] /= math.sqrt(vector_norm)
    word_embeddings[word] = embeddings
    
# for each sentence in input file, for each word pair, if not processed earlier extract their features and write to output file
pairs = set()
with io.open(args.input_filename, encoding='utf8') as input_file, io.open(args.output_filename, encoding='utf8', mode='w') as output_file:
  sent = []
  for input_line in input_file:
    if len(input_line.strip()) == 0:
      # process sent
      # for each word pair in this sentnece
      for i in xrange(len(sent)):
        for j in range(i+1, len(sent)):
          # make sure both words have embeddings and that you haven't already processed this pair somewhere else
          if sent[i] not in word_embeddings or sent[j] not in word_embeddings or (sent[i], sent[j],) in pairs or (sent[j], sent[i],) in pairs:
            continue
          # keep track of processed pairs
          pairs.add( (sent[i], sent[j],) )
          # write the features
          output_parts = []
          output_parts.append(u'{0} ||| {1} ||| '.format(sent[i], sent[j]))
          dotproduct = 0
          i_embeddings, j_embeddings = word_embeddings[sent[i]], word_embeddings[sent[j]]
          assert(len(i_embeddings) == len(j_embeddings))
          for dim in xrange(len(i_embeddings)):
            dimproduct = i_embeddings[dim] * j_embeddings[dim]
            output_parts.append(u'dim{0}={1:.3f} '.format(dim, dimproduct))
            dotproduct += dimproduct
          output_parts.append(u'dotproduct={0:.3f}\n'.format(dotproduct))
          output_file.write(u''.join(output_parts))
      sent = []
      break
    sent.append(input_line.split()[1])

