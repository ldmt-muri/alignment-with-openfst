import re
import time
import io
import sys
import argparse
from collections import defaultdict, Counter

import util

entries = defaultdict(dict)


def get_words(sd_filename, building=None):
    if building is None:
        to_return = set()
    else:
        assert isinstance(building, set)
        to_return = building
    with open(sd_filename) as fh:
        for l in fh:
            words = l.strip().split()
            to_return.update(words)
    return to_return

# parse/validate arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-b", "--brown_filename")
argParser.add_argument("-o", "--output_filename")
argParser.add_argument("-e", "--embedding_filename")
argParser.add_argument("-s", "--space-delimited-files", nargs='+')
args = argParser.parse_args()

brown_file = io.open(args.brown_filename, encoding='utf8', mode='r')

embedding_model = util.load_embedding_model(args.embedding_filename)

digit_regex = re.compile('\d')
hyphen_regex = re.compile('-')

word_set = set()
for fname in args.space_delimited_files:
    word_set = get_words(fname, word_set)

suffix_counts = defaultdict(int)
prefix_counts = defaultdict(int)
trigram_counts = Counter()
quadgram_counts = Counter()
counter = 0
brown_paths = defaultdict(str)
brown_freq = defaultdict(int)
for line in brown_file:
    counter += 1
    splits = line.split('\t')
    if len(splits) != 3:
        print 'len(splits) = ', len(splits), ' at line ', counter
        print splits
        assert False
    cluster, word, frequency = splits

    brown_paths[word] = cluster
    brown_freq[word] = frequency
    for suffix_length in range(1, 4):
        if len(word) > suffix_length:
            suffix_counts[word[-suffix_length:]] += 1
    for prefix_length in range(1, 4):
        if len(word) > prefix_length:
            prefix_counts[word[:prefix_length]] += 1
    trigram_counts.update(util.get_char_ngrams(word, 3))
    quadgram_counts.update(util.get_char_ngrams(word, 4))
min_suffix_count = 0
min_prefix_count = 0
min_ngram_count = 0

for word in word_set:
    features = {
        #    u'cluster-{}'.format(cluster):1,
        #    u'clus-{}'.format(cluster[0:4]):1
    }
    features.update(util.get_embedding_feats_dict(word, embedding_model))

    page = ord(word[0]) / 100
    features[u'unicode-page-{}'.format(page)] = 1

    for suffix_length in range(1, 4):
        if len(word) > suffix_length and suffix_counts[word[-suffix_length:]] > min_suffix_count:
            features[u'{}-suff-{}'.format(suffix_length, word[-suffix_length:])] = 1

    for prefix_length in range(1, 4):
        if len(word) > prefix_length and prefix_counts[word[:prefix_length]] > min_prefix_count:
            features[u'{}-pre-{}'.format(prefix_length, word[:prefix_length])] = 1

    trigrams = util.get_char_ngrams(word, 3)
    for w in trigrams:
        if trigram_counts[w] > min_ngram_count:
            features[u'trigram-{}'.format(w)] = 1

    quadgrams = util.get_char_ngrams(word, 4)
    for w in quadgrams:
        if quadgram_counts[w] > min_ngram_count:
            features[u'quadgram-{}'.format(w)] = 1

    # alphanumeric
    if word.isdigit():
        features[u'number'] = 1
    elif word.isalpha():
        features[u'word'] = 1
        if word.lower() != word:
            features[u'lower_' + word.lower()] = 1
    elif word.isalnum():
        features[u'alphanumeric'] = 1
    else:
        features[u'nonalphanumeric'] = 1

    # case
    if word.islower():
        features[u'lower'] = 1
    elif word.isupper():
        features[u'upper'] = 1
    elif word[0].isupper():
        features[u'upper-initial'] = 1

    # word string
    # if brown_freq[word] > 10 or brown_freq[word.lower()] > 10:
    features[word.decode('utf-8').lower().replace(u'=', u'eq')] = 1

    # word shape
    shape = [u'^']
    for c in word:
        if c.isdigit():
            if shape[-1] != u'0': shape.append(u'0')
        elif c.isalpha() and c.islower():
            if shape[-1] != u'a': shape.append(u'a')
        elif c.isalpha() and c.isupper():
            if shape[-1] != u'A': shape.append(u'A')
        else:
            if shape[-1] != u'#': shape.append(u'#')
    features[u''.join(shape)] = 1

    if len(features) == 0:
        continue

    entries[word] = features

with io.open(args.output_filename, encoding='utf8', mode='w') as output_file:
    for (word, features) in entries.iteritems():
        output_file.write(u'{} ||| {} |||'.format(word.decode('utf-8'), word.decode('utf-8')))
        for featureId in features.keys():
            # deal with space
            featureId_norm = '_'.join(featureId.strip().split())
            output_file.write(u' {}={}'.format(featureId_norm, features[featureId]))
        output_file.write(u'\n')

