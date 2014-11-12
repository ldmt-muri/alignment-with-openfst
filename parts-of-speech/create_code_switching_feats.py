import re
import time
import io
import sys
import argparse
from collections import defaultdict, Counter

import util

entries = defaultdict(dict)


def load_dict(gzipped_pickle):
    if gzipped_pickle is None or len(gzipped_pickle) == 0:
        return (set(), set())
    import cPickle as pickle, gzip

    with gzip.open(gzipped_pickle) as fh:
        r = fh.read()
        (words, ne) = pickle.loads(r)
        return (words, ne)


def get_fuzzy_dict_features(w, s, dict_name=u'fuzzy', distance=10):
    import jellyfish
    to_return = dict()
    for cand in s:
        g = w.lower().encode('utf-8')
        c = cand.lower().encode('utf-8')
        dist = jellyfish.levenshtein_distance(g, c)
        if dist < distance:
            for i in range(dist):
                to_return[u'wordlist-{}_more_than_{}'.format(dict_name, i)] = 1
            to_return[u'wordlist-{}_equals_'.format(dict_name, dist)] = 1
        return to_return


def get_dict_features(w, s, dict_name=u'words'):
    if w.lower() in s:
        # print 'set size: {}, {} in {}'.format(len(s), w, dict_name)
        return {u'wordlist-{}'.format(dict_name): 1}
    return {}


def get_words(sd_filename, building=None):
    from io import open

    if building is None:
        to_return = set()
    else:
        assert isinstance(building, set)
        to_return = building
    with open(sd_filename, encoding='utf-8') as fh:
        for l in fh:
            words = [x for x in l.strip().split()]
            to_return.update(words)
    return to_return


def get_l1_words(sd_file, label_file, l1='lang1'):
    from io import open

    to_return = set()
    with open(sd_file) as sd_f:
        with open(label_file) as lbl_f:
            for text_line, label_line in zip(sd_f, lbl_f):
                text = [x.lower() for x in text_line.strip().split()]
                label = label_line.strip().split()
                to_return.update([text[idx] for idx in range(len(label)) if label[idx] == l1])
    return to_return

# parse/validate arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-b", "--brown-filename")
argParser.add_argument("-o", "--output-filename")
argParser.add_argument("-e", "--embedding-filename")
argParser.add_argument("-d", "--dict-filename")
argParser.add_argument("-s", "--space-delimited-files", nargs='+')
argParser.add_argument("-t", "--training-texts", nargs='*')
argParser.add_argument("-l", "--training-labels", nargs='*')
argParser.add_argument("-w", "--word-list", nargs="*")
argParser.add_argument("--fuzzy-word-list", nargs="*")
argParser.add_argument("-a", "--arabic-analyzer", action="store_true")
argParser.add_argument("--morph-path", default="/usr0/home/chuchenl/git/cs-shared-task/arabic_morph")
argParser.add_argument("-n", "--normalize", action='store_true')
args = argParser.parse_args()

if args.training_labels is not None and len(args.training_labels) > 0:
    assert len(args.training_texts) == len(args.training_labels)

    l1_words = set()
    for (sd, l) in zip(args.training_texts, args.training_labels):
        l1_words.update(get_l1_words(sd, l, l1='lang1'))
else:
    l1_words = None

word_lists = []

if args.word_list is not None and len(args.word_list) > 0:

    for f in args.word_list:
        with io.open(f, encoding='utf-8') as fh:
            word_list = set()
            for l in fh:
                words_in_wordlist = l.lstrip().strip().split()
                # print 'words: {}'.format(words_in_wordlist[0].encode('utf-8'))
                word_list.update(words_in_wordlist) 
        word_lists.append(word_list)
        print 'len of wordlist {}: {}'.format(f, len(word_list))
else:
    word_lists = None

fuzzy_word_lists = []

if args.fuzzy_word_list is not None and len(args.fuzzy_word_list) > 0:

    for f in args.fuzzy_word_list:
        with io.open(f, encoding='utf-8') as fh:
            fuzzy_word_list = set()
            for l in fh:
                words_in_wordlist = l.lstrip().strip().split()
                # print 'words: {}'.format(words_in_wordlist[0].encode('utf-8'))
                fuzzy_word_list.update(words_in_wordlist)
        print 'len of fuzzy wordlist {}: {}'.format(f, len(fuzzy_word_list))
        fuzzy_word_lists.append(fuzzy_word_list)
else:
    fuzzy_word_lists = None

if args.brown_filename is not None and len(args.brown_filename) > 0:
    brown_file = io.open(args.brown_filename, encoding='utf8', mode='r')
else:
    brown_file = None

embedding_model = util.load_embedding_model(args.embedding_filename)
(words, nes) = load_dict(args.dict_filename)

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
brown_paths = defaultdict(unicode)
brown_freq = defaultdict(int)
if brown_file is not None:
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

if args.arabic_analyzer:
    import json, gzip

    with gzip.open(args.morph_path) as fh:
        l = fh.read()
        l = json.loads(l)
    analyzed = util.analyze_arabic_words(word_set, loaded=l)
else:
    analyzed = None

for word in word_set:
    features = {
        # u'cluster-{}'.format(cluster):1,
        # u'clus-{}'.format(cluster[0:4]):1
    }

    # case
    if word.islower():
        features[u'lower'] = 1
    elif word.isupper():
        features[u'upper'] = 1
    elif word[0].isupper():
        features[u'upper-initial'] = 1

    if embedding_model is not None:
        features.update(util.get_embedding_feats_dict(word, embedding_model, normalize=args.normalize))
    features.update(get_dict_features(word, words, dict_name=u'words'))
    features.update(get_dict_features(word, nes, dict_name=u'nes'))

    if analyzed is not None:
        features[u'{}_analyzable_{}'.format(word, analyzed[word])] = 1

    if l1_words is not None:
        features.update(get_dict_features(word, l1_words, dict_name=u'in_labeled_data'))
    if word_lists is not None:
        fired_wl = []
        for idx, w_list in enumerate(word_lists):
            w_list_feature = get_dict_features(word, w_list, dict_name=u'in_provided_wl_{}'.format(idx))
            if len(w_list_feature) > 0:
                fired_wl.append(idx)
            features.update(w_list_feature)
        if len(fired_wl) > 1:
            # print 'word: {} length: {}'.format(word, len(fired_wl))
            features.update({u'wl_{}_fired'.format(u','.join([unicode(x) for x in fired_wl])): 1})

    if fuzzy_word_lists is not None:
        fired_wl = []
        # print '# of fuzzy: {}'.format(len(fuzzy_word_lists))
        for idx, word_list in enumerate(fuzzy_word_lists):
            word_list_feature = get_fuzzy_dict_features(word, word_list, dict_name=u'in_provided_fuzzy_wl_{}'.format(idx), distance=3)
            if len(word_list_feature) > 0:
                fired_wl.append(idx)
            features.update(word_list_feature)
        # print 'fuzzy: {}'.format(len(fired_wl))
        if len(fired_wl) > 1:
            features.update({u'fuzzy_wl_{}_fired'.format(u','.join([unicode(x) for x in fired_wl])): 1})


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

    # word string
    # if brown_freq[word] > 10 or brown_freq[word.lower()] > 10:
    features[word.lower().replace(u'=', u'eq')] = 1

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
        output_file.write(u'{} ||| {} |||'.format(word, word))
        for featureId in features.keys():
            # deal with space
            featureId_norm = '_'.join(featureId.strip().split())
            output_file.write(u' {}={}'.format(featureId_norm, features[featureId]))
        output_file.write(u'\n')

