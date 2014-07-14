'''
utility functions
'''

__author__ = 'as1986'


def get_char_ngrams(w, n):
    '''
    gets char n-grams from word w
    '''
    assert isinstance(w, basestring)
    assert isinstance(n, int)
    if len(w) <= n:
        return [w]
    to_return = []
    for i in range(n, len(w) - n):
        to_return.append(w[i:i + n])
    return to_return


def into_single_line(original):
    '''
    string list -> single line
    '''
    n = original.strip().split('\n')
    to_join = [x.strip() for x in n]
    newline_removed = ' '.join(to_join).encode('utf-8')
    return newline_removed


def tokenize(str_list):
    '''
    calls external tokenizer
    '''
    import codecs

    w_fh = codecs.open('temp_in', 'w', 'utf-8')
    w_fh.writelines('\n'.join(str_list).decode('utf-8'))
    w_fh.close()

    import os

    os.system('tokenizer.pl {} {}'.format('temp_in', 'temp_out'))
    with codecs.open('temp_out', 'r', 'utf-8') as fh:
        to_return = [x for x in fh]

    return to_return


def leave_n_uid_out_split(list_tweets, n):
    assert isinstance(list_tweets, list)
    from itertools import permutations

    all_uid = {t['user_id'] for t in list_tweets}

    for u in permutations(all_uid, r=n):
        test_list = []
        training_list = []
        for t in list_tweets:
            if t['user_id'] in u:
                test_list.append(t)
            else:
                training_list.append(t)
        yield (test_list, training_list)


def leave_one_uid_out_split(list_tweets):
    return leave_n_uid_out_split(list_tweets, n=1)


def load_tweets_from_json_gz(fname, verbose=False):
    assert isinstance(fname, basestring)

    import json, gzip

    with gzip.open(fname) as fh:
        l = fh.read()
        to_return = json.loads(l)
        if verbose:
            print to_return
    return to_return


def gen_xy_pair(word):
    assert isinstance(word, dict)
    assert 'label' in word
    y = word['label']
    x = dict(word)
    x.pop('label')

    # FIXME encoding 'hots' ourselves
    for idx in range(len(x['hots'])):
        x['hots_' + str(idx)] = x['hots'][idx]
    x.pop('hots')
    return (y, x)


def gen_feat_label(tweets):
    feats = []
    labels = []
    for t in tweets:
        for w in t['words']:
            l, f = gen_xy_pair(w)
            labels.append(l)
            feats.append(f)
    return (labels, feats)


def dict_vectorize(dict_list):
    assert isinstance(dict_list, list)
    from sklearn.feature_extraction import DictVectorizer

    vec = DictVectorizer()
    vec.fit(dict_list)
    return vec


def load_embedding_model(fname):
    if fname is None or len(fname) == 0:
        return None
    from gensim import models

    m = models.Word2Vec.load(fname)
    return m


def transform_and_append(vec, X, model):
    words = []
    for w in X:
        if 'word' in w:
            words.append(w['word'])
        else:
            words.append(None)
    transformed = vec.transform(X)
    if model is None:
        return transformed
    return append_embedding(transformed, model, words)


def append_embedding(X, model, words):
    import numpy as np
    from gensim import models
    from scipy import sparse as sp
    import logging

    assert isinstance(model, models.Word2Vec)
    dim = model.layer1_size
    to_append = np.zeros(shape=(X.shape[0], dim))
    for idx, w in enumerate(words):
        if w in model:
            to_append[idx] = model[w]
    print 'to_append: {}'.format(to_append.shape)
    print 'X: {}'.format(X.shape)
    concat = sp.hstack([X, sp.csr_matrix(to_append)])
    logging.info('works here')
    return concat


def get_embedding_feats_dict(word, model):
    import logging
    from gensim import models
    import numpy as np

    assert isinstance(model, models.Word2Vec)
    to_return = dict()
    dim = model.layer1_size
    logging.info(u'model dim: {}'.format(dim))
    if word in model:
        vec = model[word]
        for i in range(dim):
            to_return[u'embedding_dim_{}'.format(i)] = vec[i]
    else:
        to_return[u'no_embedding_{}'.format(word)] = 1

    return to_return


def label_vectorize(labels):
    assert isinstance(labels, list)
    s = set(labels)
    return {k: idx for idx, k in enumerate(s)}
