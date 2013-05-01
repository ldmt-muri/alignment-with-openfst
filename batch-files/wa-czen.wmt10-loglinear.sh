#!/bin/bash
#PBS -l ncpus=1
#PBS -l pmem=7gb
#PBS -l walltime=10:00:00
#PBS -d .
#PBS -j oe
#PBS -o /cab0/wammar/exp/tgt-rep/align/example/wmt10czen.pbs.log

# tokenize training data
python utils/preprocess-english.py example/wmt10czen-plain-10k.en example/wmt10czen-10k.en.auto
python utils/preprocess-czech.py example/wmt10czen-plain-10k.cz example/wmt10czen-10k.cz.auto

# string-to-int training data
python utils/trie-encode-corpus.py example/wmt10czen-10k.en.auto example/wmt10czen-10k.en.vocab example/wmt10czen-10k.en.int
python utils/trie-encode-corpus.py example/wmt10czen-10k.cz.auto example/wmt10czen-10k.cz.vocab example/wmt10czen-10k.cz.int

# lowercase test data
python utils/lowercase.py example/czen-manual-alignments/corpus.en example/wmt10czen-test.en.auto
python utils/lowercase.py example/czen-manual-alignments/corpus.cz example/wmt10czen-test.cz.auto

# string-to-int test data (note: we reuse the same vocab files. the encode-corpus.py script uses the vocab file when it exists.)
python utils/trie-encode-corpus.py example/wmt10czen-test.en.auto example/wmt10czen-10k.en.vocab example/wmt10czen-test.en.int ready
python utils/trie-encode-corpus.py example/wmt10czen-test.cz.auto example/wmt10czen-10k.cz.vocab example/wmt10czen-test.cz.int ready

# compile
make

# run
./train-loglinear \
    example/wmt10czen-10k.en.int \
    example/wmt10czen-10k.cz.int \
    example/wmt10czen-test.en.int \
    example/wmt10czen-test.cz.int \
    example/wmt10czen-10k.en.vocab \
    example/wmt10czen-10k.cz.vocab \
    example/wmt10czen-10k.out 
