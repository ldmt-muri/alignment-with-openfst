#!/bin/bash
#PBS -l ncpus=1
#PBS -l pmem=7gb
#PBS -l walltime=10:00:00
#PBS -d .
#PBS -j oe
#PBS -o /cab0/wammar/exp/tgt-rep/align/example/wmt10czen.pbs.log

# tokenize training data
python ../../utils/preprocess-english.py example/wmt10czen-plain.en example/wmt10czen.en
python ../../utils/preprocess-czech.py example/wmt10czen-plain.cz example/wmt10czen.cz

# string-to-int training data
python ../../utils/encode-corpus.py example/wmt10czen.en example/wmt10czen.en.vocab example/wmt10czen.en.int
python ../../utils/encode-corpus.py example/wmt10czen.cz example/wmt10czen.cz.vocab example/wmt10czen.cz.int

# lowercase test data
python ../../utils/lowercase.py example/czen-manual-alignments/corpus.en example/wmt10czen-test.en
python ../../utils/lowercase.py example/czen-manual-alignments/corpus.cz example/wmt10czen-test.cz

# string-to-int test data (note: we reuse the same vocab files. the encode-corpus.py script uses the vocab file when it exists.)
python ../../utils/encode-corpus.py example/wmt10czen-test.en example/wmt10czen.en.vocab example/wmt10czen-test.en.int ready
python ../../utils/encode-corpus.py example/wmt10czen-test.cz example/wmt10czen.cz.vocab example/wmt10czen-test.cz.int ready

# compile
make

# run
#./train-loglinear example/wmt10czen.en.int example/wmt10czen.cz.int example/wmt10czen.en.vocab example/wmt10czen.cz.vocab example/wmt10czen.out
