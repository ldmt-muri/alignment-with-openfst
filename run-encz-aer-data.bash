#!/bin/bash
#PBS -l ncpus=1
#PBS -l pmem=7gb
#PBS -l walltime=10:00:00
#PBS -d .
#PBS -j oe
#PBS -o /cab0/wammar/exp/tgt-rep/align/example/wmt10czen.pbs.log

# string-to-int training data (which is also the test data!)
python utils/trie-encode-corpus.py example/czen-aer-tokenized.en example/czen-aer-tokenized.en.vocab example/czen-aer-tokenized.en.int
python utils/trie-encode-corpus.py example/czen-aer-tokenized.cz example/czen-aer-tokenized.cz.vocab example/czen-aer-tokenized.cz.int

# compile
make

# run loglinear model
#./train-loglinear \
#    example/czen-aer-tokenized.cz.int \
#    example/czen-aer-tokenized.en.int \
#    example/czen-aer-tokenized.cz.int \
#    example/czen-aer-tokenized.en.int \
#    example/czen-aer-tokenized.cz.vocab \
#    example/czen-aer-tokenized.en.vocab \
#    example/encz-aer-tokenized.loglinear.out 

# evaluate loglinear model
#./example/czen-manual-alignments/eval-czen.pl ./example/encz-aer-tokenized.loglinear.out.align ./example/czen-manual-alignments/czen.wal  > example/encz-aer-tokenized.loglinear.out.align.aer

# run hmm model
./train-hmm \
    example/czen-aer-tokenized.en.int \
    example/czen-aer-tokenized.cz.int \
    example/czen-aer-tokenized.en.int \
    example/czen-aer-tokenized.cz.int \
    example/encz-aer-tokenized.hmm.out

# evaluate hmm model
./example/czen-manual-alignments/eval-czen.pl ./example/encz-aer-tokenized.hmm.out.align ./example/czen-manual-alignments/czen.wal  > example/encz-aer-tokenized.hmm.out.align.aer
