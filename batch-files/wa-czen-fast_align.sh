#!/bin/bash

# train and align
~/cdec/word-aligner/fast_align -i data/$1.cz-en -d -v -o > batch-files/wa-czen-fast_align-dir/out.$1.labels

# compute aer
./data/czen-manual-alignments/eval-czen.pl \
    ./batch-files/wa-czen-fast_align-dir/out.$1.labels \
    ./data/czen-manual-alignments/czen.wal \
    > batch-files/wa-czen-fast_align-dir/out.$1.aer
