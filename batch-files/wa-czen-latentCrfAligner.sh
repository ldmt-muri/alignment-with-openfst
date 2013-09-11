#!/bin/bash

# compile
make -f Makefile-latentCrfAligner

# use cdec to generate word pair features for a partiu7clar test set
#~/cdec/word-aligner/aligner.pl --mkcls=/mal0/tools/mosesdecoder/bin/mkcls data/10k.cz-en
#mv talign cz-en-talign
#cd cz-en-talign
#make
#cd ../
#mv cz-en-talign/grammars/wordpair-features data/cz-en-wordpair-features

# the latent-CRF word alignment model
mpirun -np $2 ./train-latentCrfAligner \
    data/1k.cz-en \
    none none \
    batch-files/wa-czen-latentCrfAligner-dir/cz-en-talign/grammars/wordpairs.f-e.features \
    batch-files/wa-czen-latentCrfAligner-dir/out.$1 \
    515 \
    2> batch-files/wa-czen-latentCrfAligner-dir/out.$1.err

# example/wa-czen-latentCrfAligner-1k-lambda example/wa-czen-latentCrfAligner-1k-theta
# batch-files/wa-czen-latentCrfAligner-dir/cz-en-talign/grammars/wordpairs.f-e.features.gz 

# compute aer
./data/czen-manual-alignments/eval-czen.pl \
    ./batch-files/wa-czen-latentCrfAligner-dir/out.$1.labels \
    ./data/czen-manual-alignments/czen.wal \
    > batch-files/wa-czen-latentCrfAligner-dir/out.$1.aer
