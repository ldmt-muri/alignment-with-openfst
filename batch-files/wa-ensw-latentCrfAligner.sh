#!/bin/bash

# compile
make -f Makefile-latentCrfAligner

# use cdec to generate word pair features for a partiu7clar test set
#rm -rf batch-files/wa-ensw-latentCrfAligner-dir/en-sw-talign
#~/cdec/word-aligner/aligner.pl --mkcls=/mal0/tools/mosesdecoder/bin/mkcls example/15k.en-sw
#mv talign batch-files/wa-ensw-latentCrfAligner-dir/en-sw-talign
#cd batch-files/wa-ensw-latentCrfAligner-dir/en-sw-talign
#make
#gzip grammars/wordpairs.f-e.features.gz
#cd ../../../

# the latent-CRF word alignment model
mpirun -np $2 ./train-latentCrfAligner \
    example/15k.en-sw \
    none none \
    batch-files/wa-ensw-latentCrfAligner-dir/en-sw-talign/grammars/wordpairs.f-e.features \
    batch-files/wa-ensw-latentCrfAligner-dir/out.$1 \
    14017 \
    2> batch-files/wa-ensw-latentCrfAligner-dir/out.$1.err

# example/wa-czen-latentCrfAligner-1k-lambda example/wa-czen-latentCrfAligner-1k-theta
