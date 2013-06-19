#!/bin/bash

# compile
make -f Makefile-latentCrfAligner

# use cdec to generate word pair features for a partiu7clar test set
#rm -rf batch-files/wa-swen-latentCrfAligner-dir/sw-en-talign
#~/cdec/word-aligner/aligner.pl --mkcls=/mal0/tools/mosesdecoder/bin/mkcls example/15k.sw-en
#mv talign batch-files/wa-swen-latentCrfAligner-dir/sw-en-talign
#cd batch-files/wa-swen-latentCrfAligner-dir/sw-en-talign
#make
#cd ../../../

# the latent-CRF word alignment model
mpirun -np $2 ./train-latentCrfAligner \
    example/15k.sw-en \
    none none \
    batch-files/wa-swen-latentCrfAligner-dir/sw-en-talign/grammars/wordpairs.f-e.features \
    batch-files/wa-swen-latentCrfAligner-dir/out.$1 \
    14017 \
    2> batch-files/wa-swen-latentCrfAligner-dir/out.$1.err

# example/wa-czen-latentCrfAligner-1k-lambda example/wa-czen-latentCrfAligner-1k-theta
# batch-files/wa-czen-latentCrfAligner-dir/cz-en-talign/grammars/wordpairs.f-e.features.gz 
