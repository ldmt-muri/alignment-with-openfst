#!/bin/bash

# compile
make

# use cdec to generate word pair features for a partiu7clar test set
#~/cdec/word-aligner/aligner.pl --mkcls=/mal0/tools/mosesdecoder/bin/mkcls example/10k.cz-en
#mv talign cz-en-talign
#cd cz-en-talign
#make
#cd ../
#mv cz-en-talign/grammars/wordpair-features example/cz-en-wordpair-features

# the latent-CRF word alignment model
mpirun -np $2 ./train-latentCrfAligner     \
    example/1k.cz-en \
    example/run.latent-crf-aligner.cz-en.1k.5.lambda example/run.latent-crf-aligner.cz-en.1k.5.theta \
    example/cz-en-wordpair-features  \
    example/run.latent-crf-aligner.cz-en.$1 \
    2> example/run.latent-crf-aligner.cz-en.$1.err 


# compute aer
./example/czen-manual-alignments/eval-czen.pl ./example/run.latent-crf-aligner.cz-en.$1.labels ./example/czen-manual-alignments/czen.wal  > example/run.latent-crf-aligner.cz-en.$1.aer
