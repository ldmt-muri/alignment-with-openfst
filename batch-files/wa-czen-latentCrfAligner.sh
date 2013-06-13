#!/bin/bash

# compile
make -f Makefile-latentCrfAligner

# use cdec to generate word pair features for a partiu7clar test set
#~/cdec/word-aligner/aligner.pl --mkcls=/mal0/tools/mosesdecoder/bin/mkcls example/10k.cz-en
#mv talign cz-en-talign
#cd cz-en-talign
#make
#cd ../
#mv cz-en-talign/grammars/wordpair-features example/cz-en-wordpair-features

# the latent-CRF word alignment model
mpirun -np $2 ./train-latentCrfAligner \
    example/10k.cz-en \
    none none \
    batch-files/wa-czen-latentCrfAligner-dir/cz-en-talign/grammars/wordpairs.f-e.features \
    batch-files/wa-czen-latentCrfAligner-dir/out.$1 \
    515 \
    2> batch-files/wa-czen-latentCrfAligner-dir/out.$1.err

# example/wa-czen-latentCrfAligner-1k-lambda example/wa-czen-latentCrfAligner-1k-theta
# batch-files/wa-czen-latentCrfAligner-dir/cz-en-talign/grammars/wordpairs.f-e.features.gz 

# compute aer
./example/czen-manual-alignments/eval-czen.pl \
    ./batch-files/wa-czen-latentCrfAligner-dir/out.$1.labels \
    ./example/czen-manual-alignments/czen.wal \
    > batch-files/wa-czen-latentCrfAligner-dir/out.$1.aer
