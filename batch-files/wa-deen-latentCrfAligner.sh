#!/bin/bash

# compile
make -f Makefile-latentCrfAligner

# use cdec to generate word pair features for a partiu7clar test set
#rm -rf batch-files/wa-deen-latentCrfAligner-dir/de-en-talign
#~/cdec/word-aligner/aligner.pl --mkcls=/mal0/tools/mosesdecoder/bin/mkcls example/10k.de-en
#mv talign batch-files/wa-deen-latentCrfAligner-dir/de-en-talign
#cd batch-files/wa-deen-latentCrfAligner-dir/de-en-talign
#make
#cd grammars
#gzip -d wordpairs.f-e.features
#cd ../../../

# the latent-CRF word alignment model
mpirun -np $2 ./train-latentCrfAligner \
    example/10k.de-en \
    none none \
    batch-files/wa-deen-latentCrfAligner-dir/de-en-talign/grammars/wordpairs.f-e.features \
    batch-files/wa-deen-latentCrfAligner-dir/out.$1 \
    150 \
    2> batch-files/wa-deen-latentCrfAligner-dir/out.$1.err

# example/wa-czen-latentCrfAligner-1k-lambda example/wa-czen-latentCrfAligner-1k-theta
# batch-files/wa-czen-latentCrfAligner-dir/cz-en-talign/grammars/wordpairs.f-e.features.gz 

# compute aer
./example/aer/eval-deen-dev
./example/deen-manual-alignments/eval-deen-dev.pl \
    ./batch-files/wa-deen-latentCrfAligner-dir/out.$1.labels \
    ./example/deen-manual-alignments/test.ref.standard.A \
    > batch-files/wa-deen-latentCrfAligner-dir/out.$1.aer
