#!/bin/bash

# compile
make -f Makefile-latentCrfAligner

# use cdec to generate word pair features for a partiu7clar test set
#rm -rf batch-files/wa-aren-latentCrfAligner-dir/ar-en-talign
#~/cdec/word-aligner/aligner.pl --mkcls=/mal0/tools/mosesdecoder/bin/mkcls example/1k.ar-en
#mv talign batch-files/wa-aren-latentCrfAligner-dir/ar-en-talign
#cd batch-files/wa-aren-latentCrfAligner-dir/ar-en-talign
#make
#cd grammars
#gzip -d wordpairs.f-e.features.gz
#cd ../../../

# the latent-CRF word alignment model
mpirun -np $2 ./train-latentCrfAligner \
    example/13k.ar-en \
    batch-files/wa-aren-latentCrfAligner-dir/out.13k.10.lambda batch-files/wa-aren-latentCrfAligner-dir/out.13k.10.theta \
    batch-files/wa-aren-latentCrfAligner-dir/ar-en-talign/grammars/wordpairs.f-e.features \
    batch-files/wa-aren-latentCrfAligner-dir/out.$1 \
    13263 \
    2> batch-files/wa-aren-latentCrfAligner-dir/out.$1.err

#   size of the WER test set: 13263 \
#    batch-files/wa-aren-latentCrfAligner-dir/out.13k.10.lambda batch-files/wa-aren-latentCrfAligner-dir/out.13k.10.theta \

# compute aer
./example/aren-manual-alignments/eval-aren.pl \
    ./batch-files/wa-aren-latentCrfAligner-dir/out.$1.labels \
    ./example/aren-manual-alignments/ldc2006e93.ar-en.wal \
    > batch-files/wa-deen-latentCrfAligner-dir/out.$1.aer
