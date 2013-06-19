#!/bin/bash

# compile
make -f Makefile-latentCrfAligner

# use cdec to generate word pair features for a partiuclar test set
rm -rf batch-files/wa-enar-latentCrfAligner-dir/en-ar-talign
~/cdec/word-aligner/aligner.pl --mkcls=/mal0/tools/mosesdecoder/bin/mkcls example/13k.en-ar
mv talign batch-files/wa-enar-latentCrfAligner-dir/en-ar-talign
cd batch-files/wa-enar-latentCrfAligner-dir/en-ar-talign
make
cd grammars
gzip -d wordpairs.f-e.features.gz
cd ../../../

# the latent-CRF word alignment model
mpirun -np $2 ./train-latentCrfAligner \
    example/100.en-ar \
    none none \
    batch-files/wa-enar-latentCrfAligner-dir/en-ar-talign/grammars/wordpairs.f-e.features \
    batch-files/wa-enar-latentCrfAligner-dir/out.$1 \
    100 \
    2> batch-files/wa-enar-latentCrfAligner-dir/out.$1.err

#   size of the WER test set: 13263 \

# compute aer
./example/enar-manual-alignments/eval-enar.pl \
    ./batch-files/wa-enar-latentCrfAligner-dir/out.$1.labels \
    ./example/enar-manual-alignments/ldc2006e93.en-ar.wal \
    > batch-files/wa-deen-latentCrfAligner-dir/out.$1.aer
