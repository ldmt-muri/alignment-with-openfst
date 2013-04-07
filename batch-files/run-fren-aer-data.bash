#!/bin/bash

make

# train the latent CRF word aligner on some fr-en data such that the first 447 sentences are the manually aligned test set
mpirun -np $2 ./train-latentCrfAligner example/1k.fr-en none none talign/grammars/wordpair-features example/run.latent-crf-aligner.fr-en.$1 2> example/run.latent-crf-aligner.fr-en.$1.err 

# compute AER
./example/aer/eval-hansards-fren.pl example/run.latent-crf-aligner.fr-en.$1.labels example/aer/hansards.manual-align.wpt03-05/English-French/answers/test.wa.nullalign > example/run.latent-crf-aligner.fr-en.$1.aer

# word-pair features at talign/grammars/wordpair-features
