#!/bin/bash

# encode training data
python utils/encode-corpus.py \
    example/small.eng \
    example/small.eng.vocab \
    example/small.eng.int
python utils/encode-corpus.py \
    example/small.kin \
    example/small.kin.vocab \
    example/small.kin.int

# compile 
make

# train a loglinear model and align a test set
./train-loglinear \
    example/small.eng.int \
    example/small.kin.int \
    example/small.eng.int \
    example/small.kin.int \
    example/small.eng.vocab \
    example/small.kin.vocab \
    example/small.out

#head example/small.out.param.final
