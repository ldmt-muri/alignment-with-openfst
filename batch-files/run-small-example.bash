#!/bin/bash
#PBS -l ncpus=1
#PBS -l pmem=7gb
#PBS -l walltime=10:00:00
#PBS -d .
#PBS -j oe
#PBS -o /cab0/wammar/exp/tgt-rep/align/example/small.pbs.log

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

#./train-model1 example/small.eng.int example/small.kin.int example/small.out
#./train-hmm example/small.eng.int example/small.kin.int example/small.out
#head example/small.out.param.final
