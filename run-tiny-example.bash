#!/bin/bash
#PBS -l ncpus=1
#PBS -l pmem=7gb
#PBS -l walltime=10:00:00
#PBS -d .
#PBS -j oe
#PBS -o /mal2/wammar/exp/tgt-rep/align/example/small.pbs.log

python utils/encode-corpus.py example/tiny.eng example/tiny.eng.vocab example/tiny.eng.int
python utils/encode-corpus.py example/tiny.kin example/tiny.kin.vocab example/tiny.kin.int
make
./train-loglinear example/tiny.eng.int example/tiny.kin.int example/tiny.eng.vocab example/tiny.kin.vocab example/tiny.out
#./train-hmm example/tiny.eng.int example/tiny.kin.int example/tiny.out
