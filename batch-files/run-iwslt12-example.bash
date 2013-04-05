#!/bin/bash

#PBS -l ncpus=1
#PBS -l pmem=7gb
#PBS -l walltime=10:00:00
#PBS -d .
#PBS -j oe
#PBS -o /mal2/wammar/exp/tgt-rep/align/example/small.pbs.log

make
mpirun -np $2 ./train-latentCrfPosTagger example/tb3-pos-wsj-1000.eng example/run.latent-crf.$1 example/tb3-pos-wsj-1000.pos

#mpirun -np 1 ./train-hmm2 example/tb3-pos-wsj-1000.eng example/run.hmm.$1 example/tb3-pos-wsj-1000.pos







# USEFUL HISTORY OF COMMANDS/EXPERMINTS I RAN

#head example/medium.out.param.final
#awk '{print $1}' example/medium.out.param.final > example/medium.out.param.final.eng
#awk '{print "$2 $4"}' example/medium.out.param.final > example/medium.out.param.final.kin
#python utils/decode-corpus.py example/medium.eng.vocab example/medium.out.param.final.eng example/medium.out.param.final.eng.text
#./train-loglinear example/small.eng.int example/small.kin.int example/small.out

#./train-latentCrfModel example/iwslt12-pos.eng example/iwslt12.latentCrf.out 
#./train-latentCrfModel example/iwslt12-letters.eng example/iwslt12.latentCrf.out example/iwslt12-letters.eng.gold
#./train-latentCrfModel example/wammar-tiny-letters.eng example/iwslt12.latentCrf.out
#./train-latentCrfModel example/tb3-pos-wsj.eng example/tb3-pos-wsj.latentCrf.out example/tb3-pos-wsj.pos
#valgrind mpirun -np 2 ./train-latentCrfModel example/tb3-pos-wsj-10.eng example/tb3-pos-wsj-10.latentCrf.out

#./train-model1 example/iwslt12.eng.int example/iwslt12.trk.int example/iwslt12.out

#mpirun -np $2 ./train-latentCrfAligner example/small.kin-eng example/run.latent-crf-aligner.$1 
#mpirun -np $2 ./train-latentCrfAligner example/medium.kin-eng example/run.latent-crf-aligner.$1 

#valgrind --tool=memcheck --leak-check=yes mpirun -np 10 ./train-latentCrfModel example/tb3-pos-wsj-10.eng example/tb3-pos-wsj-10.latentCrf.out example/tb3-pos-wsj-10.pos

#make -f Makefile-hmm2
