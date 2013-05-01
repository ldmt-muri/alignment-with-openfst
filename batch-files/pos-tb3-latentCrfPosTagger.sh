#!/bin/bash

make
mpirun -np $2 ./train-latentCrfPosTagger example/tb3-pos-wsj-1000.eng example/run.latent-crf-pos-tagger.$1 example/tb3-pos-wsj-1000.pos

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

#valgrind --tool=memcheck --leak-check=yes mpirun -np 10 ./train-latentCrfModel example/tb3-pos-wsj-10.eng example/tb3-pos-wsj-10.latentCrf.out example/tb3-pos-wsj-10.pos

#make -f Makefile-hmm2
