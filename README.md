#disclaimer: 
This is work in progress. If you encounter any problems while compiling or using it, it is likely our mistake not yours. Please contact wammar@cs.cmu.edu with questions, comments, and suggestions.


#description:
This is an implementation of the CRF autoencoder framework for four tasks:
* bitext word alignment
* part-of-speech tagging
* code switching
* dependency parsing

Our NIPS 2014 [paper](http://arxiv.org/pdf/1411.1147v2.pdf) describes the CRF autoencoder framework as well as the bitext word alignment and part-of-speech induction tasks in detail. Details on code-switching can be found in our EMNLP shared task [paper](http://www.aclweb.org/anthology/W14-3909).

#dependencies:
* [cdec](https://github.com/redpony/cdec)
* [boost 1.54](http://www.boost.org/users/history/version_1_54_0.html) 
* [libLBFGS-1.10](https://github.com/downloads/chokkan/liblbfgs/liblbfgs-1.10.tar.gz) 
* [MPI-1.8](http://www.open-mpi.org/software/ompi/v1.8/) 
* [openfst-1.3.2](http://www.openfst.org/twiki/bin/view/FST/FstDownload) 
* [python 2.7](https://www.python.org/download/releases/2.7.3/) 

#how to build
I'm assuming your default compiler is either gcc 4.6.3, clang 3.1-8 (or later "fingers crossed")
* bitext word alignment: make -f Makefile-latentCrfAligner
* part-of-speech tagging: make -f Makefile-latentCrfPosTagger
* code switching: make -f Makefile-latentCrfPosTagger (this is not a typo)
* dependency parsing: make -f Makefile-latentCrfParser (still in the works)

# example invocations: 
## part of speech tagging:
```train-latentCrfPosTagger 
  --output-prefix prefix # just a filename prefix for files generated during training
  --train-data sent-per-line-space-delimited-tokens.txt # example file below
  --feat LABEL_BIGRAM --feat PRECOMPUTED --feat EMISSION 
  --feat BOUNDARY_LABELS --feat PRECOMPUTED_XIM2 --feat PRECOMPUTED_XIM1 
  --feat PRECOMPUTED_XI --feat PRECOMPUTED_XIP1 --feat PRECOMPUTED_XIP2 
  --feat OTHER_ALIGNERS
  --min-relative-diff 0.001
  --optimizer adagrad --minibatch-size 8000
  --max-iter-count 50
  --cache-feats true                                                                                                       
  --wordpair-feats word-level-features```

for a list of all options: execute ``latentCrfAligner --help``

### snippet of the file ``sent-per-line-space-delimited-tokens.txt``
```
Ms. Haag plays Elianti .
Rolls-Royce Motor Cars Inc. said it expects its U.S. sales to remain steady at about 1,200 cars in 1990 .
```

### snippet of the file ``word-level-features``
```
expects  starts-with-e 1 starts-with-ex 1 ends-with-ts 1 ends-with-s 1
plays  starts-with-p 1 starts-with-pl 1 ends-with-ys 1 ends-with-s 1
```

## using multiprocesses:
```mpirun 32 train-latentCrfAligner [options]```

