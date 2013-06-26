make -f Makefile-model1

./train-model1 example/1k.cz-en example/run.model1Aligner.1k.cz-en

# compute aer
./example/czen-manual-alignments/eval-czen.pl \
    ./example/run.model1Aligner.1k.cz-en.train.align \
    ./example/czen-manual-alignments/czen.wal
