make -f Makefile-hmmAligner

./train-hmmAligner example/1k.cz-en example/1k.cz-en example/run.hmmAligner.1k.cz-en

# compute aer
./example/czen-manual-alignments/eval-czen.pl \
    ./example/run.hmmAligner.1k.cz-en.test.align \
    ./example/czen-manual-alignments/czen.wal
