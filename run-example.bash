python util/encode-corpus.py example/small.eng example/small.eng.vocab example/small.eng.int
python util/encode-corpus.py example/small.kin example/small.kin.vocab example/small.kin.int
make
./train-model1 example/small.eng.int example/small.kin.int example/small.out
