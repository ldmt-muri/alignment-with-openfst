CC=mpiCC
SINGLE=-c
BEFORE=-x c++ -std=c++11
LIBS=-llbfgs -lfst -ldl -lboost_mpi -lboost_serialization -lboost_thread -lboost_system
OPT=-O3
INC=-I/usr/local/packages/gcc/4.7.2/include/c++/4.7.2/
DEBUG=-g -ggdb

all: train-latentCrfModel

#train-latentCrfModel:
#	$(CC) $(BEFORE) anneal/Cpp/erstream.cxx anneal/Cpp/rndlcg.cxx anneal/Cpp/registrar.cxx anneal/Cpp/randgen.cxx anneal/Cpp/r250.cxx anneal/Cpp/random.hpp anneal/Cpp/random.cxx anneal/Cpp/simann.cxx cdec-utils/fdict.cc FstUtils.cc LogLinearParams.cc HmmModel2.cc LatentCrfModel.cc train-latentCrfModel.cc $(LIBS) $(OPT) $(INC) $(DEBUG) -o train-latentCrfModel
#	$(CC) $(BEFORE)        $(LIBS) $(OPT) $(INC) $(DEBUG) -o train-latentCrfModel

train-latentCrfModel: train-latentCrfModel.o
	$(CC) train-latentCrfModel.o HmmModel2.o FstUtils.o LatentCrfModel.o LogLinearParams.o fdict.o simann.o random.o r250.o randgen.o registrar.o rndlcg.o erstream.o  $(LIBS) -o train-latentCrfModel

train-latentCrfModel.o: HmmModel2.o LatentCrfModel.o 
	$(CC) $(BEFORE) $(SINGLE) train-latentCrfModel.cc $(OPT)

LatentCrfModel.o: LogLinearParams.o simann.o random.o r250.o randgen.o registrar.o rndlcg.o erstream.o
	$(CC) $(BEFORE) $(SINGLE) LatentCrfModel.cc $(OPT)

LogLinearParams.o: fdict.o
	$(CC) $(BEFORE) $(SINGLE) LogLinearParams.cc $(OPT)

fdict.o:
	$(CC) $(BEFORE) $(SINGLE) cdec-utils/fdict.cc $(OPT)

simann.o: 
	$(CC) $(BEFORE) $(SINGLE) anneal/Cpp/simann.cxx $(OPT)

random.o:
	$(CC) $(BEFORE) $(SINGLE) anneal/Cpp/random.cxx $(OPT)

r250.o:
	$(CC) $(BEFORE) $(SINGLE) anneal/Cpp/r250.cxx $(OPT)

randgen.o:
	$(CC) $(BEFORE) $(SINGLE) anneal/Cpp/randgen.cxx $(OPT)

registrar.o:
	$(CC) $(BEFORE) $(SINGLE) anneal/Cpp/registrar.cxx $(OPT)

rndlcg.o:
	$(CC) $(BEFORE) $(SINGLE) anneal/Cpp/rndlcg.cxx $(OPT)

erstream.o: 
	$(CC) $(BEFORE) $(SINGLE) anneal/Cpp/erstream.cxx $(OPT)

HmmModel2.o: FstUtils.o
	$(CC) $(BEFORE) $(SINGLE) HmmModel2.cc $(OPT)

FstUtils.o: 
	$(CC) $(BEFORE) $(SINGLE) FstUtils.cc $(OPT)

clean:
	rm -rf train-latentCrfModel FstUtils.o HmmModel2.o erstream.o rndlcg.o registrar.o randgen.o r250.o random.o simann.o fdict.o LogLinearParams.o LatentCrfModel.o train-latentCrfModel.o train-latentCrfModel
