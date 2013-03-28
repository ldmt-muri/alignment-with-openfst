CC=mpiCC
SINGLE=-c
BEFORE=-x c++ -std=c++11
LIBS=-llbfgs -lfst -ldl -lboost_mpi -lboost_serialization -lboost_thread -lboost_system -lcmph
OPT=-O3
INC=-I/usr/local/packages/gcc/4.7.2/include/c++/4.7.2/
DEBUG=-g -ggdb

all: train-latentCrfPosTagger train-latentCrfAligner


# specific to the word alignment task
train-latentCrfAligner: train-latentCrfAligner.o
	$(CC) train-latentCrfAligner.o HmmModel2.o FstUtils.o LatentCrfModel.o LatentCrfAligner.o LogLinearParams.o fdict.o simann.o random.o r250.o randgen.o registrar.o rndlcg.o erstream.o  $(LIBS) -o train-latentCrfAligner

train-latentCrfAligner.o: HmmModel2.o LatentCrfModel.o LatentCrfAligner.o train-latentCrfAligner.cc ClustersComparer.h StringUtils.h LearningInfo.h
	$(CC) $(BEFORE) $(SINGLE) train-latentCrfAligner.cc $(OPT)

LatentCrfAligner.o: LatentCrfModel.o LatentCrfAligner.h LatentCrfAligner.cc
	$(CC) $(BEFORE) $(SINGLE) LatentCrfAligner.cc $(OPT)


# specifiic to the pos tagging task
train-latentCrfPosTagger: train-latentCrfPosTagger.o
	$(CC) train-latentCrfPosTagger.o HmmModel2.o FstUtils.o LatentCrfModel.o LatentCrfPosTagger.o LogLinearParams.o fdict.o simann.o random.o r250.o randgen.o registrar.o rndlcg.o erstream.o  $(LIBS) -o train-latentCrfPosTagger

train-latentCrfPosTagger.o: HmmModel2.o LatentCrfModel.o LatentCrfPosTagger.o LatentCrfAligner.o train-latentCrfPosTagger.cc ClustersComparer.h StringUtils.h LearningInfo.h
	$(CC) $(BEFORE) $(SINGLE) train-latentCrfPosTagger.cc $(OPT)

LatentCrfPosTagger.o: LatentCrfModel.o LatentCrfPosTagger.h
	$(CC) $(BEFORE) $(SINGLE) LatentCrfPosTagger.cc $(OPT)


# shared code
LatentCrfModel.o: LogLinearParams.o simann.o random.o r250.o randgen.o registrar.o rndlcg.o erstream.o LatentCrfModel.cc LatentCrfModel.h LatentCrfModel-inl.h Samplers.h VocabEncoder.h UnsupervisedSequenceTaggingModel.h LearningInfo.h Functors.h cdec-utils/dict.h cdec-utils/fdict.h cdec-utils/fast_sparse_vector.h
	$(CC) $(BEFORE) $(SINGLE) LatentCrfModel.cc $(OPT)

LogLinearParams.o: fdict.o LogLinearParams.cc LogLinearParams.h LogLinearParams-inl.h
	$(CC) $(BEFORE) $(SINGLE) LogLinearParams.cc $(OPT)

HmmModel2.o: FstUtils.o HmmModel2.cc HmmModel2.h LearningInfo.h
	$(CC) $(BEFORE) $(SINGLE) HmmModel2.cc $(OPT)

FstUtils.o: FstUtils.cc FstUtils.h
	$(CC) $(BEFORE) $(SINGLE) FstUtils.cc $(OPT)

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

clean:
	rm -rf train-latentCrfModel FstUtils.o HmmModel2.o erstream.o rndlcg.o registrar.o randgen.o r250.o random.o simann.o fdict.o LogLinearParams.o LatentCrfModel.o train-latentCrfModel.o train-latentCrfModel
