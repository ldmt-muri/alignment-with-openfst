all:

	#echo # compile ibm model 1
	#clang++ -x c++ LearningInfo.h StringUtils.h FstUtils.h FstUtils.cc IbmModel1.h IbmModel1.cc train-model1.cc -lfst -ldl -O0 -o train-model1

#	echo # compile the log linear model
#	clang++ -x c++ alias_sampler.h AlignmentErrorRate.h IAlignmentSampler.h LearningInfo.h StringUtils.h VocabEncoder.h FstUtils.h FstUtils.cc IbmModel1.h IbmModel1.cc LogLinearParams.h LogLinearParams.cc HmmModel.h LogLinearModel.h HmmModel.cc LogLinearModel.cc train-loglinear.cc -lfst -ldl -O0 -o train-loglinear

#	echo #compile the hmm model
#	clang++ -x c++ alias_sampler.h MultinomialParams.h  MultinomialParams.cc LearningInfo.h StringUtils.h FstUtils.h FstUtils.cc HmmModel.h HmmModel.cc train-hmm.cc -lfst -ldl -O0 -o train-hmm

#	echo #autoencoders

#	clang++ -x c++ anneal/Cpp/erstream.hpp anneal/Cpp/erstream.cxx anneal/Cpp/randgen.hpp anneal/Cpp/rndlcg.hpp anneal/Cpp/rndlcg.cxx anneal/Cpp/registrar.hpp anneal/Cpp/registrar.cxx anneal/Cpp/randgen.cxx anneal/Cpp/r250.hpp anneal/Cpp/r250.cxx anneal/Cpp/random.hpp anneal/Cpp/random.cxx anneal/Cpp/simann.hpp anneal/Cpp/simann.cxx cdec-utils/fast_sparse_vector.h Samplers.h StringUtils.h VocabEncoder.h MultinomialParams.h MultinomialParams.cc LearningInfo.h FstUtils.h FstUtils.cc LogLinearParams.h LogLinearParams.cc LatentCrfModel.h LatentCrfModel.cc train-latentCrfModel.cc -llbfgs -lfst -ldl -O3 -I/usr/include/x86_64-linux-gnu/c++/4.7/ -g -ggdb -o train-latentCrfModel

# USING mpi + boost
#	mpiCC -x c++ -std=c++0x anneal/Cpp/erstream.hpp anneal/Cpp/erstream.cxx anneal/Cpp/randgen.hpp anneal/Cpp/rndlcg.hpp anneal/Cpp/rndlcg.cxx anneal/Cpp/registrar.hpp anneal/Cpp/registrar.cxx anneal/Cpp/randgen.cxx anneal/Cpp/r250.hpp anneal/Cpp/r250.cxx anneal/Cpp/random.hpp anneal/Cpp/random.cxx anneal/Cpp/simann.hpp anneal/Cpp/simann.cxx cdec-utils/fast_sparse_vector.h cdec-utils/fdict.h cdec-utils/fdict.cc Samplers.h StringUtils.h VocabEncoder.h MultinomialParams.h LearningInfo.h FstUtils.h FstUtils.cc LogLinearParams.h LogLinearParams.cc HmmModel2.h HmmModel2.cc LatentCrfModel.h LatentCrfModel.cc train-latentCrfModel.cc -llbfgs -lfst -ldl -lboost_mpi -lboost_serialization -lboost_thread -lboost_system -O3 -I/usr/include/x86_64-linux-gnu/c++/4.7/ -g -ggdb -o train-latentCrfModel

	mpiCC -x c++ -std=c++0x anneal/Cpp/erstream.hpp anneal/Cpp/erstream.cxx anneal/Cpp/randgen.hpp anneal/Cpp/rndlcg.hpp anneal/Cpp/rndlcg.cxx anneal/Cpp/registrar.hpp anneal/Cpp/registrar.cxx anneal/Cpp/randgen.cxx anneal/Cpp/r250.hpp anneal/Cpp/r250.cxx anneal/Cpp/random.hpp anneal/Cpp/random.cxx anneal/Cpp/simann.hpp anneal/Cpp/simann.cxx cdec-utils/fast_sparse_vector.h cdec-utils/fdict.h cdec-utils/fdict.cc Samplers.h StringUtils.h VocabEncoder.h MultinomialParams.h LearningInfo.h FstUtils.h FstUtils.cc LogLinearParams.h LogLinearParams.cc HmmModel2.h HmmModel2.cc LatentCrfModel.h LatentCrfModel.cc train-latentCrfModel.cc -llbfgs -lfst -ldl -lboost_mpi -lboost_serialization -lboost_thread -lboost_system -O3 -I/usr/local/packages/gcc/4.7.2/include/c++/4.7.2/ -g -ggdb -o train-latentCrfModel

