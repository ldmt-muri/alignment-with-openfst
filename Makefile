all:

	#echo # compile ibm model 1
	#clang++ -x c++ LearningInfo.h StringUtils.h FstUtils.h FstUtils.cc IbmModel1.h IbmModel1.cc train-model1.cc -lfst -ldl -O0 -o train-model1

#	echo # compile the log linear model
#	clang++ -x c++ alias_sampler.h AlignmentErrorRate.h IAlignmentSampler.h LearningInfo.h StringUtils.h VocabEncoder.h FstUtils.h FstUtils.cc IbmModel1.h IbmModel1.cc LogLinearParams.h LogLinearParams.cc HmmModel.h LogLinearModel.h HmmModel.cc LogLinearModel.cc train-loglinear.cc -lfst -ldl -O0 -o train-loglinear

#	echo #compile the hmm model
#	clang++ -x c++ alias_sampler.h MultinomialParams.h  MultinomialParams.cc LearningInfo.h StringUtils.h FstUtils.h FstUtils.cc HmmModel.h HmmModel.cc train-hmm.cc -lfst -ldl -O0 -o train-hmm

#	echo #autoencoders
	clang++ -x c++ cdec-utils/fast_sparse_vector.h Samplers.h MultinomialParams.h MultinomialParams.cc LearningInfo.h StringUtils.h FstUtils.h FstUtils.cc VocabEncoder.h LogLinearParams.h LogLinearParams.cc LatentCrfModel.h LatentCrfModel.cc train-latentCrfModel.cc -llbfgs -lfst -ldl -O0 -I/usr/include/x86_64-linux-gnu/c++/4.7/ -g -o train-latentCrfModel

