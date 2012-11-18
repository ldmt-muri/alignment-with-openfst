all:
#	gcc -x c++ LearningInfo.h StringUtils.h FstUtils.h FstUtils.cc IbmModel1.h IbmModel1.cc train-model1.cc -lfst -ldl -O0 -o train-model1
	clang++ -x c++ AlignmentErrorRate.h IAlignmentSampler.h LearningInfo.h StringUtils.h VocabEncoder.h FstUtils.h FstUtils.cc IbmModel1.h IbmModel1.cc LogLinearParams.h LogLinearParams.cc HmmModel.h LogLinearModel.h HmmModel.cc LogLinearModel.cc train-loglinear.cc -lfst -ldl -O0 -o train-loglinear
#	clang++ -x c++ LearningInfo.h StringUtils.h FstUtils.h FstUtils.cc HmmModel.h HmmModel.cc train-hmm.cc -lfst -ldl -O0 -o train-hmm
