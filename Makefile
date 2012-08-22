all:
	gcc -x c++ LearningInfo.h StringUtils.h FstUtils.h FstUtils.cc IbmModel1.h IbmModel1.cc train-model1.cc -lfst -ldl -O0 -o train-model1
