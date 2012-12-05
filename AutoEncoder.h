#ifndef _AUTO_ENCODER_H_
#define _AUTO_ENCODER_H_

#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <set>

#include "StringUtils.h"
#include "FstUtils.h"
#include "LogLinearParams.h"
#include "MultinomialParams.h"

using namespace fst;
using namespace std;

class AutoEncoder {

 public:

  AutoEncoder(const string& textFile, 
	      const string& outputPrefix,
	      const LearningInfo& learningInfo);

  // compute the partition function Z_\lambda(x)
  double ComputeZ_lambda(vector<int>& x);

  // compute p(y, z | x)
  double ComputePYZGivenX(vector<int>& x, vector<int>& y, vector<int>& z);

  // compute p(y | x, z)
  double ComputePYGivenXZ(vector<int>& x, vector<int>& y, vector<int>& z);

 private:
  string textFilename, outputPrefix;
  set<int> xDomain, yDomain;
  LogLinearParams lambda;
  MultinomialParams::ConditionalMultinomialParam theta;
  int START_OF_SENTENCE_Y_VALUE;
};

#endif
