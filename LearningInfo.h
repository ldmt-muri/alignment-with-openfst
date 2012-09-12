#ifndef _LEARNING_INFO_H_
#define _LEARNING_INFO_H_

#include <vector>
#include <iostream>
#include <math.h>
#include <assert.h>

using namespace std;

namespace DiscriminativeLexicon {
  enum DiscriminativeLexicon {ALL, COOCC};
}

namespace Regularizer {
  enum Regularizer {NONE, L2};
}

namespace OptUtils {
  enum OptAlgorithm {GRADIENT_DESCENT, STOCHASTIC_GRADIENT_DESCENT};

  inline bool IsStochastic(OptAlgorithm a) {
    if (a == STOCHASTIC_GRADIENT_DESCENT) {
      return true;
    } else {
      return false;
    }
  }

  struct OptMethod {
    OptAlgorithm algorithm;
    float learningRate;
    int miniBatchSize;

    OptMethod() {
      algorithm = STOCHASTIC_GRADIENT_DESCENT;
      learningRate = 0.01;
      miniBatchSize = 1;
    }
  }; 
}

class LearningInfo {
 public:
  LearningInfo() {
    useMaxIterationsCount = false;
    useMinLikelihoodDiff = false;
    iterationsCount = 0;
    minLikelihoodDiff = 1.0;
    maxIterationsCount = 10;
    saveAlignmentFstsOnDisk = false;
    neighborhood = DiscriminativeLexicon::ALL;
  }

  bool IsModelConverged() {
    assert(useMaxIterationsCount || useMinLikelihoodDiff);
    
    // logging
    if(useMaxIterationsCount) {
      cerr << "iterationsCount = " << iterationsCount << ". max = " << maxIterationsCount << endl;
    }
    if(useMinLikelihoodDiff && 
       iterationsCount > 1) {
      cerr << "likelihoodDiff = " << fabs(logLikelihood[iterationsCount-1] - 
					 logLikelihood[iterationsCount-2]) << ". min = " << minLikelihoodDiff << endl;
    }
    
    // check for convergnece conditions
    if(useMaxIterationsCount && 
       maxIterationsCount < iterationsCount) {
      return true;
    } 
    if(useMinLikelihoodDiff && 
       iterationsCount > 1 &&
       minLikelihoodDiff > fabs(logLikelihood[iterationsCount-1] - 
					    logLikelihood[iterationsCount-2])) {
      return true;
    } 
    
    // none of the convergence conditions apply!
    return false;
  }
  
  // criteria 1
  bool useMaxIterationsCount;
  int maxIterationsCount;

  // criteria 2
  bool useMinLikelihoodDiff;
  float minLikelihoodDiff;

  // optimization method
  OptUtils::OptMethod optimizationMethod;

  // discriminative lexicon
  DiscriminativeLexicon::DiscriminativeLexicon neighborhood;

  // save alignment FSTs on disk
  bool saveAlignmentFstsOnDisk;

  // output
  int iterationsCount;
  vector< float > logLikelihood;
};



#endif
