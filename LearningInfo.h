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
    useEarlyStopping = false;
    trainToDevDataSize = 10;
    iterationsCount = 0;
    minLikelihoodDiff = 1.0;
    maxIterationsCount = 10;
    saveAlignmentFstsOnDisk = false;
    neighborhood = DiscriminativeLexicon::ALL;
  }

  bool IsModelConverged() {
    assert(useMaxIterationsCount || useMinLikelihoodDiff || useEarlyStopping);
    
    // logging
    if(useMaxIterationsCount) {
      cerr << "iterationsCount = " << iterationsCount << ". max = " << maxIterationsCount << endl;
    }
    if(useMinLikelihoodDiff && 
       iterationsCount > 1) {
      cerr << "likelihoodDiff = " << fabs(logLikelihood[iterationsCount-1] - 
					  logLikelihood[iterationsCount-2]) << ". min = " << minLikelihoodDiff << endl;
    }
    if(useEarlyStopping &&
       iterationsCount > 1) {
      cerr << "validationLikelihood[" << iterationsCount-2 << "] = " << validationLogLikelihood[iterationsCount-2] << endl;
      cerr << "validationLikelihood[" << iterationsCount-1 << "] = " << validationLogLikelihood[iterationsCount-1] << endl;
      cerr << "convergence criterion: stop training when loglikelihood no longer decreases, after the second iteration" << endl;
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
    if(useEarlyStopping &&
       iterationsCount > 5 &&
       validationLogLikelihood[iterationsCount-1] - validationLogLikelihood[iterationsCount-2] > 0) {
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

  // criteria 3
  // by early stopping, i mean assume convergence as soon as the likelihood of a validation set cease to increase
  bool useEarlyStopping;
  int trainToDevDataSize;

  // optimization method
  OptUtils::OptMethod optimizationMethod;

  // discriminative lexicon
  DiscriminativeLexicon::DiscriminativeLexicon neighborhood;

  // save alignment FSTs on disk
  bool saveAlignmentFstsOnDisk;

  // output
  int iterationsCount;
  vector< float > logLikelihood;
  vector< float > validationLogLikelihood;
};



#endif
