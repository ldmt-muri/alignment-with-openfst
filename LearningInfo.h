#ifndef _LEARNING_INFO_H_
#define _LEARNING_INFO_H_

#include <vector>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <map>

#include "IAlignmentSampler.h"
#include "VocabEncoder.h"

using namespace std;

namespace Distribution {
  enum Distribution {
    // the log linear distribution with all features
    TRUE, 
    // the log linear distribution, using the subset of the features which can be computed
    // as a function of the current the current target word, alignment variable, src sentence, 
    // and tgt lengths (but not previous target words or alignments).
    LOCAL, 
    // any distribution that implements the interface IAlignmentSampler
    CUSTOM};
}

namespace DiscriminativeLexicon {
  enum DiscriminativeLexicon {ALL, COOCC, IBM1FWD_BCK};
}

namespace Regularizer {
  enum Regularizer {
    NONE, 
    L2, 
    // the cumulative L1 approximation of Tsuruoka et al. 2009
    L1};
}

namespace OptUtils {
  enum OptAlgorithm {GRADIENT_DESCENT, STOCHASTIC_GRADIENT_DESCENT, BLOCK_COORD_GRADIENT_DESCENT, LBFGS};

  inline bool IsStochastic(OptAlgorithm a) {
    if (a == STOCHASTIC_GRADIENT_DESCENT || a == LBFGS) {
      return true;
    } else {
      return false;
    }
  }

  // documentation can be found at http://www.chokkan.org/software/liblbfgs/structlbfgs__parameter__t.html
  struct LbfgsParams {
    int max_iterations;

    LbfgsParams() {
      max_iterations = 10;
    }
  };

  struct OptMethod {
    OptAlgorithm algorithm, secondaryAlgorithm;
    float learningRate;
    int miniBatchSize;
    Regularizer::Regularizer regularizer;
    float regularizationStrength;
    LbfgsParams lbfgsParams;
    int lbfgsMemoryBuffer;

    OptMethod() {
      algorithm = STOCHASTIC_GRADIENT_DESCENT;
      secondaryAlgorithm = LBFGS;
      learningRate = 0.01;
      miniBatchSize = 1;
      regularizer = Regularizer::NONE;
      regularizationStrength = 1000;
      lbfgsMemoryBuffer = 500;
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
    samplesCount = 1000;
    distATGivenS = Distribution::TRUE;
    customDistribution = 0;
    neighborhoodMinIbm1FwdScore = 0.001;
    neighborhoodMinIbm1BckScore = 0.001;
    neighborhoodMinCoocc = 3;
  }

  bool IsModelConverged() {
    assert(useMaxIterationsCount || useMinLikelihoodDiff || useEarlyStopping);
    
    // logging
    if(useMaxIterationsCount) {
      cerr << "iterationsCount = " << iterationsCount << ". max = " << maxIterationsCount << endl << endl;
    }
    if(useMinLikelihoodDiff && 
       iterationsCount > 1) {
      cerr << "likelihoodDiff = " << fabs(logLikelihood[iterationsCount-1] - 
					  logLikelihood[iterationsCount-2]) << ". min = " << minLikelihoodDiff << endl << endl;
    }
    if(useEarlyStopping &&
       iterationsCount > 1) {
      cerr << "validationLikelihood[" << iterationsCount-2 << "] = " << validationLogLikelihood[iterationsCount-2] << endl;
      cerr << "validationLikelihood[" << iterationsCount-1 << "] = " << validationLogLikelihood[iterationsCount-1] << endl;
      cerr << "convergence criterion: stop training when loglikelihood no longer decreases, after the second iteration" << endl << endl;
    }
    
    // check for convergnece conditions
    if(useMaxIterationsCount && 
       maxIterationsCount <= iterationsCount) {
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
  float neighborhoodMinIbm1FwdScore;
  float neighborhoodMinIbm1BckScore;
  float neighborhoodMinCoocc;

  // save alignment FSTs on disk
  bool saveAlignmentFstsOnDisk;

  // number of samples used to approximate the posterior expectations
  int samplesCount;

  // output
  int iterationsCount;
  vector< float > logLikelihood;
  vector< float > validationLogLikelihood;  

  // distribution used to model p(a,T|S)
  Distribution::Distribution distATGivenS;
  IAlignmentSampler *customDistribution;

  // when using a proposal distribution for p(a,T|S), would you like to union the alignments of p(a|T,S) as well?
  bool unionAllCompatibleAlignments;

  // map src type IDs to strings
  VocabDecoder *srcVocabDecoder;

  // map tgt type IDs to strings
  VocabDecoder *tgtVocabDecoder;

  // ibm 1 forward log probs
  // [srcToken][tgtToken]
  std::map<int, std::map<int, float> > *ibm1ForwardLogProbs;

  // ibm 1 backward log probs
  // [tgtToken][srcToken]
  std::map<int, std::map<int, float> > *ibm1BackwardLogProbs;
};



#endif
