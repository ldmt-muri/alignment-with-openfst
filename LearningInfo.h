#ifndef _LEARNING_INFO_H_
#define _LEARNING_INFO_H_

#include <vector>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <map>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

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
  enum Regularizer {NONE, L2, L1};
}

namespace OptAlgorithm {
  enum OptAlgorithm {GRADIENT_DESCENT, BLOCK_COORD_DESCENT, LBFGS, SIMULATED_ANNEALING, EXPECTATION_MAXIMIZATION};
}

namespace DebugLevel {
  enum DebugLevel {NONE=0, ESSENTIAL=1, CORPUS=2, MINI_BATCH=3, SENTENCE=4, TOKEN=5, REDICULOUS=6, TEMP = 4};
}

// documentation can be found at http://www.chokkan.org/software/liblbfgs/structlbfgs__parameter__t.html
struct LbfgsParams {
  int maxIterations;
  int maxEvalsPerIteration;
  int memoryBuffer;
  double precision;
  
  LbfgsParams() {
    maxIterations = 10;
    memoryBuffer = 6;
    precision = 0.000000000000001;
    maxEvalsPerIteration = 20;
  }
};

struct OptMethod {
  // some optimization algorithms (e.g. coordinat descent and expectation maximization) 
  // iteratively executes a simpler optimization algorithm
  OptMethod *subOptMethod;
  // specifies the algorithm used for optimization
  OptAlgorithm::OptAlgorithm algorithm;
  // if algorithm = LBFGS, use these LBFGS hyper parameters
  LbfgsParams lbfgsParams;
  // some optimization algorithms require specifying a learning rate (e.g. gradient descent)
  float learningRate;
  // stochastic = 0 means batch optimization
  // stochastic = 1 means online optimization
  bool stochastic;
  // when stochastic = 1, specifies the minibatch size
  int miniBatchSize;
  // regularization details
  Regularizer::Regularizer regularizer;
  float regularizationStrength;
  // move-away from previous weights penalty
  float moveAwayPenalty;

  OptMethod() {
    stochastic = false;
    algorithm = OptAlgorithm::GRADIENT_DESCENT;
    learningRate = 0.01;
    miniBatchSize = 1;
    regularizer = Regularizer::NONE;
    regularizationStrength = 1000;
    subOptMethod = 0;
    moveAwayPenalty = 1.0;
  }
}; 

namespace ConstraintType {
  enum ConstraintType {
    // an observed type xI must be assigned label yI
    yIExclusive_xIString, 
    // an observed type xI can be assigned label yI (i.e. other labels can also be valid)
    yI_xIString};
}

struct Constraint {
  void *field1, *field2;
  ConstraintType::ConstraintType type;

  Constraint(const Constraint &c) {
    int yI;
    std::string xI;
    switch(c.type) {
    case ConstraintType::yIExclusive_xIString:
      c.GetFieldsOfConstraintType_yIExclusive_xIString(yI, xI);
      this->SetConstraintOfType_yIExclusive_xIString(yI, xI);
      break;
    case ConstraintType::yI_xIString:
      c.GetFieldsOfConstraintType_yI_xIString(yI, xI);
      this->SetConstraintOfType_yI_xIString(yI, xI);
      break;
    default:
      assert(false);
      break;
    }
  }

  Constraint() {
    field1 = 0;
    field2 = 0;
  }

  ~Constraint() {
    if(field1 != 0) {
      switch(type) {
      case ConstraintType::yIExclusive_xIString:
      case ConstraintType::yI_xIString:
	delete (int *)field1;
	break;
      default:
	assert(false);
	break;
      }
    }
    if(field2 != 0) {
      switch(type) {
      case ConstraintType::yIExclusive_xIString:
      case ConstraintType::yI_xIString:
	delete (string *)field2;
	break;
      default:
	assert(false);
	break;
      }
    }
  }
  
  void SetConstraintOfType_yIExclusive_xIString(int yI, const std::string &xI) {
    type = ConstraintType::yIExclusive_xIString;
    field1 = new int[1];
    *((int*)field1) = yI;
    field2 = new std::string();
    *((string*)field2) = xI;
  }
  
  void GetFieldsOfConstraintType_yIExclusive_xIString(int &yI, std::string &xI) const {
    assert(type == ConstraintType::yIExclusive_xIString);
    yI = *(int*)field1;
    xI = *(string*)field2;
  }

  void SetConstraintOfType_yI_xIString(int yI, const std::string &xI) {
    type = ConstraintType::yI_xIString;
    field1 = new int[1];
    *((int*)field1) = yI;
    field2 = new std::string();
    *((string*)field2) = xI;
  }
  
  void GetFieldsOfConstraintType_yI_xIString(int &yI, std::string &xI) const {
    assert(type == ConstraintType::yI_xIString);
    yI = *(int*)field1;
    xI = *(string*)field2;
  }
};

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
    debugLevel = 1;
    useSparseVectors = true;
    persistParamsAfterNIteration = 100;
    persistFinalParams = true;
    retryLbfgsOnRoundingErrors = true;
    zIDependsOnYIM1 = false;
    supervisedTraining = false;
    unspecified = 0;
    unspecified2 = 0;
  }

  bool IsModelConverged() {
    assert(useMaxIterationsCount || useMinLikelihoodDiff || useEarlyStopping || useMinLikelihoodRelativeDiff);
    
    // logging
    if(useMaxIterationsCount) {
      cerr << "rank #" << mpiWorld->rank() << ": iterationsCount = " << iterationsCount << ". max = " << maxIterationsCount << endl;
    }
    if(useMinLikelihoodDiff && 
       iterationsCount > 1) {
      cerr << "rank #" << mpiWorld->rank() << ": loglikelihoodDiff = " << fabs(logLikelihood[iterationsCount-1] - 
					  logLikelihood[iterationsCount-2]) << ". min = " << minLikelihoodDiff << endl;
    }
    if(useEarlyStopping &&
       iterationsCount > 1) {
      cerr << "rank #" << mpiWorld->rank() << ": validationLikelihood[" << iterationsCount-2 << "] = " << validationLogLikelihood[iterationsCount-2] << endl;
      cerr << "rank #" << mpiWorld->rank() << ": validationLikelihood[" << iterationsCount-1 << "] = " << validationLogLikelihood[iterationsCount-1] << endl;
      cerr << "rank #" << mpiWorld->rank() << ": convergence criterion: stop training when loglikelihood no longer decreases, after the second iteration" << endl;
    }

    double absoluteDiff = 0.0, relativeDiff = 0.0;
    if(useMinLikelihoodRelativeDiff &&
       iterationsCount > 1) {
      absoluteDiff = fabs(logLikelihood[iterationsCount-1] - logLikelihood[iterationsCount-2]);
      assert(logLikelihood[iterationsCount-2] != 0);
      relativeDiff = fabs(absoluteDiff / logLikelihood[iterationsCount-2]);
      cerr << "rank #" << mpiWorld->rank() << ": loglikelihoodRelativeDiff = " << relativeDiff << ".min = " << minLikelihoodRelativeDiff << endl;
    }
    
    cerr << endl;
    // check for convergnece conditions
    if(useMaxIterationsCount && 
       maxIterationsCount <= iterationsCount) {
      return true;
    } 
    if(useMinLikelihoodDiff && 
       iterationsCount > 2 &&
       minLikelihoodDiff > fabs(logLikelihood[iterationsCount-1] - 
					    logLikelihood[iterationsCount-2])) {
      return true;
    } 
    if(useEarlyStopping &&
       iterationsCount > 5 &&
       validationLogLikelihood[iterationsCount-1] - validationLogLikelihood[iterationsCount-2] > 0) {
      return true;
    }
    if(useMinLikelihoodRelativeDiff && 
       iterationsCount > 2 && 
       minLikelihoodRelativeDiff > relativeDiff) {
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

  // criteria 4
  bool useMinLikelihoodRelativeDiff;
  float minLikelihoodRelativeDiff;

  // optimization method
  OptMethod optimizationMethod;

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

  // 0 = no debug info. 
  // 1 = corpus level debug info.
  // 2 = mini-batch level debug info.
  // 3 = sentence level debug info. 
  // 4 = token level debug info.
  unsigned debugLevel;

  // this field can be used to communicate to the underlying model that certain combinations are required/forbidden
  std::vector<Constraint> constraints;

  // do we use cdec's FastSparseVector when applicable?
  bool useSparseVectors;
  
  // frequency of persisting the params
  unsigned persistParamsAfterNIteration;

  // persist the parameters after training?
  bool persistFinalParams;

  // boost mpi communicator
  boost::mpi::communicator *mpiWorld;

  // if lbfgs returns a rounding error. should we retry?
  bool retryLbfgsOnRoundingErrors;

  // in the latent crf model, does the output observations z_i depend on y_{i-1}, or only y_i?
  bool zIDependsOnYIM1;

  bool supervisedTraining;

  int unspecified, unspecified2;
};



#endif
