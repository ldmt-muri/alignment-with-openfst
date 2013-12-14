#ifndef _BASIC_TYPES_H_
#define _BASIC_TYPES_H_

enum FeatureTemplate { LABEL_BIGRAM=0, SRC_BIGRAM=1, ALIGNMENT_JUMP=2, LOG_ALIGNMENT_JUMP=3, ALIGNMENT_JUMP_IS_ZERO=4, SRC0_TGT0=5, PRECOMPUTED=6, DIAGONAL_DEVIATION=7, SYNC_START=8, SYNC_END=9, OTHER_ALIGNERS=10, NULL_ALIGNMENT=11 , NULL_ALIGNMENT_LENGTH_RATIO=12 , EMISSION=13, SRC_WORD_BIAS=14, BOUNDARY_LABELS=15};

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
  enum Regularizer {NONE, L2, L1, WeightedL2};
}

namespace OptAlgorithm {
  enum OptAlgorithm {GRADIENT_DESCENT, STOCHASTIC_GRADIENT_DESCENT, 
    BLOCK_COORD_DESCENT, LBFGS, SIMULATED_ANNEALING, EXPECTATION_MAXIMIZATION,
    ADAGRAD};
}

namespace DebugLevel {
  enum DebugLevel {NONE=0, ESSENTIAL=1, CORPUS=2, MINI_BATCH=3, SENTENCE=4, TOKEN=5, REDICULOUS=6, TEMP = 4};
}

// documentation can be found at the paper http://www.cs.berkeley.edu/~jduchi/projects/DuchiHaSi10.pdf
struct AdagradParams {
  double eta;
  int maxIterations;
};

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
    maxEvalsPerIteration = 3;
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
  // if algorithm = ADAGRAD, use these ADAGRAD hyper params
  //AdagradParams adagradParams;
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

/*
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
*/

#endif
