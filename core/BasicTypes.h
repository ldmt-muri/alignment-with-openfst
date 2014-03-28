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

// one rich observation (e.g. token, its brown cluster, its POS tag, its morphological analysis ...etc)
struct ObservationDetails {
  ObservationDetails() {}
  ObservationDetails(const std::vector<int64_t> &_details) { this->details = _details; }
  std::vector<int64_t> details;
};

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
  double l1Strength;
  
  LbfgsParams() {
    maxIterations = 10;
    memoryBuffer = 6;
    precision = 0.000000000000001;
    maxEvalsPerIteration = 3;
    l1Strength = 0.0;
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

struct PosFactorId
{
public:
  int64_t yI, yIM1, xI, xIM1, xIM2, xIP1, xIP2;
  inline void Print() const {
    std::cerr << "(yI=" << yI << ", yIM1=" << yIM1 << ", xI=" << xI << ", xIM1=" << xIM1 << ", xIM2=" << xIM2 << ", xIP1=" << xIP1 << ", xIP2=" << xIP2 << ")" << std::endl;
  }

  inline bool operator < (const PosFactorId &other) const {
    if(yI != other.yI) {
      return yI < other.yI;
    } else if(yIM1 != other.yIM1) {
      return yIM1 < other.yIM1;
    } else if(xI != other.xI){
      return xI < other.xI;
    } else if(xIM1 != other.xIM1) {
      return xIM1 < other.xIM1;
    } else if(xIM2 != other.xIM2) {
      return xIM2 < other.xIM2;
    } else if(xIP1 != other.xIP1) {
      return xIP1 < other.xIP1;
    } else if(xIP2 != other.xIP2) {
      return xIP2 < other.xIP2;
    } else {
      return false;
    }
  }

  struct PosFactorHash : public std::unary_function<PosFactorId, size_t> {
    size_t operator()(const PosFactorId& x) const {
      size_t seed = 0;
      boost::hash_combine(seed, x.yI);
      boost::hash_combine(seed, x.yIM1);
      boost::hash_combine(seed, x.xI);
      boost::hash_combine(seed, x.xIM2);
      boost::hash_combine(seed, x.xIM1);
      boost::hash_combine(seed, x.xI);
      boost::hash_combine(seed, x.xIP1);
      boost::hash_combine(seed, x.xIP2);
      return seed;
    }
  };
  
  struct PosFactorEqual : public std::unary_function<PosFactorId, bool> {
    bool operator()(const PosFactorId& left, const PosFactorId& right) const {
      return left.yI == right.yI && left.yIM1 == right.yIM1 &&
        left.xIM2 == right.xIM2 && left.xIM1 == right.xIM1 &&
        left.xI == right.xI && left.xIP1 == right.xIP1 &&
        left.xIP2 == right.xIP2;
    }
  };
};

struct ParserFactorId
{
public:
  int childPosition, childWord, parentPosition, parentWord;

  inline void Print() const {
    std::cerr << "(childPosition=" << childPosition << ", childWord=" << childWord << ", parentPosition=" << parentPosition << ", parentWord=" << parentWord << ")" << std::endl;
  }

  inline bool operator < (const ParserFactorId &other) const {
    if(childPosition != other.childPosition) {
      return childPosition < other.childPosition;
    } else if(childWord != other.childWord) {
      return childWord < other.childWord;
    } else if(parentPosition != other.parentPosition) {
      return parentPosition < other.parentPosition;
    } else if(parentWord != other.parentWord) {
      return parentWord < other.parentWord;
    } else {
      return false;
    }
  }
  
  struct ParserFactorHash : public std::unary_function<ParserFactorId, size_t> {
    size_t operator()(const ParserFactorId& x) const {
      size_t seed = 0;
      boost::hash_combine(seed, (unsigned char)x.childPosition);
      boost::hash_combine(seed, (unsigned short)x.childWord);
      boost::hash_combine(seed, (unsigned char)x.parentPosition);
      boost::hash_combine(seed, (unsigned short)x.parentWord);
      return seed;
    }
  };
  
  struct ParserFactorEqual : public std::unary_function<ParserFactorId, bool> {
    bool operator()(const ParserFactorId& left, const ParserFactorId& right) const {
      return left.childPosition == right.childPosition && left.childWord == right.childWord &&
        left.parentPosition == right.parentPosition && left.parentWord == right.parentWord;
    }
  };

};

struct AlignerFactorId 
{ 
public:
  int yI, yIM1, i, srcWord, prevSrcWord, tgtWord, prevTgtWord, nextTgtWord; 
  inline void Print() const {
    std::cerr << "(yI=" << yI << ",yIM1=" << yIM1 << ",i=" << i << ",srcWord="  << srcWord << ",prevSrcWord=" << prevSrcWord << ",tgtWord=" <<  tgtWord << ",prevTgtWord=" << prevTgtWord << ",nextTgtWord=" << nextTgtWord << ")" << std::endl;
  }
  inline bool operator < (const AlignerFactorId &other) const {
    if(yI != other.yI) {
      return yI < other.yI;
    } else if(yIM1 != other.yIM1) {
      return yIM1 < other.yIM1;
    } else if(i != other.i){
      return i < other.i;
    } else if(srcWord != other.srcWord) {
      return srcWord < other.srcWord;
    } else if(prevSrcWord != other.prevSrcWord) {
      return prevSrcWord < other.prevSrcWord;
    } else if(tgtWord != other.tgtWord) {
      return tgtWord < other.tgtWord;
    } else if(prevTgtWord != other.prevTgtWord) {
      return prevTgtWord < other.prevTgtWord;
    } else if(nextTgtWord != other.nextTgtWord){ 
      return nextTgtWord < other.nextTgtWord;
    } else {
      return false;
    }
  }

  struct AlignerFactorHash : public std::unary_function<AlignerFactorId, size_t> {
    size_t operator()(const AlignerFactorId& x) const {
      size_t seed = 0;
      boost::hash_combine(seed, (unsigned char)x.i);
      //boost::hash_combine(seed, x.nextTgtWord);
      //boost::hash_combine(seed, x.prevSrcWord);
      //boost::hash_combine(seed, x.prevTgtWord);
      boost::hash_combine(seed, (unsigned short)x.srcWord);
      boost::hash_combine(seed, (unsigned short)x.tgtWord);
      boost::hash_combine(seed, (unsigned char)x.yI);
      boost::hash_combine(seed, (unsigned char)x.yIM1);
      return seed;
      //return std::hash<int>()(x.i + x.nextTgtWord + x.prevSrcWord + x.prevTgtWord + x.srcWord + x.tgtWord + x.yI + x.yIM1);
    }
  };

  struct AlignerFactorEqual : public std::unary_function<AlignerFactorId, bool> {
    bool operator()(const AlignerFactorId& left, const AlignerFactorId& right) const {
      return left.i == right.i && left.nextTgtWord == right.nextTgtWord &&
              left.prevSrcWord == right.prevSrcWord && left.prevTgtWord == right.prevTgtWord &&
              left.srcWord == right.srcWord && left.tgtWord == right.tgtWord &&
        left.yI == right.yI && left.yIM1 == right.yIM1;
    }
  };

};

#endif
