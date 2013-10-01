#ifndef _LOG_LINEAR_PARAMS_H_
#define _LOG_LINEAR_PARAMS_H_

#include <string>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <cmath>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/functional/hash.hpp>
#include <boost/unordered_map.hpp>

#include "unordered_map_serialization.hpp"

#include "cdec-utils/fast_sparse_vector.h"

#include "LearningInfo.h"
#include "VocabEncoder.h"
#include "Samplers.h"

enum FeatureTemplate { LABEL_BIGRAM, SRC_BIGRAM, ALIGNMENT_JUMP, LOG_ALIGNMENT_JUMP, ALIGNMENT_JUMP_IS_ZERO, SRC0_TGT0, PRECOMPUTED, DIAGONAL_DEVIATION, SYNC_START, SYNC_END };

struct FeatureId {
public:
  FeatureTemplate type;
  union {
    struct { unsigned current, previous; } bigram;
    int alignmentJump;
    struct { unsigned srcWord, tgtWord; } wordPair;
    int precomputed;
  };
  static string Write(FeatureId x) {
    string y;
    return y;
  }
  static FeatureId Read(string y) {
    FeatureId x;
    return x;
  }
};

std::ostream& operator<<(std::ostream& os, const FeatureId& obj)
{
  os << obj.type;
  switch(obj.type) {
  case FeatureTemplate::LABEL_BIGRAM:
  case FeatureTemplate::SRC_BIGRAM:
    os << obj.bigram.current << obj.bigram.previous;
    break;
  case FeatureTemplate::ALIGNMENT_JUMP:
  case FeatureTemplate::LOG_ALINGNMENT_JUMP:
  case FeatureTemplate::ALIGNMENT_JUMP_IS_ZERO:
    os << obj.alignmentJump;
    break;
  case FeatureTemplate::SRC0_TGT0:
    os << obj.wordPair.srcWord << obj.wordPair.tgtWord;
    break;
  case FeatureTemplate::PRECOMPUTED:
    os << obj.precomputed;
    break;
  case FeatureTemplate::DIAGONAL_DEVIATION:
  case FeatureTemplate::SYNC_START:
  case FeatureTemplate::SYNC_END:
    break;
  default:
    assert(false);
  }

  return os;
}

std::istream& operator>>(std::istream& is, T& obj)
{
  // read obj from stream
  if( /* no valid object of T found in stream */ )
    is.setstate(std::ios::failbit);

  is >> obj.type;
  switch(obj.type) {
  case FeatureTemplate::LABEL_BIGRAM:
  case FeatureTemplate::SRC_BIGRAM:
    is >> obj.bigram.current;
    is >> obj.bigram.previous;
    break;
  case FeatureTemplate::ALIGNMENT_JUMP:
  case FeatureTemplate::LOG_ALINGNMENT_JUMP:
  case FeatureTemplate::ALIGNMENT_JUMP_IS_ZERO:
    is >> obj.alignmentJump;
    break;
  case FeatureTemplate::SRC0_TGT0:
    is >> obj.wordPair.srcWord;
    is >> obj.wordPair.tgtWord;
    break;
  case FeatureTemplate::PRECOMPUTED:
    is >> obj.precomputed;
    break;
  case FeatureTemplate::DIAGONAL_DEVIATION:
  case FeatureTemplate::SYNC_START:
  case FeatureTemplate::SYNC_END:
    break;
  default:
    assert(false);
  }
  return is;
}

struct AlignerFactorId 
{ 
public:
  int yI, yIM1, i, srcWord, prevSrcWord, tgtWord, prevTgtWord, nextTgtWord; 
  inline void Print() const {
    std::cerr << "(yI=" << yI << ",yIM1=" << yIM1 << ",i=" << i << ",srcWord="  << srcWord << ",prevSrcWord=" << prevSrcWord << ",tgtWord=" <<  tgtWord << ",prevTgtWord=" << prevTgtWord << ",nextTgtWord=" << nextTgtWord << ")" << endl;
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

class LogLinearParams {

  // inline and template member functions
#include "LogLinearParams-inl.h"

 public:

  // for the loglinear word alignment model
  LogLinearParams(const VocabEncoder &types, 
		  const boost::unordered_map<int, boost::unordered_map<int, double> > &ibmModel1ForwardLogProbs,
		  const boost::unordered_map<int, boost::unordered_map<int, double> > &ibmModel1BackwardLogProbs,
		  double gaussianStdDev = 0.01);

  // for the latent CRF model
  LogLinearParams(const VocabEncoder &types, double gaussianStdDev = 1);
  
  void LoadPrecomputedFeaturesWith2Inputs(const std::string &wordPairFeaturesFilename);

  void AddToPrecomputedFeaturesWith2Inputs(int input1, int input2, FeatureId &featureId, double featureValue);

  double Hash();

  // set learning info
  void SetLearningInfo(const LearningInfo &learningInfo);

  // given the description of one transition on the alignment FST, find the features that would fire along with their values
  // note: pos here is short for position
  void FireFeatures(int srcToken, int prevSrcToken, int tgtToken, 
		    int srcPos, int prevSrcPos, int tgtPos, 
		    int srcSentLength, int tgtSentLength, 
		    const std::vector<bool>& enabledFeatureTypes, boost::unordered_map<FeatureId, double>& activeFeatures);
  
  void FireFeatures(int yI, int yIM1, const vector<int> &x, int i, 
		    const std::vector<bool> &enabledFeatureTypes, 
		    FastSparseVector<double> &activeFeatures);
  
  void FireFeatures(int yI, int yIM1, const vector<int> &x_t, const vector<int> &x_s, int i, 
		    int START_OF_SENTENCE_Y_VALUE, int NULL_POS,
		    const std::vector<bool> &enabledFeatureTypes, 
		    FastSparseVector<double> &activeFeatures);

  // if the paramId does not exist, add it with weight drawn from gaussian. otherwise, do nothing. 
  bool AddParam(const FeatureId &paramId);
  
  // if the paramId does not exist, add it. otherwise, do nothing. 
  bool AddParam(const FeatureId &paramId, double paramWeight);

  // side effect: adds zero weights for parameter IDs present in values but not present in paramIndexes and paramWeights
  double DotProduct(const boost::unordered_map<FeatureId, double>& values);

  double DotProduct(const std::vector<double>& values);

  double DotProduct(const std::vector<double>& values, const std::vector<double>& weights);

  double DotProduct(const FastSparseVector<double> &values);

  double DotProduct(const FastSparseVector<double> &values, const std::vector<double>& weights);
 
  // updates the model parameters given the gradient and an optimization method
  void UpdateParams(const boost::unordered_map<FeatureId, double> &gradient, const OptMethod &optMethod);
  
  // override the member weights vector with this array
  void UpdateParams(const double* array, const int arrayLength);
  
  // converts a map into an array. 
  // when constrainedFeaturesCount is non-zero, length(valuesArray)  should be = valuesMap.size() - constrainedFeaturesCount, 
  // we pretend as if the constrained features don't exist by subtracting the internal index - constrainedFeaturesCount  
  void ConvertFeatureMapToFeatureArray(boost::unordered_map<FeatureId, double>& valuesMap, double* valuesArray, unsigned constrainedFeaturesCount = 0);

  // 1/2 * sum of the squares 
  double ComputeL2Norm();
  
  // applies the cumulative l1 penalty on feature weights, also updates the appliedL1Penalty values 
  void ApplyCumulativeL1Penalty(const LogLinearParams& applyToFeaturesHere, 
   				LogLinearParams& appliedL1Penalty, 
   				const double correctL1Penalty); 

  void PrintFirstNParams(unsigned n); 
  
  void PrintParams(); 
  
  static void PrintParams(boost::unordered_map<FeatureId, double> tempParams); 
  
  void PrintFeatureValues(FastSparseVector<double> &feats);

  // writes the features to a text file formatted one feature per line.  
  void PersistParams(const std::string& outputFilename); 

  // loads the parameters
  void LoadParams(const std::string &inputFilename);

  // call boost::mpi::broadcast for the essential member variables of this object 
  void Broadcast(boost::mpi::communicator &world, unsigned root);
  
  // checks whether the "otherParams" have the same parameters and values as this object 
  // disclaimer: pretty expensive, and also requires that the parameters have the same order in the underlying vectors 
  bool IsIdentical(const LogLinearParams &otherParams);

  bool LogLinearParamsIsIdentical(const LogLinearParams &otherParams);

 public:
  // the actual parameters 
  boost::unordered_map< FeatureId, int > paramIndexes;
  std::vector< double > paramWeights; 
  std::vector< double > oldParamWeights; 
  std::vector< FeatureId > paramIds; 
  
  // maps a word id into a string
  const VocabEncoder &types;

  // TODO: inappropriate for this general class. consider adding to a derived class
  // maps [srcTokenId][tgtTokenId] => forward logprob
  // maps [tgtTokenId][srcTokenId] => backward logprob
  const boost::unordered_map< int, boost::unordered_map< int, double > > &ibmModel1ForwardScores, &ibmModel1BackwardScores;
  
  const int COUNT_OF_FEATURE_TYPES;
  
  const LearningInfo *learningInfo;

  const GaussianSampler *gaussianSampler;

  const set< int > *englishClosedClassTypes;
  
  boost::unordered_map< int, boost::unordered_map< int, boost::unordered_map<FeatureId, double> > > precomputedFeaturesWithTwoInputs;
  
  boost::unordered_map< AlignerFactorId, FastSparseVector<double>, AlignerFactorId::AlignerFactorHash, AlignerFactorId::AlignerFactorEqual > factorIdToFeatures;

};

#endif
