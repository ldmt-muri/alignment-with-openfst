#ifndef _LOG_LINEAR_PARAMS_H_
#define _LOG_LINEAR_PARAMS_H_

#include <string>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <cmath>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/functional/hash.hpp>
#include <boost/unordered_map.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
//#include "boost/archive/binary_oarchive.hpp"
//#include "boost/archive/binary_iarchive.hpp"

#include "unordered_map_serialization.hpp"

#include "cdec-utils/fast_sparse_vector.h"

#include "LearningInfo.h"
#include "VocabEncoder.h"
#include "Samplers.h"

struct FeatureId {
public:
  static VocabEncoder *precomputedFeaturesEncoder;
  FeatureTemplate type;
  union {
    struct { unsigned current, previous; } bigram;
    int alignmentJump;
    struct { unsigned srcWord, tgtWord; } wordPair;
    int precomputed;
  };
  
  
  // replace this with a copy constructor 
  //const string& DecodePrecomputedFeature() const {
  //  assert(type == FeatureTemplate::PRECOMPUTED);
  //  return precomputedFeaturesEncoder->Decode(precomputed);
  //}

  void EncodePrecomputedFeature(const string& featureIdString) {
    assert(type == FeatureTemplate::PRECOMPUTED);
    precomputed = precomputedFeaturesEncoder->Encode(featureIdString);
  }
    
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & type;
    switch(type) {
      case FeatureTemplate::ALIGNMENT_JUMP:
      case FeatureTemplate::ALIGNMENT_JUMP_IS_ZERO:
      case FeatureTemplate::LOG_ALIGNMENT_JUMP:
        ar & alignmentJump;
        break;
      case FeatureTemplate::DIAGONAL_DEVIATION:
      case FeatureTemplate::SYNC_END:
      case FeatureTemplate::SYNC_START:
        break;
      case FeatureTemplate::LABEL_BIGRAM:
      case FeatureTemplate::SRC_BIGRAM:
        ar & bigram.previous;
        ar & bigram.current;
        break;
      case FeatureTemplate::PRECOMPUTED:
        ar & precomputed;
        break;
      case FeatureTemplate::SRC0_TGT0:
        ar & wordPair.srcWord;
        ar & wordPair.tgtWord;
        break;
      default:
        assert(false);
    }
  }

  bool operator!=(const FeatureId& rhs) const {
    if(type != rhs.type) return true;
    switch(type) {
      case FeatureTemplate::LABEL_BIGRAM:
      case FeatureTemplate::SRC_BIGRAM:
        return bigram.current != rhs.bigram.current || bigram.previous != rhs.bigram.previous;
        break;
      case FeatureTemplate::ALIGNMENT_JUMP:
      case FeatureTemplate::ALIGNMENT_JUMP_IS_ZERO:
      case FeatureTemplate::LOG_ALIGNMENT_JUMP:
        return alignmentJump != rhs.alignmentJump;
        break;
      case SRC0_TGT0:
        return wordPair.srcWord != rhs.wordPair.srcWord || wordPair.tgtWord != rhs.wordPair.tgtWord;
        break;
      case PRECOMPUTED:
        return precomputed != rhs.precomputed;
        break;
      case DIAGONAL_DEVIATION:
      case SYNC_START:
      case SYNC_END:
        return false;
        break;
      default:
        assert(false);
    }
  }
  
  bool operator==(const FeatureId& rhs) const {
    return !this->operator !=(rhs);
  }

  struct FeatureIdHash : public std::unary_function<FeatureId, size_t> {
    size_t operator()(const FeatureId& x) const {
      size_t seed = 0;
      boost::hash_combine(seed, x.type);
      switch(x.type) {
        case FeatureTemplate::LABEL_BIGRAM:
        case FeatureTemplate::SRC_BIGRAM:
          boost::hash_combine(seed, x.bigram.current);
          boost::hash_combine(seed, x.bigram.previous);
          break;
        case FeatureTemplate::ALIGNMENT_JUMP:
        case FeatureTemplate::ALIGNMENT_JUMP_IS_ZERO:
        case FeatureTemplate::LOG_ALIGNMENT_JUMP:
          boost::hash_combine(seed, x.alignmentJump);
          break;
        case SRC0_TGT0:
          boost::hash_combine(seed, x.wordPair.srcWord);
          boost::hash_combine(seed, x.wordPair.tgtWord);
          break;
        case PRECOMPUTED:
          boost::hash_combine(seed, x.precomputed);
          break;
        case DIAGONAL_DEVIATION:
        case SYNC_START:
        case SYNC_END:
          break;
        default:
          assert(false);
      }
      return seed;
    }
  };

  struct FeatureIdEqual : public std::unary_function<FeatureId, bool> {
    bool operator()(const FeatureId& left, const FeatureId& right) const {
      if(left.type != right.type) return false;
      switch(left.type) {
        case FeatureTemplate::LABEL_BIGRAM:
        case FeatureTemplate::SRC_BIGRAM:
          return left.bigram.current == right.bigram.current && left.bigram.previous == right.bigram.previous;
          break;
        case FeatureTemplate::ALIGNMENT_JUMP:
        case FeatureTemplate::ALIGNMENT_JUMP_IS_ZERO:
        case FeatureTemplate::LOG_ALIGNMENT_JUMP:
          return left.alignmentJump == right.alignmentJump;
          break;
        case SRC0_TGT0:
          return left.wordPair.srcWord == right.wordPair.srcWord && left.wordPair.tgtWord == right.wordPair.tgtWord;
          break;
        case PRECOMPUTED:
          return left.precomputed == right.precomputed;
          break;
        case DIAGONAL_DEVIATION:
        case SYNC_START:
        case SYNC_END:
          return true;
          break;
        default:
          assert(false);
      }
    }
  };
  
};

// a few typedefs
using unordered_map_featureId_double = boost::unordered_map<FeatureId, double, FeatureId::FeatureIdHash, FeatureId::FeatureIdEqual>;
using unordered_map_featureId_int = boost::unordered_map<FeatureId, int, FeatureId::FeatureIdHash, FeatureId::FeatureIdEqual>;

// Alias an STL compatible allocator of ints that allocates ints from the managed
// shared memory segment.  This allocator will allow to place containers
// in managed shared memory segments
typedef boost::interprocess::allocator<double, boost::interprocess::managed_shared_memory::segment_manager> ShmemDoubleAllocator;
typedef boost::interprocess::allocator<FeatureId, boost::interprocess::managed_shared_memory::segment_manager> ShmemFeatureIdAllocator;

// Alias a vector that uses the previous STL-like allocator
typedef vector<double, ShmemDoubleAllocator> ShmemVectorOfDouble;
typedef vector<FeatureId, ShmemFeatureIdAllocator> ShmemVectorOfFeatureId;

std::ostream& operator<<(std::ostream& os, const FeatureId& obj);
std::istream& operator>>(std::istream& is, FeatureId& obj);


class LogLinearParams {

  // inline and template member functions
#include "LogLinearParams-inl.h"

 public:

  // for the latent CRF model
  LogLinearParams(VocabEncoder &types, double gaussianStdDev = 1);

  template<class Archive>
    void save(Archive & os, const unsigned int version) const
    {
      cerr << "inside LogLinearParams::save()" << endl;
      assert(IsSealed());
      assert(paramIdsPtr->size() == paramWeightsPtr->size());
      int count = paramIdsPtr->size();
      os << count;
      for(int i = 0; i < paramIdsPtr->size(); i++) {
        os << (*paramIdsPtr)[i];
        os << (*paramWeightsPtr)[i];
      }
    }
  
  template<class Archive>
    void load(Archive & is, const unsigned int version)
    {
      cerr << "inside LogLinearParams::load()" << endl;
      int count;
      is >> count;
      for(int i = 0; i < count; ++i) {
        FeatureId featureId;
        is >> featureId;
        paramIdsTemp.push_back(featureId);
        double weight;
        is >> weight;
        paramWeightsTemp.push_back(weight);
      }
    }

  BOOST_SERIALIZATION_SPLIT_MEMBER()  
    
  // this method seals the set of parameters being used, not their weights
  void Seal(bool);
  bool IsSealed() const;

  // create shared memory object to hold the feature ids and weights
  void ManageSharedMemory(bool);

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
		    unordered_map_featureId_double& activeFeatures);
  
  void FireFeatures(int yI, int yIM1, const vector<int> &x, int i, 
		    FastSparseVector<double> &activeFeatures);
  
  void FireFeatures(int yI, int yIM1, const vector<int> &x_t, const vector<int> &x_s, int i, 
		    int START_OF_SENTENCE_Y_VALUE, int NULL_POS,
		    FastSparseVector<double> &activeFeatures);

  // if the paramId does not exist, add it with weight drawn from gaussian. otherwise, do nothing. 
  bool AddParam(const FeatureId &paramId);
  
  // if the paramId does not exist, add it. otherwise, do nothing. 
  bool AddParam(const FeatureId &paramId, double paramWeight);

  // side effect: adds zero weights for parameter IDs present in values but not present in paramIndexes and paramWeights
  double DotProduct(const unordered_map_featureId_double& values);

  double DotProduct(const std::vector<double>& values);

  double DotProduct(const std::vector<double>& values, const ShmemVectorOfDouble& weights);

  double DotProduct(const FastSparseVector<double> &values);

  double DotProduct(const FastSparseVector<double> &values, const ShmemVectorOfDouble& weights);
 
  // updates the model parameters given the gradient and an optimization method
  void UpdateParams(const unordered_map_featureId_double &gradient, const OptMethod &optMethod);
  
  // override the member weights vector with this array
  void UpdateParams(const double* array, const int arrayLength);
  
  // converts a map into an array. 
  // when constrainedFeaturesCount is non-zero, length(valuesArray)  should be = valuesMap.size() - constrainedFeaturesCount, 
  // we pretend as if the constrained features don't exist by subtracting the internal index - constrainedFeaturesCount  
  void ConvertFeatureMapToFeatureArray(unordered_map_featureId_double &valuesMap, double* valuesArray, unsigned constrainedFeaturesCount = 0);

  // 1/2 * sum of the squares 
  double ComputeL2Norm();
  
  // applies the cumulative l1 penalty on feature weights, also updates the appliedL1Penalty values 
  void ApplyCumulativeL1Penalty(const LogLinearParams& applyToFeaturesHere, 
   				LogLinearParams& appliedL1Penalty, 
   				const double correctL1Penalty); 

  void PrintFirstNParams(unsigned n); 
  
  void PrintParams(); 
  
  static void PrintParams(unordered_map_featureId_double &tempParams); 
  
  void PrintFeatureValues(FastSparseVector<double> &feats);

  // writes the features to a text file formatted one feature per line.  
  void PersistParams(const std::string& outputFilename, bool humanFriendly=true); 

  // loads the parameters
  void LoadParams(const std::string &inputFilename);

  // checks whether the "otherParams" have the same parameters and values as this object 
  // disclaimer: pretty expensive, and also requires that the parameters have the same order in the underlying vectors 
  bool IsIdentical(const LogLinearParams &otherParams);

  bool LogLinearParamsIsIdentical(const LogLinearParams &otherParams);

 public:
  // the actual parameters 
  unordered_map_featureId_int paramIndexes;
  ShmemVectorOfDouble *paramWeightsPtr; 
  std::vector< double > paramWeightsTemp; 
  ShmemVectorOfFeatureId *paramIdsPtr; 
  std::vector< FeatureId > paramIdsTemp; 
  
  // maps a word id into a string
  VocabEncoder &types;

  // maps precomputed feature strings into ids
  VocabEncoder precomputedFeaturesEncoder;

  const LearningInfo *learningInfo;

  const GaussianSampler *gaussianSampler;

  const set< int > *englishClosedClassTypes;
  
  boost::unordered_map< int, boost::unordered_map< int, unordered_map_featureId_double > > precomputedFeaturesWithTwoInputs;

 private:
  bool sealed;
  
};

#endif
