#ifndef _LOG_LINEAR_PARAMS_H_
#define _LOG_LINEAR_PARAMS_H_

#include <map>
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

#include "cdec-utils/fast_sparse_vector.h"

#include "LearningInfo.h"
#include "VocabEncoder.h"
#include "Samplers.h"

struct AlignerFactorId 
{ 
public:
  int yI, yIM1, i, srcWord, prevSrcWord, tgtWord, prevTgtWord, nextTgtWord; 
  int Signature() const { return yI + yIM1 + i + srcWord + prevSrcWord + tgtWord + prevTgtWord + nextTgtWord;}
  inline bool operator < (const AlignerFactorId &other) const {
    return Signature() < other.Signature();
  }
};

class LogLinearParams {

  // inline and template member functions
#include "LogLinearParams-inl.h"

 public:

  // for the loglinear word alignment model
  LogLinearParams(const VocabEncoder &types, 
		  const std::map<int, std::map<int, double> > &ibmModel1ForwardLogProbs,
		  const std::map<int, std::map<int, double> > &ibmModel1BackwardLogProbs,
		  double gaussianStdDev = 0.01);

  // for the latent CRF model
  LogLinearParams(const VocabEncoder &types, double gaussianStdDev = 1);
  
  void LoadPrecomputedFeaturesWith2Inputs(const std::string &wordPairFeaturesFilename);

  void AddToPrecomputedFeaturesWith2Inputs(int input1, int input2, std::string &featureId, double featureValue);

  double Hash();

  // set learning info
  void SetLearningInfo(const LearningInfo &learningInfo);

  // given the description of one transition on the alignment FST, find the features that would fire along with their values
  // note: pos here is short for position
  void FireFeatures(int srcToken, int prevSrcToken, int tgtToken, 
		    int srcPos, int prevSrcPos, int tgtPos, 
		    int srcSentLength, int tgtSentLength, 
		    const std::vector<bool>& enabledFeatureTypes, std::map<std::string, double>& activeFeatures);
  
  void FireFeatures(int yI, int yIM1, const vector<int> &x, int i, 
		    const std::vector<bool> &enabledFeatureTypes, 
		    FastSparseVector<double> &activeFeatures);
  
  void FireFeatures(int yI, int yIM1, const vector<int> &x_t, const vector<int> &x_s, int i, 
		    int START_OF_SENTENCE_Y_VALUE, int NULL_POS,
		    const std::vector<bool> &enabledFeatureTypes, 
		    FastSparseVector<double> &activeFeatures);

  // if the paramId does not exist, add it with weight drawn from gaussian. otherwise, do nothing. 
  bool AddParam(const std::string &paramId);
  
  // if the paramId does not exist, add it. otherwise, do nothing. 
  bool AddParam(const std::string &paramId, double paramWeight);

  // side effect: adds zero weights for parameter IDs present in values but not present in paramIndexes and paramWeights
  double DotProduct(const std::map<std::string, double>& values);

  double DotProduct(const std::vector<double>& values);

  double DotProduct(const std::vector<double>& values, const std::vector<double>& weights);

  double DotProduct(const FastSparseVector<double> &values);

  double DotProduct(const FastSparseVector<double> &values, const std::vector<double>& weights);
 
  // updates the model parameters given the gradient and an optimization method
  void UpdateParams(const std::map<std::string, double> &gradient, const OptMethod &optMethod);
  
  // override the member weights vector with this array
  void UpdateParams(const double* array, const int arrayLength);
  
  // converts a map into an array. 
  // when constrainedFeaturesCount is non-zero, length(valuesArray)  should be = valuesMap.size() - constrainedFeaturesCount, 
  // we pretend as if the constrained features don't exist by subtracting the internal index - constrainedFeaturesCount  
  void ConvertFeatureMapToFeatureArray(map<string, double>& valuesMap, double* valuesArray, unsigned constrainedFeaturesCount = 0);

  // 1/2 * sum of the squares 
  double ComputeL2Norm();
  
  // applies the cumulative l1 penalty on feature weights, also updates the appliedL1Penalty values 
  void ApplyCumulativeL1Penalty(const LogLinearParams& applyToFeaturesHere, 
   				LogLinearParams& appliedL1Penalty, 
   				const double correctL1Penalty); 

  void PrintFirstNParams(unsigned n); 
  
  void PrintParams(); 
  
  static void PrintParams(std::map<std::string, double> tempParams); 
  
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
  std::map< std::string, int > paramIndexes;
  std::vector< double > paramWeights; 
  std::vector< double > oldParamWeights; 
  std::vector< std::string > paramIds; 
  
  // maps a word id into a string
  const VocabEncoder &types;

  // TODO: inappropriate for this general class. consider adding to a derived class
  // maps [srcTokenId][tgtTokenId] => forward logprob
  // maps [tgtTokenId][srcTokenId] => backward logprob
  const std::map< int, std::map< int, double > > &ibmModel1ForwardScores, &ibmModel1BackwardScores;
  
  const int COUNT_OF_FEATURE_TYPES;
  
  const LearningInfo *learningInfo;

  const GaussianSampler *gaussianSampler;

  const std::set< int > *englishClosedClassTypes;
  
  std::map< int, std::map< int, std::map<std::string, double> > > precomputedFeaturesWithTwoInputs;

  std::map< AlignerFactorId, FastSparseVector<double> > factorIdToFeatures;

};

#endif
