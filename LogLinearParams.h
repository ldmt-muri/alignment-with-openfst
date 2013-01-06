#ifndef _LOG_LINEAR_PARAMS_H_
#define _LOG_LINEAR_PARAMS_H_

#include <map>
#include <string>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <cmath>

#include "LearningInfo.h"
#include "VocabEncoder.h"

class LogLinearParams {
 public:

  // for the loglinear word alignment model
  LogLinearParams(const VocabDecoder &srcTypes, 
		  const VocabDecoder &tgtTypes, 
		  const std::map<int, std::map<int, double> > &ibmModel1ForwardLogProbs,
		  const std::map<int, std::map<int, double> > &ibmModel1BackwardLogProbs);

  // for the latent CRF model
  LogLinearParams(const VocabDecoder &types);
  
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
		    std::map<string, double> &activeFeatures);
    
  // if the paramId does not exist, add it. otherwise, do nothing. 
  bool AddParam(std::string paramId, double paramWeight=0.0);

  // compute dot product between a sparse vector (passed) represented as a map, and the feature weights (member)
  double DotProduct(const std::map<string, double>& values);
  
  // compute dot product of two vectors 
  double DotProduct(const std::vector<double>& values, const std::vector<double>& weights);

  // compute dot product of feature values (passed), and feature weights (member)
  double DotProduct(const std::vector<double>& values);

  // given description of an arc in the alignment transducer, compute the local arc probability
  double ComputeLogProb(int srcToken, int prevSrcToken, int tgtToken, int srcPos, int prevSrcPos, int tgtPos, int srcSentLength, int tgtSentLength,
		       const std::vector<bool>& enabledFeatureTypes);

  // updates the model parameters given the gradient and an optimization method
  void UpdateParams(const std::map<std::string, double> &gradient, const OptMethod &optMethod);
  
  // override the member weights vector with this array
  void UpdateParams(const double* array, const int arrayLength);

  // update a single parameter's value (adds the parameter if necessary)
  void UpdateParam(const std::string paramId, const double newValue) {
    if(!AddParam(paramId, newValue)) {
      paramWeights[paramIndexes[paramId]] = newValue;
    }
  }

  // copies the weight of the specified feature from paramWeights vector to oldParamWeights vector
  void UpdateOldParam(const std::string paramId) {
    if(!AddParam(paramId)) {
      oldParamWeights[paramIndexes[paramId]] = paramWeights[paramIndexes[paramId]];
    }
  }

  // copies the weight of all features from paramWeights vector to oldParamWeights vector
  void UpdateOldParams() {
    for(int i = 0; i < paramWeights.size(); i++) {
      oldParamWeights[i] = paramWeights[i];
    }
  }  

  // returns the current weight of this param (adds the parameter if necessary)
  double GetParam(const std::string paramId) {
    AddParam(paramId);
    return paramWeights[paramIndexes[paramId]];
  }
  
  // returns the difference between new and old weights of a parameter, given its string ID
  double GetParamNewMinusOld(const std::string paramId) {
    AddParam(paramId);
    return paramWeights[paramIndexes[paramId]] - oldParamWeights[paramIndexes[paramId]];
  }

  int GetParamsCount() {
    assert(paramWeights.size() == paramIndexes.size());
    return paramWeights.size();
  }

  // returns a pointer to the array of parameter weights
  double* GetParamWeightsArray() {
    return paramWeights.data();
  }

  // returns a pointer to the array of old parameter weights
  double* GetOldParamWeightsArray() {
    return oldParamWeights.data();
  }

  // converts a map into an array.
  // when constrainedFeaturesCount is non-zero, length(valuesArray)  should be = valuesMap.size() - constrainedFeaturesCount,
  // we pretend as if the constrained features don't exist by subtracting the internal index - constrainedFeaturesCount 
  void ConvertFeatureMapToFeatureArray(map<string, double>& valuesMap, double* valuesArray, unsigned constrainedFeaturesCount = 0) {
    // init to 0
    for(int i = constrainedFeaturesCount; i < paramIndexes.size(); i++) {
      valuesArray[i-constrainedFeaturesCount] = 0;
    }
    // set the active features
    for(map<string, double>::const_iterator valuesMapIter = valuesMap.begin(); valuesMapIter != valuesMap.end(); valuesMapIter++) {
      // skip constrained features
      if(paramIndexes[valuesMapIter->first] < constrainedFeaturesCount) {
	continue;
      }
      // set the modified index in valuesArray
      valuesArray[ paramIndexes[valuesMapIter->first]-constrainedFeaturesCount ] = valuesMapIter->second;
    }
  }

  // 1/2 * sum of the squares
  double ComputeL2Norm() {
    double l2 = 0;
    for(int i = 0; i < paramWeights.size(); i++) {
      l2 += paramWeights[i] * paramWeights[i];
    }
    return l2/2;
  }
  
  // applies the cumulative l1 penalty on feature weights, also updates the appliedL1Penalty values
  void ApplyCumulativeL1Penalty(const LogLinearParams& applyToFeaturesHere,
				LogLinearParams& appliedL1Penalty,
				const double correctL1Penalty);

  // clear model parameters
  inline void Clear() {
    paramIndexes.clear();
    paramWeights.clear();
  }

  void PrintFirstNParams(unsigned n);

  void PrintParams();
  
  static void PrintParams(std::map<std::string, double> tempParams);

  // writes the features to a text file formatted one feature per line. 
  void PersistParams(const std::string& outputFilename);
  
  std::map< std::string, int > paramIndexes;
  std::vector< double > paramWeights;
  std::vector< double > oldParamWeights;

  // maps a word id into a string
  const VocabDecoder &srcTypes, &tgtTypes;

  // TODO: inappropriate for this general class. consider adding to a derived class
  // maps [srcTokenId][tgtTokenId] => forward logprob
  // maps [tgtTokenId][srcTokenId] => backward logprob
  const std::map< int, std::map< int, double > > &ibmModel1ForwardScores, &ibmModel1BackwardScores;

  const int COUNT_OF_FEATURE_TYPES;

  const LearningInfo *learningInfo;
};

#endif
