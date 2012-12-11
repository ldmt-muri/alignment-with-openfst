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

  LogLinearParams(const VocabDecoder &srcTypes, 
		  const VocabDecoder &tgtTypes, 
		  const std::map<int, std::map<int, float> > &ibmModel1ForwardLogProbs,
		  const std::map<int, std::map<int, float> > &ibmModel1BackwardLogProbs);

  LogLinearParams(const VocabDecoder &types);

  // given the description of one transition on the alignment FST, find the features that would fire along with their values
  // note: pos here is short for position
  void FireFeatures(int srcToken, int prevSrcToken, int tgtToken, 
		    int srcPos, int prevSrcPos, int tgtPos, 
		    int srcSentLength, int tgtSentLength, 
		    const std::vector<bool>& enabledFeatureTypes, std::map<std::string, float>& activeFeatures);

  void FireFeatures(int yI, int yIM1, const vector<int> &x, int i, 
		    const std::vector<bool> &enabledFeatureTypes, 
		    std::map<string, float> &activeFeatures);
    
  // compute dot product of two sparse vectors, each represented with a map. 
  float DotProduct(const std::map<std::string, float>& values, const std::map<std::string, float>& weights);

  // compute dot product of feature values (passed), and feature weights (member variable 'params')
  float DotProduct(const std::map<std::string, float>& values);

  // given description of an arc in the alignment transducer, compute the local arc probability
  float ComputeLogProb(int srcToken, int prevSrcToken, int tgtToken, int srcPos, int prevSrcPos, int tgtPos, int srcSentLength, int tgtSentLength,
		       const std::vector<bool>& enabledFeatureTypes);

  // updates the model parameters given the gradient and an optimization method
  void UpdateParams(const map<string, float> &gradient, const OptUtils::OptMethod &optMethod);

  // use gradient based methods to update the model parameter weights
  void UpdateParams(const LogLinearParams &gradient, const OptUtils::OptMethod &optMethod);

  // applies the accumulative l1 penalty on feature weights, also updates the appliedL1Penalty values
  void ApplyCumulativeL1Penalty(const LogLinearParams& applyToFeaturesHere,
				LogLinearParams& appliedL1Penalty,
				const double correctL1Penalty);

  // compute the orthographic similarity between two strings
  float ComputeOrthographicSimilarity(const std::string& srcWord, const std::string& tgtWord);

  // levenshtein distance
  int LevenshteinDistance(const std::string& x, const std::string& y);

  // clear model parameters
  inline void Clear() {
    params.clear();
  }

  // writes the features to a text file formatted one feature per line. 
  void PersistParams(const std::string& outputFilename);
  
  std::map< std::string, float > params;

  // maps a word id into a string
  const VocabDecoder &srcTypes, &tgtTypes;

  // maps [srcTokenId][tgtTokenId] => forward logprob
  // maps [tgtTokenId][srcTokenId] => backward logprob
  const std::map< int, std::map< int, float > > &ibmModel1ForwardScores, &ibmModel1BackwardScores;
};

#endif
