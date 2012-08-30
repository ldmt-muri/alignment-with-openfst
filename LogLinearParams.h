#ifndef _LOG_LINEAR_PARAMS_H_
#define _LOG_LINEAR_PARAMS_H_

#include <map>
#include <string>
#include <sstream>
#include <fstream>
#include <assert.h>

#include "LearningInfo.h"

class LogLinearParams {

 public:
  // given the description of one transition on the alignment FST, find the features that would fire along with their values
  void FireFeatures(int srcToken, int tgtToken, int srcPos, int tgtPos, int srcSentLength, int tgtSentLength, std::map<std::string, float>& activeFeatures);

  // compute dot product of two sparse vectors, each represented with a map. 
  float DotProduct(const std::map<std::string, float>& values, const std::map<std::string, float>& weights);

  // given description of an arc in the alignment transducer, compute the local arc probability
  float ComputeLogProb(int srcToken, int tgtToken, int srcPos, int tgtPos, int srcSentLength, int tgtSentLength);

  // updates the model parameters given the gradient and an optimization method
  void UpdateParams(const LogLinearParams& gradient, const OptUtils::OptMethod& optMethod);

  // clear model parameters
  inline void Clear() {
    params.clear();
  }

  // writes the features to a text file formatted one feature per line. 
  void PersistParams(const std::string& outputFilename);
  
  std::map< std::string, float > params;
};

#endif
