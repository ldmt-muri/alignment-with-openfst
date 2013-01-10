#ifndef _MULTINOMIAL_PARAMS_H_
#define _MULTINOMIAL_PARAMS_H_

#include <map>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <cmath>

#include "Samplers.h"

namespace MultinomialParams {

  // parameters for describing a multinomial distribution p(x)=y such that x is the key and y is a log probability
  typedef std::map<int, float> MultinomialParam;
  
  // parameters for describing a set of conditional multinomial distributions p(x|y)=z such that y is the first key, x is the nested key, z is a log probability
  typedef std::map<int, MultinomialParam> ConditionalMultinomialParam;

  static const int NLOG_ZERO = 300;
  static const int NLOG_INF = -200;

  inline float nLog(double prob) {
    if(prob <= 0) {
      //      std::cerr << "ERROR: MultinomialParams::nLog(" << prob << ") is undefined. instead, I returned " << NLOG_ZERO << " and continued." << std::endl;
      std::cerr << "$";
      return NLOG_ZERO;
    }
    return -1.0 * log(prob);
  }
  
  inline double nExp(float exponent) {
    if(exponent <= NLOG_INF) {
      //      std::cerr << "ERROR: MultinomialParams::nExp(" << exponent << ") is infinity. returned I returned exp(" << NLOG_INF << ") and continued." << std::endl;
      std::cerr << "#";
      exponent = NLOG_INF;
    }
    return exp(-1.0 * exponent);
  }

  // refactor variable names here (e.g. translations)
  // normalizes ConditionalMultinomialParam parameters such that \sum_t p(t|s) = 1 \forall s
  void NormalizeParams(ConditionalMultinomialParam& params);
  
  // zero all parameters
  void ClearParams(ConditionalMultinomialParam& params);

  // refactor variable names here (e.g. translations)
  void PrintParams(const ConditionalMultinomialParam& params);

  void PersistParams(std::ofstream& paramsFile, const ConditionalMultinomialParam& params);

  void PersistParams(const std::string& paramsFilename, const ConditionalMultinomialParam& params);

  // sample an integer from a multinomial
  int SampleFromMultinomial(const MultinomialParam params);
}

#endif
