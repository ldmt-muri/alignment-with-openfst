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
#include "VocabEncoder.h"

namespace MultinomialParams {

  // parameters for describing a multinomial distribution p(x)=y such that x is the key and y is a log probability
  typedef std::map<int, double> MultinomialParam;
  
  // parameters for describing a set of conditional multinomial distributions p(x|y)=z such that y is the first key, x is the nested key, z is a log probability
  typedef std::map<int, MultinomialParam> ConditionalMultinomialParam;

  static const int NLOG_ZERO = 300;
  static const int NLOG_INF = -200;

  static MultinomialParam AccumulateMultinomials(const MultinomialParam& p1, const MultinomialParam& p2) {
    MultinomialParam pTotal(p1);
    for(std::map<int, double>::const_iterator p2Iter = p2.begin(); p2Iter != p2.end(); p2Iter++) {
      pTotal[p2Iter->first] += p2Iter->second;
    }
    return pTotal;
  }

  static ConditionalMultinomialParam AccumulateConditionalMultinomials(const ConditionalMultinomialParam& p1, const ConditionalMultinomialParam& p2) {
    ConditionalMultinomialParam pTotal(p1);
    for(std::map<int, MultinomialParam>::const_iterator p2Iter = p2.begin(); p2Iter != p2.end(); p2Iter++) {
      MultinomialParam &subPTotal = pTotal[p2Iter->first];
      for(std::map<int, double>::const_iterator subP2Iter = p2Iter->second.begin(); subP2Iter != p2Iter->second.end(); subP2Iter++) {
	subPTotal[subP2Iter->first] += subP2Iter->second;
      }
    }
    return pTotal;
  }

  inline double nLog(double prob) {
    if(prob <= 0) {
      //      std::cerr << "ERROR: MultinomialParams::nLog(" << prob << ") is undefined. instead, I returned " << NLOG_ZERO << " and continued." << std::endl;
      std::cerr << "$";
      return NLOG_ZERO;
    }
    return -1.0 * log(prob);
  }
  
  inline double nExp(double exponent) {
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
  void PrintParams(const ConditionalMultinomialParam& params, const VocabEncoder &encoder);

  void PersistParams(std::ofstream& paramsFile, const ConditionalMultinomialParam& params);
  void PersistParams(std::ofstream& paramsFile, const ConditionalMultinomialParam& params, const VocabEncoder &vocabEncoder);
  void PersistParams(const std::string& paramsFilename, const ConditionalMultinomialParam& params);
  void PersistParams(const std::string& paramsFilename, const ConditionalMultinomialParam& params, const VocabEncoder &vocabEncoder);

  // sample an integer from a multinomial
  int SampleFromMultinomial(const MultinomialParam params);
}

#endif
