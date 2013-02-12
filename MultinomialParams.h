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
#include <utility>
#include <tuple>

#include "Samplers.h"
#include "VocabEncoder.h"
#include "FstUtils.h"

namespace MultinomialParams {

  // parameters for describing a multinomial distribution p(x)=y such that x is the key and y is a log probability
  typedef std::map<int, double> MultinomialParam;

  // parameters for describing a set of conditional multinomial distributions p(x|y)=z such that y is the first key, x is the nested key, z is a log probability. y is of type ContextType. x is integer. z is double.
  // ContextType is the type of things you want to condition on. 
  template <class ContextType>
  class ConditionalMultinomialParam {

  public:
    ConditionalMultinomialParam() {
    }
    ConditionalMultinomialParam(ConditionalMultinomialParam &x) {
      params = x.params;
    }
    inline MultinomialParam& operator[](ContextType key) {
      return params[key];
    }
    
  public:
    std::map<ContextType, MultinomialParam> params;
  };

  static const int NLOG_SMOOTHING_CONSTANT = 1;
  static const int NLOG_ZERO = 300;
  static const int NLOG_INF = -200;

  template <typename ContextType>
  static std::map<ContextType, double> AccumulateMultinomials(const std::map<ContextType, double>& p1, 
							      const std::map<ContextType, double>& p2) {
    std::map<ContextType, double> pTotal(p1);
    for(typename std::map<ContextType, double>::const_iterator p2Iter = p2.begin(); p2Iter != p2.end(); p2Iter++) {
      pTotal[p2Iter->first] += p2Iter->second;
    }
    return pTotal;
  }

  template <typename ContextType>
  static std::map<ContextType, MultinomialParam> AccumulateConditionalMultinomials(const std::map<ContextType, MultinomialParam>& p1,
										    const std::map<ContextType, MultinomialParam>& p2) {
    std::map<ContextType, MultinomialParam> pTotal(p1);
    for(typename std::map<ContextType, MultinomialParam>::const_iterator p2Iter = p2.begin(); p2Iter != p2.end(); p2Iter++) {
      //MultinomialParam &subPTotal = pTotal[p2Iter->first];
      for(MultinomialParam::const_iterator subP2Iter = p2Iter->second.begin(); subP2Iter != p2Iter->second.end(); subP2Iter++) {
	pTotal[p2Iter->first][subP2Iter->first] += subP2Iter->second;
      }
    }
    return pTotal;
  }

  template <typename ContextType>
  static std::map<ContextType, MultinomialParam> AccumulateConditionalMultinomialsLogSpace(const std::map<ContextType, MultinomialParam>& p1,
											   const std::map<ContextType, MultinomialParam>& p2) {
    std::map<ContextType, MultinomialParam> pTotal(p1);
    for(typename std::map<ContextType, MultinomialParam>::const_iterator p2Iter = p2.begin(); p2Iter != p2.end(); p2Iter++) {
      //MultinomialParam &subPTotal = pTotal[p2Iter->first];
      for(MultinomialParam::const_iterator subP2Iter = p2Iter->second.begin(); subP2Iter != p2Iter->second.end(); subP2Iter++) {
	pTotal[p2Iter->first][subP2Iter->first] = 
	  fst::Plus(fst::LogWeight( pTotal[p2Iter->first][subP2Iter->first] ),
		    fst::LogWeight(subP2Iter->second)).Value();
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
  template <typename ContextType>
  void NormalizeParams(ConditionalMultinomialParam<ContextType>& params);
  
  // refactor variable names here (e.g. translations)
  // normalizes ConditionalMultinomialParam parameters such that \sum_t p(t|s) = 1 \forall s
  template <typename ContextType>
  void NormalizeParams(ConditionalMultinomialParam<ContextType> &params) {
    // iterate over src tokens in the model
    for(typename map<ContextType, MultinomialParam>::iterator srcIter = params.params.begin(); srcIter != params.params.end(); srcIter++) {
      MultinomialParam &translations = (*srcIter).second;
      double fTotalProb = 0.0;
      // iterate over tgt tokens logsumming over the logprob(tgt|src) 
      for(MultinomialParam::iterator tgtIter = translations.begin(); tgtIter != translations.end(); tgtIter++) {
	double temp = (*tgtIter).second;
	fTotalProb += nExp(temp);
      }
      // exponentiate to find p(*|src) before normalization
      // iterate again over tgt tokens dividing p(tgt|src) by p(*|src)
      double fVerifyTotalProb = 0.0;
      for(MultinomialParam::iterator tgtIter = translations.begin(); tgtIter != translations.end(); tgtIter++) {
	double fUnnormalized = nExp((*tgtIter).second);
	double fNormalized = fUnnormalized / fTotalProb;
	fVerifyTotalProb += fNormalized;
	double fLogNormalized = nLog(fNormalized);
	(*tgtIter).second = fLogNormalized;
      }
    }
  }
  
  // zero all parameters
  template <typename ContextType>
    void ClearParams(ConditionalMultinomialParam<ContextType>& params, bool smooth=false) {
    for (typename map<ContextType, MultinomialParam>::iterator srcIter = params.params.begin(); srcIter != params.params.end(); srcIter++) {
      for (MultinomialParam::iterator tgtIter = srcIter->second.begin(); tgtIter != srcIter->second.end(); tgtIter++) {
	if(smooth) {
	  tgtIter->second = NLOG_SMOOTHING_CONSTANT;
	} else {
	  tgtIter->second = NLOG_ZERO;
	}
      }
    }
  }
  
  // refactor variable names here (e.g. translations)
  template <typename ContextType>
  void PrintParams(const ConditionalMultinomialParam<ContextType>& params) {
    // iterate over src tokens in the model
    int counter = 0;
    for(typename map<ContextType, MultinomialParam>::const_iterator srcIter = params.params.begin(); srcIter != params.params.end(); srcIter++) {
      const MultinomialParam &translations = (*srcIter).second;
      // iterate over tgt tokens 
      for(MultinomialParam::const_iterator tgtIter = translations.begin(); tgtIter != translations.end(); tgtIter++) {
	std::cerr << "-logp(" << (*tgtIter).first << "|" << (*srcIter).first << ")=-log(" << nExp((*tgtIter).second) << ")=" << (*tgtIter).second << std::endl;
      }
    } 
  }
  
  // refactor variable names here (e.g. translations)
  template <typename ContextType>
  void PrintParams(const ConditionalMultinomialParam<ContextType>& params, const VocabEncoder &encoder) {
    // iterate over src tokens in the model
    int counter = 0;
    for(typename ConditionalMultinomialParam<ContextType>::const_iterator srcIter = params.params.begin(); srcIter != params.params.end(); srcIter++) {
      const MultinomialParam &translations = (*srcIter).second;
      // iterate over tgt tokens 
      for(MultinomialParam::const_iterator tgtIter = translations.begin(); tgtIter != translations.end(); tgtIter++) {
	std::cerr << "-logp(" << encoder.Decode((*tgtIter).first) << "|" << (*srcIter).first << ")=-log(" << nExp((*tgtIter).second) << ")=" << (*tgtIter).second << std::endl;
      }
    } 
  }

  inline void PersistParams1(const std::string &paramsFilename, 
			     const ConditionalMultinomialParam<int> &params, 
			     const VocabEncoder &vocabEncoder) {
    std::ofstream paramsFile(paramsFilename.c_str(), std::ios::out);

    for (std::map<int, MultinomialParam>::const_iterator srcIter = params.params.begin(); 
	 srcIter != params.params.end(); 
	 srcIter++) {
      for (MultinomialParam::const_iterator tgtIter = srcIter->second.begin(); tgtIter != srcIter->second.end(); tgtIter++) {
	// line format: 
	// srcTokenId tgtTokenId logP(tgtTokenId|srcTokenId) p(tgtTokenId|srcTokenId)
	paramsFile << srcIter->first << " " << vocabEncoder.Decode(tgtIter->first) << " " << tgtIter->second << " " << nExp(tgtIter->second) << std::endl;
      }
    }

    paramsFile.close();
  }

  inline void PersistParams2(const std::string &paramsFilename, 
			     const ConditionalMultinomialParam< std::pair<int, int> > &params, 
			     const VocabEncoder &vocabEncoder) {
    std::ofstream paramsFile(paramsFilename.c_str(), std::ios::out);
    for (std::map< std::pair<int, int>, MultinomialParam>::const_iterator srcIter = params.params.begin(); 
	 srcIter != params.params.end(); 
	 srcIter++) {
      for (MultinomialParam::const_iterator tgtIter = srcIter->second.begin(); tgtIter != srcIter->second.end(); tgtIter++) {
	// line format: 
	// srcTokenId tgtTokenId logP(tgtTokenId|srcTokenId) p(tgtTokenId|srcTokenId)
	paramsFile << srcIter->first.first << "->" << srcIter->first.second << " " << vocabEncoder.Decode(tgtIter->first) << " " << tgtIter->second << " " << nExp(tgtIter->second) << std::endl;
      }
    }
    paramsFile.close();

  }
  
  // sample an integer from a multinomial
  inline int SampleFromMultinomial(const MultinomialParam params) {
    // generate a pseudo random number between 0 and 1
    double randomProb = ((double) rand() / (RAND_MAX));
    
    // find the lucky value
    for(MultinomialParam::const_iterator paramIter = params.begin(); 
	paramIter != params.end(); 
	paramIter++) {
      double valueProb = nExp(paramIter->second);
      if(randomProb <= valueProb) {
	return paramIter->first;
      } else {
	randomProb -= valueProb;
      }
    }
    
    // if you get here, one of the following two things happened: \sum valueProb_i > 1 OR randomProb > 1
    assert(false);
  }
}

#endif
