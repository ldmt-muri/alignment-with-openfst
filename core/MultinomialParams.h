#ifndef _MULTINOMIAL_PARAMS_H_
#define _MULTINOMIAL_PARAMS_H_

#include <boost/math/special_functions/digamma.hpp>
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

#include <boost/iterator.hpp>
#include <boost/unordered_map.hpp>

#include "../wammar-utils/unordered_map_serialization.hpp"

#include "../wammar-utils/Samplers.h"
#include "VocabEncoder.h"
#include "../wammar-utils/FstUtils.h"

namespace MultinomialParams {

  // parameters for describing a multinomial distribution p(x)=y such that x is the key and y is a log probability
  typedef boost::unordered_map<int64_t, double> MultinomialParam;

  // forward declarations
  double nLog(double prob);
  double nExp(double prob);
  
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

    double Hash() {
      double hash = 0.0;
      for(typename boost::unordered_map<ContextType, MultinomialParam>::const_iterator cIter = params.begin(); cIter != params.end(); cIter++) {
	for(MultinomialParam::const_iterator mIter = cIter->second.begin(); mIter != cIter->second.end(); mIter++) {
	  hash += mIter->second;
	}
      }
      return hash;
    }
    
    void GaussianInit(double mean = 0.0, double std = 1.0) {
      GaussianSampler sampler(mean, std);
      for(typename boost::unordered_map<ContextType, MultinomialParam>::iterator cIter = params.begin(); cIter != params.end(); cIter++) {
        for(MultinomialParam::iterator mIter = cIter->second.begin(); mIter != cIter->second.end(); mIter++) {
          mIter->second = nExp(sampler.Draw());
        }
      }
      NormalizeParams(*this, 1.0, false, true);
    }
    
    // refactor variable names here (e.g. translations)
    void PrintParams() {
      // iterate over src tokens in the model
      for(typename boost::unordered_map<ContextType, MultinomialParam>::const_iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
        const MultinomialParam &translations = (*srcIter).second;
        // iterate over tgt tokens 
        for(MultinomialParam::const_iterator tgtIter = translations.begin(); tgtIter != translations.end(); tgtIter++) {
          std::cerr << "-logp(" << (*tgtIter).first << "|" << (*srcIter).first << ")=-log(" << nExp((*tgtIter).second) << ")=" << (*tgtIter).second << std::endl;
        }
      } 
    }
    
    // refactor variable names here (e.g. translations)
    void PrintParams(const VocabEncoder &encoder, bool decodeContext=true, bool decodeDecision=false) {
      // iterate over src tokens in the model
      for(typename boost::unordered_map<ContextType, MultinomialParam>::const_iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
        const MultinomialParam &translations = (*srcIter).second;
        // iterate over tgt tokens 
        for(MultinomialParam::const_iterator tgtIter = translations.begin(); tgtIter != translations.end(); tgtIter++) {
          std::cerr << "-logp(";
          if(decodeDecision) {
            cerr << encoder.Decode((*tgtIter).first); 
          } else {
            cerr << tgtIter->first;
          }
          cerr << "|";
          if(decodeContext) {
            cerr << encoder.Decode((*srcIter).first);
          } else {
            cerr << srcIter->first;
          }
          cerr << ")=-log(" << nExp((*tgtIter).second) << ")=" << (*tgtIter).second << std::endl;
        }
      } 
    }
    
  public:
    boost::unordered_map<ContextType, MultinomialParam> params;
  };
  
  static const int NLOG_SMOOTHING_CONSTANT = 1;
  static const int NLOG_ZERO = 300;
  static const int NLOG_INF = -200;

  // refactor variable names here (e.g. translations)
  // normalizes ConditionalMultinomialParam parameters such that \sum_t p(t|s) = 1 \forall s
  // for smoothing, use alpha > 1.0
  // for sparsity, use alpha < 1.0
  // for MLE, use alpha = 1.0
  template <typename ContextType>
    void NormalizeParams(ConditionalMultinomialParam<ContextType> &params, 
                         double symDirichletAlpha = 1.0, bool unnormalizedParamsAreInNLog = true,
                         bool normalizedParamsAreInNLog = true, bool useVariationalInference = false) {
    assert(symDirichletAlpha >= 1.0 || useVariationalInference); // for smaller values, we should use variational bayes
    // iterate over src tokens in the model
    for(auto srcIter = params.params.begin(); srcIter != params.params.end(); srcIter++) {
      MultinomialParam &translations = (*srcIter).second;
      double fTotalProb = 0.0;
      // iterate over tgt tokens logsumming over the logprob(tgt|src) 
      for(MultinomialParam::iterator tgtIter = translations.begin(); tgtIter != translations.end(); tgtIter++) {
        // MAP inference with dirichlet prior
        double temp = unnormalizedParamsAreInNLog? nExp((*tgtIter).second) : (*tgtIter).second;
        fTotalProb += useVariationalInference?
          temp + symDirichletAlpha :
          temp + symDirichletAlpha - 1;
      }
      // fix fTotalProb
      if(fTotalProb == 0.0 && !useVariationalInference){
        fTotalProb = 1.0;
      } else if (useVariationalInference) {
        fTotalProb = exp( boost::math::digamma(fTotalProb) );
      }
      // exponentiate to find p(*|src) before normalization
      // iterate again over tgt tokens dividing p(tgt|src) by p(*|src)
      for(MultinomialParam::iterator tgtIter = translations.begin(); tgtIter != translations.end(); tgtIter++) {
        // MAP inference with dirichlet prior
        double temp = unnormalizedParamsAreInNLog? nExp((*tgtIter).second) : (*tgtIter).second;
        double fUnnormalized = useVariationalInference?
          exp( boost::math::digamma( temp + symDirichletAlpha) ) :
          temp + symDirichletAlpha - 1;
        double fNormalized = fUnnormalized / fTotalProb;
        (*tgtIter).second = normalizedParamsAreInNLog? nLog(fNormalized) : fNormalized;
      }
    }
  }

  // line format:
  // event context logP(event|context)
  // event and/or context can be an integer or a string
  inline void PersistParams(const std::string &paramsFilename, 
			    const ConditionalMultinomialParam<int64_t> &params, 
			    const VocabEncoder &vocabEncoder,
			    bool decodeContext=false,
			    bool decodeEvent=false) {
    cerr << "persisting multinomial parameters in " << paramsFilename << endl;
    std::ofstream paramsFile(paramsFilename.c_str(), std::ios::out);
    for (boost::unordered_map<int64_t, MultinomialParam>::const_iterator srcIter = params.params.begin(); 
         srcIter != params.params.end(); 
         srcIter++) {
      for (MultinomialParam::const_iterator tgtIter = srcIter->second.begin(); tgtIter != srcIter->second.end(); tgtIter++) {
        // write context
        if(decodeContext) {
          paramsFile << vocabEncoder.Decode(srcIter->first) << " ";
        } else {
          paramsFile << srcIter->first << " ";
        }
        // write event
        if(decodeEvent) {
          paramsFile << vocabEncoder.Decode(tgtIter->first) << " ";
        } else {
          paramsFile << tgtIter->first << " ";
        }
        // print logprob
        paramsFile << -tgtIter->second << std::endl;
      }
    }
    
    paramsFile.close();
  }

  // line format:
  // event context nlogP(event|context)
  // event and/or context can be an integer or a string
  inline void LoadParams(const std::string &paramsFilename,
                         ConditionalMultinomialParam<int64_t> &params,
                         const VocabEncoder &vocabEncoder,
                         bool encodeContext=false,
                         bool encodeEvent=false) {
    
    // set all parameters to zeros
    for ( auto & context : params.params) {
      for (auto & decision : context.second) {
        decision.second = NLOG_ZERO;
      }
    }
    
    std::ifstream paramsFile(paramsFilename.c_str(), std::ios::in);
    string line;
    while( getline(paramsFile, line) ) {
      if(line.size() == 0) {
        continue;
      }
      std::vector<string> splits;
      StringUtils::SplitString(line, ' ', splits);
      // check format
      if(splits.size() != 3) {
        assert(false);
        exit(1);
      }
      // nlogp
      stringstream nlogPString;
      nlogPString << splits[2];
      double nlogP;
      nlogPString >> nlogP;
      nlogP *= -1.0;
      assert(nlogP >= 0);
      // event
      int64_t event = -1;
      if(encodeEvent) {
        event = vocabEncoder.ConstEncode(splits[1]);
      } else {
        stringstream ss;
        ss << splits[1];
        ss >> event;
      }
      assert(event >= 0);
      if(event == vocabEncoder.UnkInt()) { 
        //cerr << "WARNING: event '" << splits[1] << "' is == vocabEncoder.UnkInt()" << endl;
        continue; 
      }
      // context
      int64_t context = -1;
      if(encodeContext) {
        context = vocabEncoder.ConstEncode(splits[0]);
      } else {
        stringstream ss;
        ss << splits[0];
        ss >> context;
      }
      assert(context >= 0);
      if(context == vocabEncoder.UnkInt()) { 
        //cerr << "WARNING: context '" << splits[0] << "' is == vocabEncoder.UnkInt()" << endl;
        continue; 
      }
      // skip irrelevant parameters
      if(params.params.find(context) == params.params.end() || params[context].find(event) == params[context].end()) {
        continue;
      }
      params[context][event] = nlogP;
      if(params[context].find(-event) != params[context].end()) {
        // also add the negative event 
        params[context][-event] = nlogP;
      }
    }
    paramsFile.close();
    // renormalize
    NormalizeParams(params, 1.0, true, true, false);
    
    // persist the parameters we just loaded
    //PersistParams(paramsFilename + ".reloaded", params, vocabEncoder, true, true);
  }
  
  template <typename ContextType>
  static boost::unordered_map<ContextType, double> AccumulateMultinomials(const boost::unordered_map<ContextType, double>& p1, 
							      const boost::unordered_map<ContextType, double>& p2) {
    boost::unordered_map<ContextType, double> pTotal(p1);
    for(typename boost::unordered_map<ContextType, double>::const_iterator p2Iter = p2.begin(); p2Iter != p2.end(); p2Iter++) {
      pTotal[p2Iter->first] += p2Iter->second;
    }
    return pTotal;
  }

  template <typename ContextType>
  static boost::unordered_map<ContextType, MultinomialParam> AccumulateConditionalMultinomials(const boost::unordered_map<ContextType, MultinomialParam>& p1,
										    const boost::unordered_map<ContextType, MultinomialParam>& p2) {
    boost::unordered_map<ContextType, MultinomialParam> pTotal(p1);
    for(typename boost::unordered_map<ContextType, MultinomialParam>::const_iterator p2Iter = p2.begin(); p2Iter != p2.end(); p2Iter++) {
      for(MultinomialParam::const_iterator subP2Iter = p2Iter->second.begin(); subP2Iter != p2Iter->second.end(); subP2Iter++) {
	pTotal[p2Iter->first][subP2Iter->first] += subP2Iter->second;
      }
    }
    return pTotal;
  }

  template <typename ContextType>
  static boost::unordered_map<ContextType, MultinomialParam> AccumulateConditionalMultinomialsLogSpace(const boost::unordered_map<ContextType, MultinomialParam>& p1,
											   const boost::unordered_map<ContextType, MultinomialParam>& p2) {
    boost::unordered_map<ContextType, MultinomialParam> pTotal(p1);
    for(typename boost::unordered_map<ContextType, MultinomialParam>::const_iterator p2Iter = p2.begin(); p2Iter != p2.end(); p2Iter++) {
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
      return NLOG_ZERO;
    }
    return -1.0 * log(prob);
  }
  
  inline double nExp(double exponent) {
    if(exponent <= NLOG_INF) {
      exponent = NLOG_INF;
    }
    return exp(-1.0 * exponent);
  }
  
  
  // zero all parameters
  template <typename ContextType>
    void ClearParams(ConditionalMultinomialParam<ContextType>& params, bool smooth=false) {
    for (typename unordered_map<ContextType, MultinomialParam>::iterator srcIter = params.params.begin(); srcIter != params.params.end(); srcIter++) {
      for (MultinomialParam::iterator tgtIter = srcIter->second.begin(); tgtIter != srcIter->second.end(); tgtIter++) {
	if(smooth) {
	  tgtIter->second = NLOG_SMOOTHING_CONSTANT;
	} else {
	  tgtIter->second = NLOG_ZERO;
	}
      }
    }
  }
  
  // sample an integer from a multinomial
  inline int64_t SampleFromMultinomial(const MultinomialParam params) {
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
