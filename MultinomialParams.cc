#include "MultinomialParams.h"

using namespace MultinomialParams;

  // refactor variable names here (e.g. translations)
  // normalizes ConditionalMultinomialParam parameters such that \sum_t p(t|s) = 1 \forall s
void MultinomialParams::NormalizeParams(ConditionalMultinomialParam& params) {
    // iterate over src tokens in the model
    for(ConditionalMultinomialParam::iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
      std::map< int, float > &translations = (*srcIter).second;
      float fTotalProb = 0.0;
      // iterate over tgt tokens logsumming over the logprob(tgt|src) 
      for(std::map< int, float >::iterator tgtIter = translations.begin(); tgtIter != translations.end(); tgtIter++) {
	float temp = (*tgtIter).second;
	fTotalProb += nExp(temp);
      }
      // exponentiate to find p(*|src) before normalization
      // iterate again over tgt tokens dividing p(tgt|src) by p(*|src)
      float fVerifyTotalProb = 0.0;
      for(std::map< int, float >::iterator tgtIter = translations.begin(); tgtIter != translations.end(); tgtIter++) {
	float fUnnormalized = nExp((*tgtIter).second);
	float fNormalized = fUnnormalized / fTotalProb;
	fVerifyTotalProb += fNormalized;
	float fLogNormalized = nLog(fNormalized);
	(*tgtIter).second = fLogNormalized;
      }
    }
  }
  
  // zero all parameters
void MultinomialParams::ClearParams(ConditionalMultinomialParam& params) {
    for (ConditionalMultinomialParam::iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
      for (std::map<int, float>::iterator tgtIter = srcIter->second.begin(); tgtIter != srcIter->second.end(); tgtIter++) {
	tgtIter->second = NLOG_ZERO;
      }
    }
  }
  
  // refactor variable names here (e.g. translations)
void MultinomialParams::PrintParams(const ConditionalMultinomialParam& params) {
    // iterate over src tokens in the model
    int counter = 0;
    for(ConditionalMultinomialParam::const_iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
      const std::map< int, float > &translations = (*srcIter).second;
      // iterate over tgt tokens 
      for(std::map< int, float >::const_iterator tgtIter = translations.begin(); tgtIter != translations.end(); tgtIter++) {
	std::cerr << "-logp(" << (*tgtIter).first << "|" << (*srcIter).first << ")=-log(" << nExp((*tgtIter).second) << ")=" << (*tgtIter).second << std::endl;
      }
    } 
  }
  
  // refactor variable names here (e.g. translations)
void MultinomialParams::PrintParams(const ConditionalMultinomialParam& params, const VocabEncoder &encoder) {
    // iterate over src tokens in the model
    int counter = 0;
    for(ConditionalMultinomialParam::const_iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
      const std::map< int, float > &translations = (*srcIter).second;
      // iterate over tgt tokens 
      for(std::map< int, float >::const_iterator tgtIter = translations.begin(); tgtIter != translations.end(); tgtIter++) {
	std::cerr << "-logp(" << encoder.Decode((*tgtIter).first) << "|" << (*srcIter).first << ")=-log(" << nExp((*tgtIter).second) << ")=" << (*tgtIter).second << std::endl;
      }
    } 
  }
  

void MultinomialParams::PersistParams(std::ofstream& paramsFile, const ConditionalMultinomialParam& params) {
  for (ConditionalMultinomialParam::const_iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
    for (std::map<int, float>::const_iterator tgtIter = srcIter->second.begin(); tgtIter != srcIter->second.end(); tgtIter++) {
      // line format: 
      // srcTokenId tgtTokenId logP(tgtTokenId|srcTokenId) p(tgtTokenId|srcTokenId)
      paramsFile << srcIter->first << " " << tgtIter->first << " " << tgtIter->second << " " << nExp(tgtIter->second) << std::endl;
    }
  }
}

void MultinomialParams::PersistParams(std::ofstream &paramsFile, const ConditionalMultinomialParam &params, const VocabEncoder &vocabEncoder) {
  for (ConditionalMultinomialParam::const_iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
    for (std::map<int, float>::const_iterator tgtIter = srcIter->second.begin(); tgtIter != srcIter->second.end(); tgtIter++) {
      // line format: 
      // srcTokenId tgtTokenId logP(tgtTokenId|srcTokenId) p(tgtTokenId|srcTokenId)
      paramsFile << srcIter->first << " " << vocabEncoder.Decode(tgtIter->first) << " " << tgtIter->second << " " << nExp(tgtIter->second) << std::endl;
    }
  }
}

void MultinomialParams::PersistParams(const std::string& paramsFilename, const ConditionalMultinomialParam& params) {
  std::ofstream paramsFile(paramsFilename.c_str(), std::ios::out);
  PersistParams(paramsFile, params);
  paramsFile.close();
}

void MultinomialParams::PersistParams(const std::string& paramsFilename, const ConditionalMultinomialParam& params, const VocabEncoder &vocabEncoder) {
  std::ofstream paramsFile(paramsFilename.c_str(), std::ios::out);
  PersistParams(paramsFile, params, vocabEncoder);
  paramsFile.close();
}

  // sample an integer from a multinomial
int MultinomialParams::SampleFromMultinomial(const MultinomialParam params) {
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
