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
 private:
  std::map< std::string, int > paramsStringIdToIntId;
  std::vector<double> paramsValueArray;
  std::vector<double> gradientValueArray;
  std::vector<std::string> paramsStringIdArray;

 public:

  LogLinearParams(const VocabDecoder &srcTypes, 
		  const VocabDecoder &tgtTypes, 
		  const std::map<int, std::map<int, double> > &ibmModel1ForwardLogProbs,
		  const std::map<int, std::map<int, double> > &ibmModel1BackwardLogProbs);

  LogLinearParams(const VocabDecoder &types);

  // given the description of one transition on the alignment FST, find the features that would fire along with their values
  // note: pos here is short for position
  void FireFeatures(int srcToken, int prevSrcToken, int tgtToken, 
		    int srcPos, int prevSrcPos, int tgtPos, 
		    int srcSentLength, int tgtSentLength, 
		    const std::vector<bool>& enabledFeatureTypes, std::map<std::string, double>& activeFeatures);

  void FireFeatures(int yI, int yIM1, const vector<int> &x, int i, 
		    const std::vector<bool> &enabledFeatureTypes, 
		    std::map<string, double> &activeFeatures);
    
  // compute dot product of two sparse vectors, each represented with a map. 
  double DotProduct(const std::map<std::string, double>& values, const std::map<std::string, double>& weights);

  // compute dot product of feature values (passed), and feature weights (member variable 'params')
  double DotProduct(const std::map<std::string, double>& values);

  // given description of an arc in the alignment transducer, compute the local arc probability
  double ComputeLogProb(int srcToken, int prevSrcToken, int tgtToken, int srcPos, int prevSrcPos, int tgtPos, int srcSentLength, int tgtSentLength,
		       const std::vector<bool>& enabledFeatureTypes);

  // updates the model parameters given the gradient and an optimization method
  void UpdateParams(const map<string, double> &gradient, const OptUtils::OptMethod &optMethod);

  // use gradient based methods to update the model parameter weights
  void UpdateParams(const LogLinearParams &gradient, const OptUtils::OptMethod &optMethod);

  // applies the accumulative l1 penalty on feature weights, also updates the appliedL1Penalty values
  void ApplyCumulativeL1Penalty(const LogLinearParams& applyToFeaturesHere,
				LogLinearParams& appliedL1Penalty,
				const double correctL1Penalty);

  // compute the orthographic similarity between two strings
  double ComputeOrthographicSimilarity(const std::string& srcWord, const std::string& tgtWord);

  // levenshtein distance
  int LevenshteinDistance(const std::string& x, const std::string& y);

  // clear model parameters
  inline void Clear() {
    params.clear();
  }

  // when the lbfgs minimizer updates the parameter weights array, this method is called to reflect the updates
  // on the map
  void UpdateParams(const double* array, const int arrayLength) {
    for(int i = 0; i < arrayLength; i++) {
      params[ paramsStringIdArray[i] ] = paramsValueArray[i];
    }
  }

  // TODO: refactor into something more generally useful
  // lbfgs requires the evaluate callback function to return the gradient as an array. we compute it in a map<string,double>. 
  // this function rewrites the gradient in an array and "return" it.
  // assumptions:
  // - array is preallocated to size equals the size of paramsStringIdArray 
  void MapToArray(const std::map<std::string, double>& gradient, double* array) {
    // for each feature
    for(int i = 0; i < paramsStringIdArray.size(); i++) {
      // if the the gradient for this fetaure is zero, set the corresponding value in the array to zero
      if(gradient.count(paramsStringIdArray[i]) == 0) {
	array[i] = 0;
      } else {
	array[i] = gradient.find(paramsStringIdArray[i])->second;
      }
    }
  }

  // updates the array represetntaion of the features and their values, and sets the pointer to the array and its length
  void UpdateArray(double** array, int* arrayLength) {
    for(std::map< std::string, double>::const_iterator paramsIter = params.begin();
	paramsIter != params.end();
	paramsIter++) {
      // add features not already present to the paramsValueArray
      if(paramsStringIdToIntId.count(paramsIter->first) == 0) {
	// set this new feature's integer id in paramsStringIdArray (which is the same as its integer id in paramsValueArray)
	paramsStringIdToIntId[paramsIter->first] = paramsValueArray.size();
	// add an entry for this new feature in the string id array
	paramsStringIdArray.push_back(paramsIter->first);
	// make room for the new feature in the values array, and set its value
	paramsValueArray.push_back(paramsIter->second);
      } else {
	// set this feature's value in paramsValueArray
	paramsValueArray[paramsStringIdToIntId[paramsIter->first]] = paramsIter->second;
      }
    }
    // report back to the caller
    *array = paramsValueArray.data();
    *arrayLength = paramsValueArray.size();
  }

  // writes the features to a text file formatted one feature per line. 
  void PersistParams(const std::string& outputFilename);
  
  std::map< std::string, double > params;

  // maps a word id into a string
  const VocabDecoder &srcTypes, &tgtTypes;

  // maps [srcTokenId][tgtTokenId] => forward logprob
  // maps [tgtTokenId][srcTokenId] => backward logprob
  const std::map< int, std::map< int, double > > &ibmModel1ForwardScores, &ibmModel1BackwardScores;
};

#endif
