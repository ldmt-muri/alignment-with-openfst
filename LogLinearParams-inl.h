#ifndef _LOG_LINEAR_PARAMS_INL_H_
#define _LOG_LINEAR_PARAMS_INL_H_

public:
  // given description of an arc in the alignment transducer, compute the local arc probability
  inline double ComputeLogProb(int srcToken, int prevSrcToken, int tgtToken, int srcPos, int prevSrcPos, int tgtPos, 
					 int srcSentLength, int tgtSentLength) {
    unordered_map_featureId_double activeFeatures;
    FireFeatures(srcToken, prevSrcToken, tgtToken, srcPos, prevSrcPos, tgtPos, srcSentLength, tgtSentLength, activeFeatures);
    // compute log prob
    double result = DotProduct(activeFeatures);    
    return result;
  }
  
  // update a single parameter's value (adds the parameter if necessary)
  inline void UpdateParam(const FeatureId &paramId, const double newValue) {
    assert(sealed);
    if(!AddParam(paramId, newValue)) {
      (*paramWeightsPtr)[paramIndexes[paramId]] = newValue;
    }
  }

  // update a single parameter's value, identified by its integer index in the weights array
  // assumptions:
  // - there's a parameter with such index
  inline void UpdateParam(const unsigned paramIndex, const double newValue) {
    assert(sealed);
    if(paramWeightsPtr->size() <= paramIndex) {
      assert(false);
    }
    (*paramWeightsPtr)[paramIndex] = newValue;
  }

  // checks whether a parameter exists
  inline bool ParamExists(const FeatureId &paramId) {
    return paramIndexes.count(paramId) == 1;
  }

  // checks whether a parameter exists
  inline bool ParamExists(const unsigned &paramIndex) {
    return paramIndex < paramIndexes.size();
  }

  // returns the int index of the parameter in the underlying array
  inline unsigned GetParamIndex(const FeatureId &paramId) {
    AddParam(paramId);
    return paramIndexes[paramId];
  }

  // returns the string identifier of the parameter given its int index in the weights array
  inline FeatureId GetParamId(const unsigned paramIndex) {
    assert(sealed);
    assert(paramIndex < paramWeightsPtr->size());
    return (*paramIdsPtr)[paramIndex];
  }

  // returns the current weight of this param (adds the parameter if necessary)
  inline double GetParamWeight(const FeatureId &paramId) {
    assert(sealed);
    AddParam(paramId);
    return (*paramWeightsPtr)[paramIndexes[paramId]];
  }
  
  // returns the current weight of this param (adds the parameter if necessary)
  inline double GetParamWeight(unsigned &paramIndex) {
    assert(sealed);
    assert(paramIndex < paramWeightsPtr->size());
    return (*paramWeightsPtr)[paramIndex];
  }
  
  inline int GetParamsCount() {
    if(sealed) {
      assert(paramWeightsPtr->size() == paramIdsPtr->size());
      assert(paramWeightsPtr->size() == paramIndexes.size());
    } else {
      assert(paramWeightsTemp.size() == paramIdsTemp.size());
      assert(paramWeightsTemp.size() == paramIndexes.size());
    }
    return paramIndexes.size();
  }

  // returns a pointer to the array of parameter weights
  inline double* GetParamWeightsArray() { 
     assert(sealed);
     return paramWeightsPtr->data(); 
   } 

#endif
