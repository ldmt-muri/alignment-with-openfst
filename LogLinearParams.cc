#include "LogLinearParams.h"

using namespace std;

LogLinearParams::LogLinearParams(const VocabDecoder &srcTypes, 
				 const VocabDecoder &tgtTypes, 
				 const std::map<int, std::map<int, double> > &ibmModel1ForwardLogProbs,
				 const std::map<int, std::map<int, double> > &ibmModel1BackwardLogProbs) :
  srcTypes(srcTypes), 
  tgtTypes(tgtTypes), 
  ibmModel1ForwardScores(ibmModel1ForwardLogProbs), 
  ibmModel1BackwardScores(ibmModel1BackwardLogProbs)
{
}

LogLinearParams::LogLinearParams(const VocabDecoder &types) : 
  srcTypes(types), 
  tgtTypes(types),
  ibmModel1ForwardScores(map<int, map<int, double> >()),
  ibmModel1BackwardScores(map<int, map<int, double> >()) {
}

// if there's another parameter with the same ID already, do nothing
bool LogLinearParams::AddParam(string paramId, double paramWeight) {
  if(paramIndexes.count(paramId) == 0) {
    // check class's integrity
    assert(paramIndexes.size() == paramWeights.size());
    int newParamIndex = paramIndexes.size();
    paramIndexes[paramId] = newParamIndex;
    paramWeights.push_back(paramWeight);
    return true;
  } else {
    return false;
  }
}

// side effect: adds zero weights for parameter IDs present in values but not present in paramIndexes and paramWeights
double LogLinearParams::DotProduct(const map<string, double>& values) {
  double dotProduct = 0;
  // for each active feature
  for(map<string, double>::const_iterator valuesIter = values.begin(); valuesIter != values.end(); valuesIter++) {
    // make sure there's a corresponding feature in paramIndexes and paramWeights
    AddParam(valuesIter->first);
    // then update the dot product
    dotProduct += valuesIter->second * paramWeights[paramIndexes[valuesIter->first]];
  }
  return dotProduct;
}


// assumptions:
// -both vectors are of the same size
double LogLinearParams::DotProduct(const vector<double>& values, const vector<double>& weights) {
  assert(values.size() == weights.size());
  double dotProduct = 0;
  for(int i = 0; i < values.size(); i++) {
    dotProduct += values[i] * weights[i];
  }
  return dotProduct;
}

// compute the dot product between the values vector (passed) and the paramWeights vector (member)
// assumptions:
// - values and paramWeights are both of the same size
double LogLinearParams::DotProduct(const vector<double>& values) {
  return DotProduct(values, paramWeights);
}

double LogLinearParams::ComputeLogProb(int srcToken, int prevSrcToken, int tgtToken, int srcPos, int prevSrcPos, int tgtPos, 
				      int srcSentLength, int tgtSentLength, 
				      const std::vector<bool>& enabledFeatureTypes) {

  map<string, double> activeFeatures;
  FireFeatures(srcToken, prevSrcToken, tgtToken, srcPos, prevSrcPos, tgtPos, srcSentLength, tgtSentLength, enabledFeatureTypes, activeFeatures);
  // compute log prob
  double result = DotProduct(activeFeatures);
  
  //  cerr << "RESULT=" << result << endl << endl;

  return result;
}

void LogLinearParams::FireFeatures(int yI, int yIM1, const vector<int> &x, int i, 
				   const std::vector<bool> &enabledFeatureTypes, 
				   std::map<string, double> &activeFeatures) {

  stringstream temp;

  // F51: yIM1-yI pair
  if(enabledFeatureTypes.size() > 51 && enabledFeatureTypes[51]) {
    temp.str("");
    temp << "F51:" << yIM1 << ":" << yI;
    activeFeatures[temp.str()] = 1.0;
  }

  // F52: yI-xIM2 pair
  if(enabledFeatureTypes.size() > 52 && enabledFeatureTypes[52]) {
    temp.str("");
    int xIM2 = i-2 >= 0? x[i-2] : -1;
    temp << "F52:" << yI << ":" << xIM2;
    activeFeatures[temp.str()] = 1.0;
  }

  // F53: yI-xIM1 pair
  if(enabledFeatureTypes.size() > 53 && enabledFeatureTypes[53]) {
    temp.str("");
    int xIM1 = i-1 >= 0? x[i-1] : -1;
    temp << "F53:" << yI << ":" << xIM1;
    activeFeatures[temp.str()] = 1.0;
  }

  // F54: yI-xI pair
  if(enabledFeatureTypes.size() > 54 && enabledFeatureTypes[54]) {
    temp.str("");
    temp << "F54:" << yI << ":" << x[i];
    activeFeatures[temp.str()] = 1.0;
  }
  
  // F55: yI-xIP1 pair
  if(enabledFeatureTypes.size() > 55 && enabledFeatureTypes[55]) {
    temp.str("");
    int xIP1 = i+1 < x.size()? x[i+1] : -1;
    temp << "F55:" << yI << ":" << xIP1;
    activeFeatures[temp.str()] = 1.0;
  }

  // F56: yI-xIP2 pair
  if(enabledFeatureTypes.size() > 56 && enabledFeatureTypes[56]) {
    temp.str("");
    int xIP2 = i+2 < x.size()? x[i+2] : -1; 
   temp << "F56:" << yI << ":" << xIP2;
    activeFeatures[temp.str()] = 1.0;
  }

  // F57: yI-i pair
  if(enabledFeatureTypes.size() > 57 && enabledFeatureTypes[57]) {
    temp.str("");
    temp << "F57:" << yI << ":" << i;
    activeFeatures[temp.str()] = 1.0;
  }

  const VocabDecoder &types = srcTypes;
  const std::string& xIString = types.Decode(x[i]);
  unsigned xIStringSize = xIString.size();
  const std::string& xIM1String = i-1 >= 0?
    types.Decode(x[i-1]):
    "_start_";
  unsigned xIM1StringSize = xIM1String.size();
  const std::string& xIP1String = i+1 < x.size()?
    types.Decode(x[i+1]):
    "_end_";
  unsigned xIP1StringSize = xIP1String.size();
  // F58: yI-prefix(xI)
  if(enabledFeatureTypes.size() > 58 && enabledFeatureTypes[58]) {
    if(xIStringSize > 0) {
      temp.str("");
      temp << "F58:" << yI << ":" << xIString[0];
      activeFeatures[temp.str()] = 1.0;
    }
    if(xIStringSize > 1) {
      temp.str("");
      temp << "F58:" << yI << ":" << xIString[0] << xIString[1];
      activeFeatures[temp.str()] = 1.0;
    }
    if(xIStringSize > 2) {
      temp.str("");
      temp << "F58:" << yI << ":" << xIString[0] << xIString[1] << xIString[2];
      activeFeatures[temp.str()] = 1.0;
    }
  }

  // F59: yI-suffix(xI)
  if(enabledFeatureTypes.size() > 59 && enabledFeatureTypes[59]) {
    if(xIStringSize > 0) {
      temp.str("");
      temp << "F59:" << yI << ":" << xIString[xIStringSize-1];
      activeFeatures[temp.str()] = 1.0;
    }
    if(xIStringSize > 1) {
      temp.str("");
      temp << "F59:" << yI << ":" << xIString[xIStringSize-2] << xIString[xIStringSize-1];
      activeFeatures[temp.str()] = 1.0;
    }
    if(xIStringSize > 2) {
      temp.str("");
      temp << "F59:" << yI << ":" << xIString[xIStringSize-3] << xIString[xIStringSize-2] << xIString[xIStringSize-1];
      activeFeatures[temp.str()] = 1.0;
    }
  }

  // F60: yI-prefix(xIP1)
  if(enabledFeatureTypes.size() > 60 && enabledFeatureTypes[60]) {
    if(xIP1StringSize > 0) {
      temp.str("");
      temp << "F60:" << yI << ":" << xIP1String[0];
      activeFeatures[temp.str()] = 1.0;
    }
    if(xIP1StringSize > 1) {
      temp.str("");
      temp << "F60:" << yI << ":" << xIP1String[0] << xIP1String[1];
      activeFeatures[temp.str()] = 1.0;
    }
    if(xIP1StringSize > 2) {
      temp.str("");
      temp << "F60:" << yI << ":" << xIP1String[0] << xIP1String[1] << xIP1String[2];
      activeFeatures[temp.str()] = 1.0;
    }
  }

  // F61: yI-suffix(xIP1)
  if(enabledFeatureTypes.size() > 61 && enabledFeatureTypes[61]) {
    if(xIP1StringSize > 0) {
      temp.str("");
      temp << "F61:" << yI << ":" << xIP1String[xIP1StringSize-1];
      activeFeatures[temp.str()] = 1.0;
    }
    if(xIP1StringSize > 1) {
      temp.str("");
      temp << "F61:" << yI << ":" << xIP1String[xIP1StringSize-2] << xIP1String[xIP1StringSize-1];
      activeFeatures[temp.str()] = 1.0;
    }
    if(xIP1StringSize > 2) {
      temp.str("");
      temp << "F61:" << yI << ":" << xIP1String[xIP1StringSize-3] << xIP1String[xIP1StringSize-2] << xIP1String[xIP1StringSize-1];
      activeFeatures[temp.str()] = 1.0;
    }
  }

  // F62: yI-prefix(xIM1)
  if(enabledFeatureTypes.size() > 62 && enabledFeatureTypes[62]) {
    if(xIM1StringSize > 0) {
      temp.str("");
      temp << "F62:" << yI << ":" << xIM1String[0];
      activeFeatures[temp.str()] = 1.0;
    }
    if(xIM1StringSize > 1) {
      temp.str("");
      temp << "F62:" << yI << ":" << xIM1String[0] << xIM1String[1];
      activeFeatures[temp.str()] = 1.0;
    }
    if(xIM1StringSize > 2) {
      temp.str("");
      temp << "F62:" << yI << ":" << xIM1String[0] << xIM1String[1] << xIM1String[2];
      activeFeatures[temp.str()] = 1.0;
    }
  }

  // F63: yI-suffix(xIM1)
  if(enabledFeatureTypes.size() > 63 && enabledFeatureTypes[63]) {
    if(xIM1StringSize > 0) {
      temp.str("");
      temp << "F63:" << yI << ":" << xIM1String[xIM1StringSize-1];
      activeFeatures[temp.str()] = 1.0;
    }
    if(xIM1StringSize > 1) {
      temp.str("");
      temp << "F63:" << yI << ":" << xIM1String[xIM1StringSize-2] << xIM1String[xIM1StringSize-1];
      activeFeatures[temp.str()] = 1.0;
    }
    if(xIM1StringSize > 2) {
      temp.str("");
      temp << "F63:" << yI << ":" << xIM1String[xIM1StringSize-3] << xIM1String[xIM1StringSize-2] << xIM1String[xIM1StringSize-1];
      activeFeatures[temp.str()] = 1.0;
    }
  }

  // F64: yI-hash(xI) ==> to be implemented in StringUtils ==> converts McDonald2s ==> AaAaaaia
  if(enabledFeatureTypes.size() > 64 && enabledFeatureTypes[64]) {
    temp.str("");
    temp << "F64:" << yI << ":";
    for(int j = 0; j < xIStringSize; j++) {
      char xIStringJ = xIString[j];
      if(xIStringJ >= 'a' && xIStringJ <= 'z') {
	temp << 's'; // for small
      } else if(xIStringJ >= 'A' && xIStringJ <= 'Z') {
	temp << 'c'; // for capital
      } else if(xIStringJ >= '0' && xIStringJ <= '9') {
	temp << 'd'; // for digit
      } else if(xIStringJ == '@') {
	temp << 'a'; // for 'at'
      } else if(xIStringJ == '.') {
	temp << 'p'; // for period
      } else {
	temp << '*';
      }
    }
    activeFeatures[temp.str()] = 1.0;
  }

  // F65: yI-capitalInitial(xI)
  if(enabledFeatureTypes.size() > 65 && enabledFeatureTypes[65]) {
    temp.str("");
    temp << "F65:" << yI << ":CapitalInitial";
    activeFeatures[temp.str()] = 1.0;
  }
}

void LogLinearParams::FireFeatures(int srcToken, int prevSrcToken, int tgtToken, int srcPos, int prevSrcPos, int tgtPos, 
				   int srcSentLength, int tgtSentLength, 
				   const std::vector<bool>& enabledFeatureTypes, 
				   std::map<string, double>& activeFeatures) {
  
  // for debugging
  //    cerr << "srcToken=" << srcToken << " tgtToken=" << tgtToken << " srcPos=" << srcPos << " tgtPos=" << tgtPos;
  //    cerr << " srcSentLength=" << srcSentLength << " tgtSentLength=" << tgtSentLength << endl;

  assert(srcToken != 0 && tgtToken != 0);

  stringstream temp;

  // F1: src-tgt pair (subset of word association features in Chris et al. 2011)
  if(enabledFeatureTypes.size() > 1 && enabledFeatureTypes[1]) {
    temp << "F1:" << srcToken << "-" << tgtToken;
    activeFeatures[temp.str()] = 1.0;
  }

  // F2: diagonal-bias (positional features in Chris et al. 2011, which follows Blunsom and Cohn 2006)
  if(enabledFeatureTypes.size() > 2 && enabledFeatureTypes[2]) {
    activeFeatures["F2:diagonal-bias"] = fabs((double) srcPos / srcSentLength) - ((double) tgtPos / tgtSentLength);
  }

  // F3: src token (source features in Chris et al. 2011)
  if(enabledFeatureTypes.size() > 3 && enabledFeatureTypes[3]) {
    temp.str("");
    temp << "F3:" << srcToken;
    activeFeatures[temp.str()] = 1.0;
  }

  // F4: alignment jump distance (subset of src path features in Chris et al. 2011)
  // for debugging only
  int alignmentJumpWidth = abs(srcPos - prevSrcPos);
  int discretizedAlignmentJumpWidth = (int) (log(alignmentJumpWidth) / log(1.3));
  if(enabledFeatureTypes.size() > 4 && enabledFeatureTypes[4]) {
    temp.str("");
    temp << "F4:" << discretizedAlignmentJumpWidth;
    activeFeatures[temp.str()] = 1.0;
  }

  // F5: alignment jump direction (subset of src path features in Chris et al. 2011)
  int alignmentJumpDirection = srcPos > prevSrcPos? +1 : srcPos < prevSrcPos? -1 : 0;
  if(enabledFeatureTypes.size() > 5 && enabledFeatureTypes[5]) {
    temp.str("");
    temp << "F5:" << alignmentJumpDirection;
    activeFeatures[temp.str()] = 1.0;
  }
 
  // F6: alignment jump from/to (subset of src path features in Chris et al. 2011)
  if(enabledFeatureTypes.size() > 6 && enabledFeatureTypes[6]) {
    temp.str("");
    temp << "F6:" << prevSrcPos << ":" << srcPos;
    activeFeatures[temp.str()] = 1.0;
  }

  // F7: orthographic similarity (subset of word association features in Chris et al. 2011)
  const std::string& srcTokenString = srcTypes.Decode(srcToken);
  const std::string& tgtTokenString = tgtTypes.Decode(tgtToken);
  //  cerr << "computing ortho-similarity(" << srcToken << " (" << srcTokenString << ") " << ", " << tgtToken << " (" << tgtTokenString << ") )" << endl;
  //  double orthographicSimilarity = ComputeOrthographicSimilarity(srcTokenString, tgtTokenString);
  //  if(enabledFeatureTypes.size() > 7 && enabledFeatureTypes[7]) {
  //    activeFeatures["F7:orthographic-similarity"] = orthographicSimilarity;
  //  }

  // F12: ibm model 1 forward logprob (subset of word association features in Chris et al. 2011)
  double ibm1Forward = ibmModel1ForwardScores.find(srcToken)->second.find(tgtToken)->second;
  if(enabledFeatureTypes.size() > 12 && enabledFeatureTypes[12]) {
    temp.str("");
    temp << "F12:" << srcToken << ":" << tgtToken;
    activeFeatures[temp.str()] = ibm1Forward;
  }

  // F13: ibm model 1 backward logprob (subset of word association features in Chris et al. 2011)
  double ibm1Backward = ibmModel1BackwardScores.find(tgtToken)->second.find(srcToken)->second;
  if(enabledFeatureTypes.size() > 13 && enabledFeatureTypes[13]) {
    temp.str("");
    temp << "F13:" << srcToken << ":" << tgtToken;
    activeFeatures[temp.str()] = ibm1Backward;
  }

  // F14: log of the geometric mean of ibm model 1 forward/backward prob (subset of word association features in Chris et al. 2011)
  if(enabledFeatureTypes.size() > 14 && enabledFeatureTypes[14]) {
    temp.str("");
    temp << "F14:" << srcToken << ":" << tgtToken;
    activeFeatures[temp.str()] = 0.5 * (ibm1Forward + ibm1Backward);
  }

  // F15: discretized Dice's coefficient (subset of word association features in Chris et al. 2011)
  // TODO

  // F16: word cluster associations (e.g. to encode things like nouns tend to translate as nouns) (subset of word association features in Chris et al. 2011)
  // TODO

  // F17: <F2:diagonal_bias, srcWordClassType>  (subset of positional features in Chris et al. 2011)
  // TODO

  // F18: srcWordClassType (subset of source features in Chris et al. 2011)
  // TODO

  // F19: alignment jump direction AND (discretized) width (subset of src path features in Chris et al. 2011)
  if(enabledFeatureTypes.size() > 19 && enabledFeatureTypes[19]) {
    temp.str("");
    temp << "F19:" << alignmentJumpDirection * discretizedAlignmentJumpWidth;
    activeFeatures[temp.str()] = 1.0;
  }

  // F20: <discretized alignment jump, tgtLength> (subset of src path features in Chris et al. 2011)
  if(enabledFeatureTypes.size() > 20 && enabledFeatureTypes[20]) {
    temp.str("");
    temp << "F20:" << alignmentJumpDirection * discretizedAlignmentJumpWidth << ":" << tgtSentLength;
    activeFeatures[temp.str()] = 1.0;
  }

  // F21: <discretized alignment jump, class of srcToken> (subset of src path features in Chris et al. 2011)
  // TODO

  // F22: <discretized alignment jump, class of srcToken, class of prevSrcToken> (subset of src path features in Chris et al. 2011)
  // TODO

  // F23: <srcToken, prevSrcToken> (subset of src path features in Chris et al. 2011)
  if(enabledFeatureTypes.size() > 23 && enabledFeatureTypes[23]) {
    temp.str("");
    temp << "F23:" << srcToken << ":" << prevSrcToken;
    activeFeatures[temp.str()] = 1.0;
  }

  // F24: is this a named entity translated twice? (subset of tgt string features in Chris et al. 2011)
  // note: in this implementation, the feature fires when the current translation is ortho-similar, and a[i] == a[i-1].
  //       ideally, it should also fire when the current translation is orth-similar, and a[i] == a[i+1]
  //  if(enabledFeatureTypes.size() > 24 && enabledFeatureTypes[24]) {
  //    if(orthographicSimilarity > 0 && prevSrcPos == srcPos) {
  //      temp.str("");
  //      temp << "F24:repeated-NE";
  //      activeFeatures[temp.str()] = orthographicSimilarity;
  //    }
  //}
}

// each line consists of: <featureStringId><space><featureWeight>\n
void LogLinearParams::PersistParams(const string& outputFilename) {
  ofstream paramsFile(outputFilename.c_str());
  
  for (map<string, int>::const_iterator paramsIter = paramIndexes.begin(); paramsIter != paramIndexes.end(); paramsIter++) {
    paramsFile << paramsIter->first << " " << paramWeights[paramsIter->second] << endl;
  }

  paramsFile.close();
}

void LogLinearParams::PrintFirstNParams(unsigned n) {
  for (map<string, int>::const_iterator paramsIter = paramIndexes.begin(); n-- > 0 && paramsIter != paramIndexes.end(); paramsIter++) {
    cerr << paramsIter->first << " " << paramWeights[paramsIter->second] << endl;
  }
}

// each line consists of: <featureStringId><space><featureWeight>\n
void LogLinearParams::PrintParams() {
  assert(paramIndexes.size() == paramWeights.size());
  PrintFirstNParams(paramIndexes.size());
}

void LogLinearParams::PrintParams(std::map<std::string, double> tempParams) {
  for(map<string, double>::const_iterator paramsIter = tempParams.begin(); paramsIter != tempParams.end(); paramsIter++) {
    cerr << paramsIter->first << " " << paramsIter->second << endl;
  }
}


// use gradient based methods to update the model parameter weights
void LogLinearParams::UpdateParams(const map<string, double> &gradient, const OptMethod& optMethod) {
  switch(optMethod.algorithm) {
  case OptAlgorithm::GRADIENT_DESCENT:
    for(map<string, double>::const_iterator gradientIter = gradient.begin();
	gradientIter != gradient.end();
	gradientIter++) {
      // in case this parameter does not exist in paramWeights/paramIndexes
      AddParam(gradientIter->first);
      // update the parameter weight
      paramWeights[ paramIndexes[gradientIter->first] ] -= optMethod.learningRate * gradientIter->second;
    }
    break;
  default:
    assert(false);
    break;
  }
}

// override the member weights vector with this array
void LogLinearParams::UpdateParams(const double* array, const int arrayLength) {
  assert(arrayLength == paramWeights.size());
  assert(paramWeights.size() == paramIndexes.size());
  for(int i = 0; i < arrayLength; i++) {
    paramWeights[i] = array[i];
  }
}

// TODO: reimplement l1 penalty
// if using a cumulative L1 regularizer, apply the cumulative l1 penalty
void LogLinearParams::ApplyCumulativeL1Penalty(const LogLinearParams& applyToFeaturesHere,
					       LogLinearParams& appliedL1Penalty,
					       const double correctL1Penalty) {
  assert(false);
  /*  for(map<string, double>::const_iterator featuresIter = applyToFeaturesHere.params.begin();
      featuresIter != applyToFeaturesHere.params.end();
      featuresIter++) {
    double currentFeatureWeight = params[featuresIter->first];
    if(currentFeatureWeight >= 0) {
      currentFeatureWeight = max(0.0, currentFeatureWeight - (correctL1Penalty + appliedL1Penalty.params[featuresIter->first]));
    } else {
      currentFeatureWeight = min(0.0, currentFeatureWeight + (correctL1Penalty - appliedL1Penalty.params[featuresIter->first]));
    }
    appliedL1Penalty.params[featuresIter->first] += currentFeatureWeight - params[featuresIter->first];
    params[featuresIter->first] = currentFeatureWeight;
    }*/
}
