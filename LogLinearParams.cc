#include "LogLinearParams.h"

using namespace std;
using namespace OptUtils;

LogLinearParams::LogLinearParams(const VocabDecoder &srcTypes, 
				 const VocabDecoder &tgtTypes, 
				 const std::map<int, std::map<int, float> > &ibmModel1ForwardLogProbs,
				 const std::map<int, std::map<int, float> > &ibmModel1BackwardLogProbs) :
  srcTypes(srcTypes), 
  tgtTypes(tgtTypes), 
  ibmModel1ForwardScores(ibmModel1ForwardLogProbs), 
  ibmModel1BackwardScores(ibmModel1BackwardLogProbs)
{
}

LogLinearParams::LogLinearParams(const VocabDecoder &types) : 
  srcTypes(types), 
  tgtTypes(types),
  ibmModel1ForwardScores(map<int, map<int, float> >()),
  ibmModel1BackwardScores(map<int, map<int, float> >())  
{
}

float LogLinearParams::DotProduct(const map<string, float>& values) {
  return DotProduct(values, params);
}


float LogLinearParams::DotProduct(const map<string, float>& values, const map<string, float>& weights) {
  // for effeciency
  if(values.size() > weights.size()) {
    return DotProduct(weights, values);
  }

  float dotProduct = 0;
  for (map<string, float>::const_iterator valuesIter = values.begin(); valuesIter != values.end(); valuesIter++) {
    float weight = 0;
    map<string, float>::const_iterator weightIter = weights.find(valuesIter->first);
    if (weightIter != weights.end()) {
      weight = weightIter->second;
    }
    dotProduct += valuesIter->second * weight;
  }
  return dotProduct;
}

float LogLinearParams::ComputeLogProb(int srcToken, int prevSrcToken, int tgtToken, int srcPos, int prevSrcPos, int tgtPos, 
				      int srcSentLength, int tgtSentLength, 
				      const std::vector<bool>& enabledFeatureTypes) {

  map<string, float> activeFeatures;
  FireFeatures(srcToken, prevSrcToken, tgtToken, srcPos, prevSrcPos, tgtPos, srcSentLength, tgtSentLength, enabledFeatureTypes, activeFeatures);
  // compute log prob
  float result = DotProduct(activeFeatures, params);
  
    //  cerr << "RESULT=" << result << endl << endl;

  return result;
}

void LogLinearParams::FireFeatures(int yI, int yIM1, const vector<int> &x, int i, 
				   const std::vector<bool> &enabledFeatureTypes, 
				   std::map<string, float> &activeFeatures) {

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

}


void LogLinearParams::FireFeatures(int srcToken, int prevSrcToken, int tgtToken, int srcPos, int prevSrcPos, int tgtPos, 
				   int srcSentLength, int tgtSentLength, 
				   const std::vector<bool>& enabledFeatureTypes, 
				   std::map<string, float>& activeFeatures) {
  
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
    activeFeatures["F2:diagonal-bias"] = fabs((float) srcPos / srcSentLength) - ((float) tgtPos / tgtSentLength);
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
  //  float orthographicSimilarity = ComputeOrthographicSimilarity(srcTokenString, tgtTokenString);
  //  if(enabledFeatureTypes.size() > 7 && enabledFeatureTypes[7]) {
  //    activeFeatures["F7:orthographic-similarity"] = orthographicSimilarity;
  //  }

  // F8: <srcToken, tgtPrefix1> (subset of word association features in Chris et al. 2011)
  string tgtPrefix1 = tgtTokenString.length() > 0? tgtTokenString.substr(0,1) : "";
  if(enabledFeatureTypes.size() > 8 && enabledFeatureTypes[8]) {
    temp.str("");
    temp << "F8:" << srcToken << ":" << tgtPrefix1;
    activeFeatures[temp.str()] = 1.0;
  }

  // F9: <srcToken, tgtPrefix2> (subset of word association features in Chris et al. 2011)
  string tgtPrefix2 = tgtTokenString.length() > 1? tgtTokenString.substr(0,2) : "";
  if(enabledFeatureTypes.size() > 9 && enabledFeatureTypes[9]) {
    temp.str("");
    temp << "F9:" << srcToken << ":" << tgtPrefix2;
    activeFeatures[temp.str()] = 1.0;
  }

  // F10: <srcPrefix1, tgtToken> (subset of word association features in Chris et al. 2011)
  string srcPrefix1 = srcTokenString.size() > 0? srcTokenString.substr(0,1) : "";
  if(enabledFeatureTypes.size() > 10 && enabledFeatureTypes[10]) {
    temp.str("");
    temp << "F10:" << srcPrefix1 << ":" << tgtToken;
    activeFeatures[temp.str()] = 1.0;
  }

  // F11: <srcPrefix2, tgtToken> (subset of word association features in Chris et al. 2011)
  string srcPrefix2 = srcTokenString.size() > 1? srcTokenString.substr(0,2) : "";
  if(enabledFeatureTypes.size() > 11 && enabledFeatureTypes[11]) {
    temp.str("");
    temp << "F11:" << srcPrefix2 << ":" << tgtToken;
    activeFeatures[temp.str()] = 1.0;
  }

  // F12: ibm model 1 forward logprob (subset of word association features in Chris et al. 2011)
  float ibm1Forward = ibmModel1ForwardScores.find(srcToken)->second.find(tgtToken)->second;
  if(enabledFeatureTypes.size() > 12 && enabledFeatureTypes[12]) {
    temp.str("");
    temp << "F12:" << srcToken << ":" << tgtToken;
    activeFeatures[temp.str()] = ibm1Forward;
  }

  // F13: ibm model 1 backward logprob (subset of word association features in Chris et al. 2011)
  float ibm1Backward = ibmModel1BackwardScores.find(tgtToken)->second.find(srcToken)->second;
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
  
  for (map<string, float>::const_iterator paramsIter = params.begin(); paramsIter != params.end(); paramsIter++) {
    paramsFile << paramsIter->first << " " << paramsIter->second << endl;
  }

  paramsFile.close();
}

// use gradient based methods to update the model parameter weights
void LogLinearParams::UpdateParams(const LogLinearParams &gradient, const OptMethod& optMethod) {
  UpdateParams(gradient.params, optMethod);
}

// use gradient based methods to update the model parameter weights
void LogLinearParams::UpdateParams(const map<string, float> &gradient, const OptMethod& optMethod) {
  switch(optMethod.algorithm) {
  case GRADIENT_DESCENT:
  case STOCHASTIC_GRADIENT_DESCENT:
    for(map<string, float>::const_iterator gradientIter = gradient.begin();
	gradientIter != gradient.end();
	gradientIter++) {
      this->params[gradientIter->first] -= optMethod.learningRate * gradientIter->second;
    }
    break;
  default:
    assert(false);
    break;
  }
}

// if using a cumulative L1 regularizer, apply the cumulative l1 penalty
void LogLinearParams::ApplyCumulativeL1Penalty(const LogLinearParams& applyToFeaturesHere,
					       LogLinearParams& appliedL1Penalty,
					       const double correctL1Penalty) {
  for(map<string, float>::const_iterator featuresIter = applyToFeaturesHere.params.begin();
      featuresIter != applyToFeaturesHere.params.end();
      featuresIter++) {
    float currentFeatureWeight = params[featuresIter->first];
    if(currentFeatureWeight >= 0) {
      currentFeatureWeight = max(0.0, currentFeatureWeight - (correctL1Penalty + appliedL1Penalty.params[featuresIter->first]));
    } else {
      currentFeatureWeight = min(0.0, currentFeatureWeight + (correctL1Penalty - appliedL1Penalty.params[featuresIter->first]));
    }
    appliedL1Penalty.params[featuresIter->first] += currentFeatureWeight - params[featuresIter->first];
    params[featuresIter->first] = currentFeatureWeight;
  }
}

// compute a measure of orthographic similarity between two words
float LogLinearParams::ComputeOrthographicSimilarity(const std::string& srcWord, const std::string& tgtWord) {
  if(srcWord.length() == 0 || tgtWord.length() == 0) {
    return 0.0;
  }
  int levenshteinDistance = LevenshteinDistance(srcWord, tgtWord);
  if(levenshteinDistance > (srcWord.length() + tgtWord.length()) / 2) {
    return 0.0;
  } else {
    float similarity = (srcWord.length() + tgtWord.length()) / ((float)levenshteinDistance + 1);
    return similarity;
  }
}

int LogLinearParams::LevenshteinDistance(const std::string& x, const std::string& y) {
  if(x.length() == 0 && y.length() == 0) {
    return 0;
  }

  if(x.length() == 0) {
    return y.length();
  } else if (y.length() == 0) {
    return x.length();
  } else {
    int cost = x[0] != y[0]? 1 : 0;
    std::string xSuffix = x.substr(1);
    std::string ySuffix = y.substr(1);
    return std::min( std::min( LevenshteinDistance(xSuffix, y) + 1,
			       LevenshteinDistance(x, ySuffix) + 1),
		     LevenshteinDistance(xSuffix, ySuffix) + cost);
  }
}
