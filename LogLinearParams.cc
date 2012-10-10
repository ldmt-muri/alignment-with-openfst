#include "LogLinearParams.h"

using namespace std;
using namespace OptUtils;

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

float LogLinearParams::ComputeLogProb(int srcToken, int tgtToken, int srcPos, int prevSrcPos, int tgtPos, 
				      int srcSentLength, int tgtSentLength, 
				      const std::vector<bool>& enabledFeatureTypes) {
  map<string, float> activeFeatures;
  FireFeatures(srcToken, tgtToken, srcPos, prevSrcPos, tgtPos, srcSentLength, tgtSentLength, enabledFeatureTypes, activeFeatures);
  // compute log prob
  float result = DotProduct(activeFeatures, params);
  
  // for debugging
  //  cerr << "srcToken=" << srcToken << " tgtToken=" << tgtToken << " srcPos=" << srcPos << " tgtPos=" << tgtPos;
  //  cerr << " srcSentLength=" << srcSentLength << " tgtSentLength=" << tgtSentLength << endl;
  //  cerr << "RESULT=" << result << endl << endl;

  return result;
}

void LogLinearParams::FireFeatures(int srcToken, int tgtToken, int srcPos, int prevSrcPos, int tgtPos, 
				   int srcSentLength, int tgtSentLength, 
				   const std::vector<bool>& enabledFeatureTypes, 
				   std::map<string, float>& activeFeatures) {
  stringstream temp;

  // F1: src-tgt pair (subset of word association features in Chris et al. 2011)
  if(enabledFeatureTypes.size() > 1 && enabledFeatureTypes[1]) {
    temp << "F1:" << srcToken << "-" << tgtToken;
    activeFeatures[temp.str()] = 1.0;
  }

  // F2: diagonal-bias (positional features in Chris et al. 2011)
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
  if(enabledFeatureTypes.size() > 4 && enabledFeatureTypes[4]) {
    temp.str("");
    temp << "F4:" << abs(srcPos - prevSrcPos);
    activeFeatures[temp.str()] = 1.0;
  }

  // F5: alignment jump direction (subset of src path features in Chris et al. 2011)
  if(enabledFeatureTypes.size() > 5 && enabledFeatureTypes[5]) {
    temp.str("");
    int dir = srcPos > prevSrcPos? +1 : srcPos < prevSrcPos? -1 : 0;
    temp << "F5:" << dir;
    activeFeatures[temp.str()] = 1.0;
  }
 
  // F6: alignment jump from/to (subset of src path features in Chris et al. 2011)
  if(enabledFeatureTypes.size() > 6 && enabledFeatureTypes[6]) {
    temp.str("");
    temp << "F6:" << prevSrcPos << ":" << srcPos;
    activeFeatures[temp.str()] = 1.0;
  }
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
void LogLinearParams::UpdateParams(const LogLinearParams& gradient, const OptMethod& optMethod) {
  switch(optMethod.algorithm) {
  case GRADIENT_DESCENT:
  case STOCHASTIC_GRADIENT_DESCENT:
    for(map<string, float>::const_iterator gradientIter = gradient.params.begin();
	gradientIter != gradient.params.end();
	gradientIter++) {
      params[gradientIter->first] -= gradientIter->second;
    }
    break;
  default:
    assert(false);
    break;
  }
}
