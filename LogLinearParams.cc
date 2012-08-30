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

float LogLinearParams::ComputeLogProb(int srcToken, int tgtToken, int srcPos, int tgtPos, 
				     int srcSentLength, int tgtSentLength) {
  map<string, float> activeFeatures;
  FireFeatures(srcToken, tgtToken, srcPos, tgtPos, srcSentLength, tgtSentLength, activeFeatures);
  // compute log prob
  float result = DotProduct(activeFeatures, params);
  
  // for debugging
  //  cerr << "srcToken=" << srcToken << " tgtToken=" << tgtToken << " srcPos=" << srcPos << " tgtPos=" << tgtPos;
  //  cerr << " srcSentLength=" << srcSentLength << " tgtSentLength=" << tgtSentLength << endl;
  //  cerr << "RESULT=" << result << endl << endl;

  return result;
}

void LogLinearParams::FireFeatures(int srcToken, int tgtToken, int srcPos, int tgtPos, 
				   int srcSentLength, int tgtSentLength, std::map<string, float>& activeFeatures) {
  stringstream temp;

  // F1: src-tgt pair
  temp << "F1:" << srcToken << "-" << tgtToken;
  activeFeatures[temp.str()] = 1.0;

  // F2: diagonal-bias
  activeFeatures["F2:diagonal-bias"] = fabs((float) srcPos / srcSentLength) - ((float) tgtPos / tgtSentLength);
  
  // F3: src token
  temp.str("");
  temp << "F3:" << srcToken;
  activeFeatures[temp.str()] = 1.0;
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
