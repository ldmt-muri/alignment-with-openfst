#include "LogLinearParams.h"

using namespace std;

LogLinearParams::LogLinearParams(const VocabDecoder &srcTypes, 
				 const VocabDecoder &tgtTypes, 
				 const std::map<int, std::map<int, double> > &ibmModel1ForwardLogProbs,
				 const std::map<int, std::map<int, double> > &ibmModel1BackwardLogProbs,
				 double gaussianStdDev) :
  srcTypes(srcTypes), 
  tgtTypes(tgtTypes), 
  ibmModel1ForwardScores(ibmModel1ForwardLogProbs), 
  ibmModel1BackwardScores(ibmModel1BackwardLogProbs),
  COUNT_OF_FEATURE_TYPES(100) {
  learningInfo = 0;
  gaussianSampler = new GaussianSampler(0.0, gaussianStdDev);
}

LogLinearParams::LogLinearParams(const VocabDecoder &types, double gaussianStdDev) : 
  srcTypes(types), 
  tgtTypes(types),
  ibmModel1ForwardScores(map<int, map<int, double> >()),
  ibmModel1BackwardScores(map<int, map<int, double> >()),
  COUNT_OF_FEATURE_TYPES(100) {
  gaussianSampler = new GaussianSampler(0.0, gaussianStdDev);
}

void LogLinearParams::SetLearningInfo(const LearningInfo &learningInfo) {
  this->learningInfo = &learningInfo;
}

// initializes the parameter weight by drawing from a gaussian
bool LogLinearParams::AddParam(string paramId) {
  // sample paramWeight from an approx of gaussian with mean 0 and variance of 0.01
  double paramWeight = gaussianSampler->Draw();
  
  // add param
  return AddParam(paramId, paramWeight);
}

// if there's another parameter with the same ID already, do nothing
bool LogLinearParams::AddParam(string paramId, double paramWeight) {
  if(learningInfo->debugLevel >= DebugLevel::REDICULOUS) {
    cerr << "rank #" << learningInfo->mpiWorld->rank() << ": executing AddParam(" << paramId << "," << paramWeight << "); where |paramIndexes| = " << paramIndexes.size() << ", |paramWeights| = " << paramWeights.size() << endl;
  }
  bool returnValue;
  if(paramIndexes.count(paramId) == 0) {
    if(learningInfo->debugLevel >= DebugLevel::REDICULOUS) {
      cerr << "rank #" << learningInfo->mpiWorld->rank() << ": paramId is new.\n";
    }
    // check class's integrity
    assert(paramIndexes.size() == paramWeights.size());
    assert(paramIndexes.size() == oldParamWeights.size());
    assert(paramIndexes.size() == paramIds.size());
    // do the work
    int newParamIndex = paramIndexes.size();
    paramIndexes[paramId] = newParamIndex;
    paramWeights.push_back(paramWeight);
    oldParamWeights.push_back(0.0);
    paramIds.push_back(paramId);
    returnValue = true;
  } else {
    if(learningInfo->debugLevel >= DebugLevel::REDICULOUS) {
      cerr << "paramId already exists.";
    }
    returnValue = false;
  }  
  if(learningInfo->debugLevel >= DebugLevel::REDICULOUS) {
    cerr << "returning " << returnValue << endl;
  }
  return returnValue;
}

// features for the latnet crf model
void LogLinearParams::FireFeatures(int yI, int yIM1, const vector<int> &x, int i, 
				   const std::vector<bool> &enabledFeatureTypes, 
				   std::map<string, double> &activeFeatures) {
  FastSparseVector<double> f;
  FireFeatures(yI, yIM1, x, i, enabledFeatureTypes, f);
  for(FastSparseVector<double>::iterator fIter = f.begin(); fIter != f.end(); ++fIter) {
    activeFeatures[GetParamId(fIter->first)] += fIter->second;
  }
}

// features for the latent crf model
void LogLinearParams::FireFeatures(int yI, int yIM1, const vector<int> &x, int i, 
				   const std::vector<bool> &enabledFeatureTypes, 
				   FastSparseVector<double> &activeFeatures) {

  stringstream temp;

  const VocabDecoder &types = srcTypes;
  const int &xI = x[i];
  const std::string& xIString = types.Decode(x[i]);
  unsigned xIStringSize = xIString.size();
  const int &xIM1 = i-1 >= 0? x[i-1] : -1;
  const std::string& xIM1String = i-1 >= 0?
    types.Decode(x[i-1]):
    "_start_";
  unsigned xIM1StringSize = xIM1String.size();
  const int &xIM2 = i-2 >= 0? x[i-2] : -1;
  const std::string& xIM2String = i-2 >= 0?
    types.Decode(x[i-2]):
    "_start_";
  const int &xIP1 = i+1 < x.size()? x[i+1] : -1;
  const std::string &xIP1String = i+1 < x.size()?
    types.Decode(x[i+1]):
    "_end_";
  unsigned xIP1StringSize = xIP1String.size();
  const int &xIP2 = i+2 < x.size()? x[i+2] : -1; 
  const std::string &xIP2String = i+2 < x.size()?
    types.Decode(x[i+2]):
    "_end_";

  // F51: yIM1-yI pair
  if(enabledFeatureTypes.size() > 51 && enabledFeatureTypes[51]) {
    temp.str("");
    temp << "F51:" << yIM1 << ":" << yI;
    AddParam(temp.str());
    activeFeatures[paramIndexes[temp.str()]] += 1.0;
  }
  
  // F52: yI-xIM2 pair
  if(enabledFeatureTypes.size() > 52 && enabledFeatureTypes[52]) {
    temp.str("");
    temp << "F52:" << yI << ":" << xIM2String;
    AddParam(temp.str());
    activeFeatures[paramIndexes[temp.str()]] += 1.0;
  }

  // F53: yI-xIM1 pair
  if(enabledFeatureTypes.size() > 53 && enabledFeatureTypes[53]) {
    temp.str("");
    temp << "F53:" << yI << ":" << xIM1String;
    AddParam(temp.str());
    activeFeatures[paramIndexes[temp.str()]] += 1.0;
  }

  // F54: yI-xI pair
  if(enabledFeatureTypes.size() > 54 && enabledFeatureTypes[54]) {
    temp.str("");
    temp << "F54:" << yI << ":" << xIString;
    AddParam(temp.str());
    activeFeatures[paramIndexes[temp.str()]] += 1.0;
  }
  
  // F55: yI-xIP1 pair
  if(enabledFeatureTypes.size() > 55 && enabledFeatureTypes[55]) {
    temp.str("");
    temp << "F55:" << yI << ":" << xIP1String;
    AddParam(temp.str());
    activeFeatures[paramIndexes[temp.str()]] += 1.0;
  }

  // F56: yI-xIP2 pair
  if(enabledFeatureTypes.size() > 56 && enabledFeatureTypes[56]) {
    temp.str("");
    temp << "F56:" << yI << ":" << xIP2String;
    AddParam(temp.str());
    activeFeatures[paramIndexes[temp.str()]] += 1.0;
  }

  // F57: yI-i pair
  if(enabledFeatureTypes.size() > 57 && enabledFeatureTypes[57]) {
    if(i < 2) {
      temp.str("");
      temp << "F57:" << yI << ":" << i;
      AddParam(temp.str());
      activeFeatures[paramIndexes[temp.str()]] += 1.0;
    }
  }

  // F58: yI-prefix(xI)
  if(enabledFeatureTypes.size() > 58 && enabledFeatureTypes[58]) {
    //    if(xIStringSize > 0) {
    //      temp.str("");
    //      temp << "F58:" << yI << ":" << xIString[0];
    //      activeFeatures[temp.str()] += 1.0;
    //    }
    if(xIStringSize > 1) {
      temp.str("");
      temp << "F58:" << yI << ":" << xIString[0] << xIString[1];
      AddParam(temp.str());
      activeFeatures[paramIndexes[temp.str()]] += 1.0;
    }
    if(xIStringSize > 2) {
      temp.str("");
      temp << "F58:" << yI << ":" << xIString[0] << xIString[1] << xIString[2];
      AddParam(temp.str());
      activeFeatures[paramIndexes[temp.str()]] += 1.0;
    }
  }

  // F59: yI-suffix(xI)
  if(enabledFeatureTypes.size() > 59 && enabledFeatureTypes[59]) {
    if(xIStringSize > 0) {
      temp.str("");
      temp << "F59:" << yI << ":" << xIString[xIStringSize-1];
      AddParam(temp.str());
      activeFeatures[paramIndexes[temp.str()]] += 1.0;
    }
    if(xIStringSize > 1) {
      temp.str("");
      temp << "F59:" << yI << ":" << xIString[xIStringSize-2] << xIString[xIStringSize-1];
      AddParam(temp.str());
      activeFeatures[paramIndexes[temp.str()]] += 1.0;
    }
    if(xIStringSize > 2) {
      temp.str("");
      temp << "F59:" << yI << ":" << xIString[xIStringSize-3] << xIString[xIStringSize-2] << xIString[xIStringSize-1];
      AddParam(temp.str());
      activeFeatures[paramIndexes[temp.str()]] += 1.0;
    }
  }

  // F60: yI-prefix(xIP1)
  if(enabledFeatureTypes.size() > 60 && enabledFeatureTypes[60]) {
    //    if(xIP1StringSize > 0 && i+1 < x.size()) {
    //      temp.str("");
    //      temp << "F60:" << yI << ":" << xIP1String[0];
    //      activeFeatures[temp.str()] += 1.0;
    //    }
    if(xIP1StringSize > 1 && i+1 < x.size()) {
      temp.str("");
      temp << "F60:" << yI << ":" << xIP1String[0] << xIP1String[1];
      AddParam(temp.str());
      activeFeatures[paramIndexes[temp.str()]] += 1.0;
    }
    if(xIP1StringSize > 2 && i+1 < x.size()) {
      temp.str("");
      temp << "F60:" << yI << ":" << xIP1String[0] << xIP1String[1] << xIP1String[2];
      AddParam(temp.str());
      activeFeatures[paramIndexes[temp.str()]] += 1.0;
    }
  }

  // F61: yI-suffix(xIP1)
  if(enabledFeatureTypes.size() > 61 && enabledFeatureTypes[61]) {
    //    if(xIP1StringSize > 0 && i+1 < x.size()) {
    //      temp.str("");
    //      temp << "F61:" << yI << ":" << xIP1String[xIP1StringSize-1];
    //      activeFeatures[temp.str()] += 1.0;
    //    }
    if(xIP1StringSize > 1 && i+1 < x.size()) {
      temp.str("");
      temp << "F61:" << yI << ":" << xIP1String[xIP1StringSize-2] << xIP1String[xIP1StringSize-1];
      AddParam(temp.str());
      activeFeatures[paramIndexes[temp.str()]] += 1.0;
    }
    if(xIP1StringSize > 2 && i+1 < x.size()) {
      temp.str("");
      temp << "F61:" << yI << ":" << xIP1String[xIP1StringSize-3] << xIP1String[xIP1StringSize-2] << xIP1String[xIP1StringSize-1];
      AddParam(temp.str());
      activeFeatures[paramIndexes[temp.str()]] += 1.0;
    }
  }

  // F62: yI-prefix(xIM1)
  if(enabledFeatureTypes.size() > 62 && enabledFeatureTypes[62]) {
    //    if(xIM1StringSize > 0 && i-1 >= 0) {
    //      temp.str("");
    //      temp << "F62:" << yI << ":" << xIM1String[0];
    //      activeFeatures[temp.str()] += 1.0;
    //    }
    if(xIM1StringSize > 1 && i-1 >= 0) {
      temp.str("");
      temp << "F62:" << yI << ":" << xIM1String[0] << xIM1String[1];
      AddParam(temp.str());
      activeFeatures[paramIndexes[temp.str()]] += 1.0;
    }
    if(xIM1StringSize > 2 && i-1 >= 0) {
      temp.str("");
      temp << "F62:" << yI << ":" << xIM1String[0] << xIM1String[1] << xIM1String[2];
      AddParam(temp.str());
      activeFeatures[paramIndexes[temp.str()]] += 1.0;
    }
  }

  // F63: yI-suffix(xIM1)
  if(enabledFeatureTypes.size() > 63 && enabledFeatureTypes[63]) {
    //    if(xIM1StringSize > 0 && i-1 >= 0) {
    //      temp.str("");
    //      temp << "F63:" << yI << ":" << xIM1String[xIM1StringSize-1];
    //      activeFeatures[temp.str()] += 1.0;
    //    }
    if(xIM1StringSize > 1 && i-1 >= 0) {
      temp.str("");
      temp << "F63:" << yI << ":" << xIM1String[xIM1StringSize-2] << xIM1String[xIM1StringSize-1];
      AddParam(temp.str());
      activeFeatures[paramIndexes[temp.str()]] += 1.0;
    }
    if(xIM1StringSize > 2 && i-1 >= 0) {
      temp.str("");
      temp << "F63:" << yI << ":" << xIM1String[xIM1StringSize-3] << xIM1String[xIM1StringSize-2] << xIM1String[xIM1StringSize-1];
      AddParam(temp.str());
      activeFeatures[paramIndexes[temp.str()]] += 1.0;
    }
  }

  // F64: yI-hash(xI) ==> to be implemented in StringUtils ==> converts McDonald2s ==> AaAaaaia
  if(enabledFeatureTypes.size() > 64 && enabledFeatureTypes[64]) {
    temp.str("");
    temp << "F64:" << yI << ":";
    for(int j = 0; j < xIStringSize; j++) {
      char xIStringJ = xIString[j];
      if(xIStringJ >= 'a' && xIStringJ <= 'z') {
	temp << 'a'; // for small
      } else if(xIStringJ >= 'A' && xIStringJ <= 'Z') {
	temp << 'A'; // for capital
      } else if(xIStringJ >= '0' && xIStringJ <= '9') {
	temp << '0'; // for digit
      } else if(xIStringJ == '@') {
	temp << '@'; // for 'at'
      } else if(xIStringJ == '.') {
	temp << '.'; // for period
      } else {
	temp << '*';
      }
    }
    AddParam(temp.str());
    activeFeatures[paramIndexes[temp.str()]] += 1.0;
  }

  // F65: yI-capitalInitial(xI)
  if(enabledFeatureTypes.size() > 65 && enabledFeatureTypes[65]) {
    if(xIString[0] >= 'A' && xIString[0] <= 'Z') { 
      temp.str("");
      temp << "F65:" << yI << ":CapitalInitial";
      AddParam(temp.str());
      activeFeatures[paramIndexes[temp.str()]] += 1.0;
    }
  }

  // F66: yI-(|x|-i)
  if(enabledFeatureTypes.size() > 66 && enabledFeatureTypes[66]) {
    if(x.size() - i < 2) {
      temp.str("");
      temp << "F66:" << yI << ":" << (x.size()-i);
      AddParam(temp.str());
      activeFeatures[paramIndexes[temp.str()]] += 1.0;
    }
  }

  // F67: yI-capitalInitial(xI) && i > 0
  if(enabledFeatureTypes.size() > 67 && enabledFeatureTypes[67]) {
    if(i > 0 && xIString[0] >= 'A' && xIString[0] <= 'Z') { 
      temp.str("");
      temp << "F67:" << yI << ":CapitalInitialWithinSent";
      AddParam(temp.str());
      activeFeatures[paramIndexes[temp.str()]] += 1.0;
    }
  }

  // F68: yI-specialSuffix(xI)
  if(enabledFeatureTypes.size() > 68 && enabledFeatureTypes[68]) {
    if(xIStringSize > 0 && xIString[xIStringSize-1] == 's') {
      temp.str("");
      temp << "F68:" << yI << ":" << xIString[xIStringSize-1];
      AddParam(temp.str());
      activeFeatures[paramIndexes[temp.str()]] += 1.0;
    }
    if(xIStringSize > 1) {
      if((xIString[xIStringSize-2] == 'l' && xIString[xIStringSize-1] == 'y') ||
	 (xIString[xIStringSize-2] == 'e' && xIString[xIStringSize-1] == 'd')) {
	temp.str("");
	temp << "F68:" << yI << ":" << xIString[xIStringSize-2] << xIString[xIStringSize-1];
	AddParam(temp.str());
	activeFeatures[paramIndexes[temp.str()]] += 1.0;
      }
    }
    if(xIStringSize > 2) {
      if((xIString[xIStringSize-3] == 'i' && xIString[xIStringSize-2] == 'e' && xIString[xIStringSize-1] == 's') ||
	 (xIString[xIStringSize-3] == 'i' && xIString[xIStringSize-2] == 't' && xIString[xIStringSize-1] == 'y') ||
	 (xIString[xIStringSize-3] == 'i' && xIString[xIStringSize-2] == 'o' && xIString[xIStringSize-1] == 'n') ||
	 (xIString[xIStringSize-3] == 'o' && xIString[xIStringSize-2] == 'g' && xIString[xIStringSize-1] == 'y') ||
	 (xIString[xIStringSize-3] == 'i' && xIString[xIStringSize-2] == 'n' && xIString[xIStringSize-1] == 'g')) {      
	temp.str("");
	temp << "F68:" << yI << ":" << xIString[xIStringSize-3] << xIString[xIStringSize-2] << xIString[xIStringSize-1];
	AddParam(temp.str());
	activeFeatures[paramIndexes[temp.str()]] += 1.0;
      }
    }
    if(xIStringSize > 3) {
      if(xIString[xIStringSize-4] == 't' && xIString[xIStringSize-3] == 'i' && xIString[xIStringSize-2] == 'o' && xIString[xIStringSize-1] == 'n') {
	temp.str("");
	temp << "F68:" << yI << ":" << xIString[xIStringSize-4] << xIString[xIStringSize-3] << xIString[xIStringSize-2] << xIString[xIStringSize-1];
	AddParam(temp.str());
	activeFeatures[paramIndexes[temp.str()]] += 1.0;
      }
    }
  }

  // F69: yI-coarserHash(xI) ==> to be implemented in StringUtils ==> converts McDonald2s ==> AaAaaaia
  if(enabledFeatureTypes.size() > 69 && enabledFeatureTypes[69]) {
    bool small = false, capital = false, number = false, punctuation = false, other = false;
    temp.str("");
    temp << "F69:" << yI << ":";
    for(int j = 0; j < xIStringSize; j++) {
      char xIStringJ = xIString[j];
      if(!small && xIStringJ >= 'a' && xIStringJ <= 'z') {
	small = true;
      } else if(!capital && xIStringJ >= 'A' && xIStringJ <= 'Z') {
	capital = true;
      } else if(!number && xIStringJ >= '0' && xIStringJ <= '9') {
	number = true;
      } else if(!punctuation && (xIStringJ == ':' || xIStringJ == '.' || xIStringJ == ';' || xIStringJ == ',' || xIStringJ == '?' || xIStringJ == '!')) {
	punctuation = true;
      } else if(!other) {
	other = true;
      }
    }
    if(small) {
      temp << 'a'; 
    }
    if(capital) {
      temp << 'A'; 
    }
    if(number) {
      temp << '0';
    } 
    if(punctuation) {
      temp << ';';
    }
    if(other) {
      temp << '#';
    }
    AddParam(temp.str());
    activeFeatures[paramIndexes[temp.str()]] += 1.0;
  }

  // F70: yI-xIM2 pair, where xIM2 is closed vocab
  if(enabledFeatureTypes.size() > 70 && enabledFeatureTypes[70]
     && types.IsClosedVocab(xIM2)) {
    temp.str("");
    temp << "F70:" << yI << ":" << xIM2String;
    AddParam(temp.str());
    activeFeatures[paramIndexes[temp.str()]] += 1.0;
  }

  // F71: yI-xIM1 pair, where xIM1 is closed vocab
  if(enabledFeatureTypes.size() > 71 && enabledFeatureTypes[71]
     && types.IsClosedVocab(xIM1)) {
    temp.str("");
    temp << "F71:" << yI << ":" << xIM1String;
    AddParam(temp.str());
    activeFeatures[paramIndexes[temp.str()]] += 1.0;
  }

  // F72: yI-xI pair, where xI is closed vocab
  if(enabledFeatureTypes.size() > 72 && enabledFeatureTypes[72]
     && types.IsClosedVocab(xI)) {
    temp.str("");
    temp << "F72:" << yI << ":" << xIString;
    AddParam(temp.str());
    activeFeatures[paramIndexes[temp.str()]] += 1.0;
  }
  
  // F73: yI-xIP1 pair, where xIP1 is closed vocab
  if(enabledFeatureTypes.size() > 73 && enabledFeatureTypes[73]
     && types.IsClosedVocab(xIP1)) {
    temp.str("");
    temp << "F73:" << yI << ":" << xIP1String;
    AddParam(temp.str());
    activeFeatures[paramIndexes[temp.str()]] = 1.0;
  }

  // F74: yI-xIP2 pair, where xIP2 is closed vocab
  if(enabledFeatureTypes.size() > 74 && enabledFeatureTypes[74]
     && types.IsClosedVocab(xIP2)) {
    temp.str("");
    temp << "F74:" << yI << ":" << xIP2String;
    AddParam(temp.str());
    activeFeatures[paramIndexes[temp.str()]] += 1.0;
  }
  
  // F75: yI
  if(enabledFeatureTypes.size() > 75 && enabledFeatureTypes[75]) {
    temp.str("");
    temp << "F75:" << yI;
    AddParam(temp.str());
    activeFeatures[paramIndexes[temp.str()]] += 1.0;
  }
}

double LogLinearParams::Hash() {
  double hash = 0.0;
  for(vector<double>::const_iterator paramIter = paramWeights.begin(); paramIter != paramWeights.end(); paramIter++) {
    hash += *paramIter;
  }
  return hash;
} 

void LogLinearParams::FireFeatures(int srcToken, int prevSrcToken, int tgtToken, int srcPos, int prevSrcPos, int tgtPos, 
				   int srcSentLength, int tgtSentLength, 
				   const std::vector<bool>& enabledFeatureTypes, 
				   std::map<string, double>& activeFeatures) {
  
  // for debugging
  //    cerr << "srcToken=" << srcToken << " tgtToken=" << tgtToken << " srcPos=" << srcPos << " tgtPos=" << tgtPos;
  //    cerr << " srcSentLength=" << srcSentLength << " tgtSentLength=" << tgtSentLength << endl;

  assert(activeFeatures.size() == 0);
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
    cerr << paramsIter->first << " " << paramWeights[paramsIter->second] << " at " << paramsIter->second << endl;
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
  cerr << "##################" << endl;
  cerr << "pointer to internal weights: " << paramWeights.data() << ". pointer to external weights: " << array << endl;
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
