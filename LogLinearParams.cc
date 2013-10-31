#include "LogLinearParams.h"

//VocabEncoder* FeatureId::precomputedFeaturesEncoder = 0;

using namespace std;
using namespace boost;

std::ostream& operator<<(std::ostream& os, const FeatureId& obj)
{
  os << obj.type;
  switch(obj.type) {
  case FeatureTemplate::LABEL_BIGRAM:
  case FeatureTemplate::SRC_BIGRAM:
    os << '|' << obj.bigram.previous << '|' << obj.bigram.current;
    break;
  case FeatureTemplate::ALIGNMENT_JUMP:
  case FeatureTemplate::LOG_ALIGNMENT_JUMP:
  case FeatureTemplate::ALIGNMENT_JUMP_IS_ZERO:
    os << '|' << obj.alignmentJump;
    break;
  case FeatureTemplate::SRC0_TGT0:
    os << '|' << obj.wordPair.srcWord << '|' << obj.wordPair.tgtWord;
    break;
  case FeatureTemplate::PRECOMPUTED:
    //os << '|' << obj.DecodePrecomputedFeature();
    os << '|' << obj.precomputed;
    break;
  case FeatureTemplate::DIAGONAL_DEVIATION:
  case FeatureTemplate::SYNC_START:
  case FeatureTemplate::SYNC_END:
    break;
  default:
    assert(false);
  }
  os << '|';
  return os;
}

std::istream& operator>>(std::istream& is, FeatureId& obj)
{
  // read obj from stream
  int temp;
  string temp2;
  is >> temp;
  obj.type = (FeatureTemplate)temp;
  char separator;
  switch(obj.type) {
  case FeatureTemplate::LABEL_BIGRAM:
  case FeatureTemplate::SRC_BIGRAM:
    is >> separator >> obj.bigram.current;
    assert(separator == '|');
    is >> separator >> obj.bigram.previous;
    assert(separator == '|');
    break;
  case FeatureTemplate::ALIGNMENT_JUMP:
  case FeatureTemplate::LOG_ALIGNMENT_JUMP:
  case FeatureTemplate::ALIGNMENT_JUMP_IS_ZERO:
    is >> separator >> obj.alignmentJump;
    assert(separator == '|');
    break;
  case FeatureTemplate::SRC0_TGT0:
    is >> separator >> obj.wordPair.srcWord;
    assert(separator == '|');
    is >> separator >> obj.wordPair.tgtWord;
    assert(separator == '|');
    break;
  case FeatureTemplate::PRECOMPUTED:
    //is >> separator >> temp2;
    //obj.EncodePrecomputedFeature(temp2);
    is >> separator >> obj.precomputed;
    assert(separator == '|');
    break;
  case FeatureTemplate::DIAGONAL_DEVIATION:
  case FeatureTemplate::SYNC_START:
  case FeatureTemplate::SYNC_END:
    break;
  default:
    is.setstate(std::ios::failbit);
    assert(false);
  }
  is >> separator;
  assert(separator == '|');
  return is;
}

LogLinearParams::~LogLinearParams() {
  if(sharedMemorySegment != 0) {
    delete sharedMemorySegment;
  }
}

LogLinearParams::LogLinearParams(VocabEncoder &types, 
				 double gaussianStdDev) :
  types(types) {
  learningInfo = 0;
  gaussianSampler = new GaussianSampler(0.0, gaussianStdDev);
  //FeatureId::precomputedFeaturesEncoder = &precomputedFeaturesEncoder;
  sealed = false;
  paramIdsPtr = 0;
  paramWeightsPtr = 0;
  sharedMemorySegment = 0;
}

bool LogLinearParams::IsSealed() const {
  return sealed;
}



void* LogLinearParams::MapToSharedMemory(bool create, const string objectNickname) {

  if(string(objectNickname) == string("paramWeights")) {
    ShmemDoubleAllocator sharedMemoryDoubleAllocator(sharedMemorySegment->get_segment_manager()); 
    if(create) {
      return sharedMemorySegment->construct<ShmemVectorOfDouble> (objectNickname.c_str()) (sharedMemoryDoubleAllocator);
    } else {
      return sharedMemorySegment->find<ShmemVectorOfDouble> (objectNickname.c_str()).first;
    }

  } else if (string(objectNickname) == string("paramIds")) {
    ShmemFeatureIdAllocator sharedMemoryFeatureIdAllocator(sharedMemorySegment->get_segment_manager());
    if(create) {
      return sharedMemorySegment->construct<ShmemVectorOfFeatureId> (objectNickname.c_str()) (sharedMemoryFeatureIdAllocator);
    } else {
      return sharedMemorySegment->find<ShmemVectorOfFeatureId> (objectNickname.c_str()).first;
    }

  } /*else if (string(objectNickname) == string("precomputedFeaturesWithTwoInputs")) {
    ShmemOuterValueAllocator sharedMemoryNestedMapAllocator(sharedMemorySegment->get_segment_manager());
    if(create) {
      return sharedMemorySegment->construct<ShmemNestedMap> (objectNickname) (std::less<OuterKeyType>(), sharedMemoryNestedMapAllocator);
      } else {
      return sharedMemorySegment->find<ShmemNestedMap> (objectNickname).first;
    }
    } */ else {
    assert(false);
  }
}

OuterMappedType* LogLinearParams::MapWordPairFeaturesToSharedMemory(bool create, string& objectNickname) {
  ShmemInnerValueAllocator sharedMemorySimpleMapAllocator(sharedMemorySegment->get_segment_manager()); 
  if(create) {
    //return 0;
    return sharedMemorySegment->construct< OuterMappedType > (objectNickname.c_str()) (std::less<InnerKeyType>(), sharedMemorySimpleMapAllocator);
  } else {
    //return 0;
    return sharedMemorySegment->find< OuterMappedType > (objectNickname.c_str()).first;
  }
}

void LogLinearParams::SetSharedMemorySegment(bool create)
{
  size_t segmentSize = 30 * 1024; // in GBs
  segmentSize *= 1024 * 1024;
  string SEGMENT_NAME = "segment";
  using namespace boost::interprocess;
  // Shared memory front-end that is able to construct objects
  // associated with a c-string. Erase previous shared memory with the name
  // to be used and create the memory segment at the specified address and initialize resources
  if(create) {
    cerr << "remove any shared memory object with the same name '" << SEGMENT_NAME << "'...";
    shared_memory_object::remove(SEGMENT_NAME.c_str());
    cerr << "done" << endl;

    // create or open the shared memory segments
    cerr << "requesting " << segmentSize << " bytes of managed shared memory for segment " << SEGMENT_NAME << "..." << endl;
    sharedMemorySegment = new managed_shared_memory(open_or_create, SEGMENT_NAME.c_str(), segmentSize);
    cerr << "done." << endl;
    bool dummy = true;
    mpi::broadcast<bool>(*learningInfo->mpiWorld, dummy, 0);
  } else {
    // master must 
    bool dummy;
    mpi::broadcast<bool>(*learningInfo->mpiWorld, dummy, 0);
    cerr << "opening segment " << SEGMENT_NAME << " ...";
    sharedMemorySegment = new managed_shared_memory(open_only, SEGMENT_NAME.c_str());      
    cerr << "done.";
  }
  assert(sharedMemorySegment->get_size() == segmentSize);
}

void LogLinearParams::Seal() {
  assert(!sealed);
  assert(paramIdsPtr == 0 && paramWeightsPtr == 0);
  if(learningInfo->mpiWorld->rank() == 0) {
    paramWeightsPtr = (ShmemVectorOfDouble *) MapToSharedMemory(true, "paramWeights");
    assert(paramWeightsPtr != 0);
    paramIdsPtr = (ShmemVectorOfFeatureId *) MapToSharedMemory(true, "paramIds");
    assert(paramIdsPtr != 0);
    
    // copy paramIdsTemp and paramWeightsTemp, and wipe off temporary parameters you had
    assert(paramWeightsTemp.size() == paramIdsTemp.size());
    for(int i = 0; i < paramWeightsTemp.size(); ++i) {
      paramWeightsPtr->push_back(paramWeightsTemp[i]);
      paramIdsPtr->push_back(paramIdsTemp[i]);
    }
    paramWeightsTemp.clear(); 
    paramIdsTemp.clear(); 
  } else {
    // this is done by the slaves, not the master
    assert(learningInfo->mpiWorld->rank() != 0);
    // first, wipe off all the parameters you already have to save memory
    paramWeightsTemp.clear(); 
    paramIdsTemp.clear(); 

    // map paramIds and paramWeights to shared memory
    paramWeightsPtr = (ShmemVectorOfDouble *)MapToSharedMemory(false, "paramWeights");
    paramIdsPtr = (ShmemVectorOfFeatureId *)MapToSharedMemory(false, "paramIds");
    //precomputedFeaturesWithTwoInputsPtr = (ShmemNestedMap *)MapToSharedMemory(false, "precomputedFeaturesWithTwoInputs");
  }
  assert(paramIdsPtr != 0 && paramWeightsPtr != 0);
  sealed = true;
}

// add a featureId/featureValue pair to the map at precomputedFeatures[input1][input2]
void LogLinearParams::AddToPrecomputedFeaturesWith2Inputs(int input1, int input2, FeatureId &featureId, double featureValue) {
  assert(false);
  assert(!sealed);
  std::tuple<int, int> srcTgtPair(input1, input2);
  //(*precomputedFeaturesWithTwoInputsPtr)[srcTgtPair][featureId] = featureValue;
  //  AddParam(featureId);
}

// by two inputs, i mean that a precomputed feature value is a function of two strings
// example line in the precomputed features file:
// madrasa ||| school ||| F52:editdistance=7 F53:capitalconsistency=1
void LogLinearParams::LoadPrecomputedFeaturesWith2Inputs(const string &wordPairFeaturesFilename) {
  assert(learningInfo->mpiWorld->rank() == 0);

  // create the map
  //precomputedFeaturesWithTwoInputsPtr = (ShmemNestedMap *) MapToSharedMemory(true, "precomputedFeaturesWithTwoInputs");
  //assert(precomputedFeaturesWithTwoInputsPtr != 0);
  
  cerr << "rank " << learningInfo->mpiWorld->rank() << " is going to read the word pair features file..." << endl;
  //int expectedSrcTgtPairs = 25 * 1000; expectedSrcTgtPairs *= 1000;
    //precomputedFeaturesEncoder.ReserveVocabSize(expectedVocabSize);
    //precomputedFeaturesWithTwoInputsPtr->reserve( expectedSrcTgtPairs );
    //cerr << "precomputedFeaturesWithTwoInputs.reserve(" << expectedSrcTgtPairs << ") done." << endl;
    ifstream wordPairFeaturesFile(wordPairFeaturesFilename.c_str(), ios::in);
    string line;
    int featsCounter = 0;
    std::hash<std::string> str_hash;
    while( getline(wordPairFeaturesFile, line) ) {
      if(line.size() == 0) {
        continue;
      }
      std::vector<string> splits;
      StringUtils::SplitString(line, ' ', splits);
      // check format
      if(splits.size() < 5) {
        assert(false);
        exit(1);
      }
      // splitsIter
      vector<string>::iterator splitsIter = splits.begin();
      // read the first input
      string &input1String = *(splitsIter++);
      int input1 = types.ConstEncode(input1String);
      // skip |||
      splitsIter++;
      // read the second input
      string &input2String = *(splitsIter++);
      int input2 = types.ConstEncode(input2String);
      // skip |||
      splitsIter++;
      std::stringstream srcTgtPairKey;
      srcTgtPairKey << input1 << " " << input2;
      string srcTgtPairKeyString = srcTgtPairKey.str();
      //offset_ptr<std::_Rb_tree_node<std::pair<const FeatureId, double> >, long, unsigned long, 0UL> *tempMap = MapWordPa
      auto tempMap = MapWordPairFeaturesToSharedMemory(true, srcTgtPairKeyString);
      // the remaining elements are precomputed features for (input1, input2)
      while(splitsIter != splits.end()) {
        // read feature id
        std::vector<string> featureIdAndValue;
        StringUtils::SplitString(*(splitsIter++), '=', featureIdAndValue);
        assert(featureIdAndValue.size() == 2);
        FeatureId featureId;
        featureId.type = FeatureTemplate::PRECOMPUTED;
        //featureId.precomputed = precomputedFeaturesEncoder.Encode(featureIdAndValue[0]);
        featureId.precomputed = str_hash(featureIdAndValue[0]);

        // read feature value
        double featureValue;
        stringstream temp;
        temp << featureIdAndValue[1];
        temp >> featureValue;
        
        // store it for quick retrieval later on
        //std::tuple<int, int> srcTgtPair(input1, input2);
        //if(precomputedFeaturesWithTwoInputsPtr->find(srcTgtPair) == precomputedFeaturesWithTwoInputsPtr->end() && \
        //   precomputedFeaturesWithTwoInputsPtr->size() % 1000000 == 0) {
        //  cerr << "precomputedFeaturesWithTwoInputs.size() == " << precomputedFeaturesWithTwoInputsPtr->size() << endl;
        //}
        //(*precomputedFeaturesWithTwoInputsPtr)[srcTgtPair][featureId] = featureValue;
        tempMap->insert( std::pair<FeatureId, double>( featureId, featureValue ));
        
        //if(++featsCounter % 10000000 == 0) {
        //  cerr << "total featsCounter (including repetitions) = " << featsCounter << endl;
        //}
      }
    }
    
    wordPairFeaturesFile.close();
    cerr << "rank " << learningInfo->mpiWorld->rank() << " finished reading the word pair features file." << endl;

}

void LogLinearParams::SetLearningInfo(const LearningInfo &learningInfo) {
  this->learningInfo = &learningInfo;
  SetSharedMemorySegment(learningInfo.mpiWorld->rank() == 0);
}

int LogLinearParams::AddParams(const std::vector< FeatureId > &paramIds) {
  paramIndexes.reserve(paramIndexes.size() + paramIds.size());
  paramIdsTemp.reserve(paramIdsTemp.size() + paramIds.size());
  paramWeightsTemp.reserve(paramWeightsTemp.size() + paramIds.size());
  cerr << "paramIdsTemp.capacity() = " << paramIdsTemp.capacity() << endl;
  int newParamsCount = 0;
  for(int i = 0; i < paramIds.size(); ++i) {
    if(AddParam(paramIds[i])) {
      newParamsCount++;
    }
  }
  return newParamsCount;
}

// initializes the parameter weight by drawing from a gaussian
bool LogLinearParams::AddParam(const FeatureId &paramId) {
  // does the parameter already exist?
  if(paramIndexes.count(paramId) > 0) {
    return false;
  }

  // sample paramWeight from an approx of gaussian with mean 0 and variance of 0.01
  double paramWeight = 0;
  if(this->learningInfo->initializeLambdasWithGaussian) {
    paramWeight = -1.0 * fabs(gaussianSampler->Draw());
  } else if (this->learningInfo->initializeLambdasWithOne) { 
    paramWeight = -0.01;
  } else if (this->learningInfo->initializeLambdasWithZero) {
    paramWeight = 0.0;
  } else {
    assert(false);
  }
  
  // add param
  return AddParam(paramId, paramWeight);
}

// if there's another parameter with the same ID already, do nothing
bool LogLinearParams::AddParam(const FeatureId &paramId, double paramWeight) {
  bool returnValue;
  if(paramIndexes.count(paramId) == 0) {
    
    // new features are not allowed when the object is sealed
    assert(!sealed);

    // check class's integrity -- the object is not sealed
    assert(paramWeightsPtr == 0);
    assert(paramIdsPtr == 0);
    assert(paramIndexes.size() == paramIdsTemp.size());

    // do the work
    int newParamIndex = paramIndexes.size();
    paramIndexes[paramId] = newParamIndex;
    paramIdsTemp.push_back(paramId);
    paramWeightsTemp.push_back(paramWeight);
    returnValue = true;
  } else {
    returnValue = false;
  }  
  return returnValue;
}

void LogLinearParams::PrintFeatureValues(FastSparseVector<double> &feats) {
  assert(sealed);
  cerr << "active features are: " << endl;
  for(auto feat = feats.begin();
      feat != feats.end();
      ++feat) {
    cerr << "  index=" << feat->first << ", id=" << (*paramIdsPtr)[feat->first] << ", val=" << feat->second << endl;
  }
}

// x_t is the tgt sentence, and x_s is the src sentence (which has a null token at position 0)
void LogLinearParams::FireFeatures(int yI, int yIM1, const vector<int> &x_t, const vector<int> &x_s, int i, 
				   int START_OF_SENTENCE_Y_VALUE, int FIRST_POS,
				   FastSparseVector<double> &activeFeatures) {
  // debug info
  if(learningInfo->debugLevel >= DebugLevel::REDICULOUS) {
    cerr << "executing FireFeatures(yI=" << yI << ", yIM1=" << yIM1 << ", x_t.size()=" << x_t.size();
    cerr << ", x_s.size()=" << x_s.size() << ", i=" << i << ", START=" << START_OF_SENTENCE_Y_VALUE;
    cerr << ", FIRST_POS=" << FIRST_POS;
    cerr << ", activeFeatures.size()=" << activeFeatures.size() << endl;
  }

  // first, yI and yIM1 are not zero-based. LatentCrfAligner::FIRST_POS maps to the first position in the src sentence (this means the null token, if null alignments are enabled, or the first token in the src sent). 
  yI -= FIRST_POS;
  yIM1 -= FIRST_POS;
      
  // find the src token aligned according to x_t_i
  assert(yI < (int)x_s.size());
  assert(yIM1 < (int)x_s.size()); 
  assert(yI >= 0);
  assert(yIM1 >= -2);
  int srcToken = x_s[yI];
  int prevSrcToken = yIM1 >= 0? x_s[yI] : START_OF_SENTENCE_Y_VALUE;
  int tgtToken = x_t[i];
  int prevTgtToken = i > 0? x_t[i-1] : -1;
  int nextTgtToken = (i < x_t.size() - 1)? x_t[i+1] : -1;
  //std::tuple<int, int> srcTgtPair(srcToken, tgtToken);
  std::stringstream srcTgtPairKey;
  srcTgtPairKey << srcToken << " " << tgtToken;
  string srcTgtPairKeyStr = srcTgtPairKey.str();
  auto precomputedFeatures = MapWordPairFeaturesToSharedMemory(false, srcTgtPairKeyStr);
  //  precomputedFeaturesWithTwoInputsPtr->size() != 0?
  //  &(  (*precomputedFeaturesWithTwoInputsPtr)[srcTgtPair] ) : 0;
  
  FeatureId featureId;
  
  for(auto featTemplateIter = learningInfo->featureTemplates.begin();
      featTemplateIter != learningInfo->featureTemplates.end(); ++featTemplateIter) {
    
    switch(*featTemplateIter) {
      case FeatureTemplate::ALIGNMENT_JUMP_IS_ZERO:
      featureId.type = FeatureTemplate::ALIGNMENT_JUMP_IS_ZERO;
      featureId.alignmentJump = (yI == yIM1) ? 0 : 1;
      AddParam(featureId);
      activeFeatures[paramIndexes[featureId]] += 1.0;
      break;

      case FeatureTemplate::LOG_ALIGNMENT_JUMP:
      featureId.type = FeatureTemplate::LOG_ALIGNMENT_JUMP;
      featureId.alignmentJump = log(yI - yIM1);
      AddParam(featureId);
      activeFeatures[paramIndexes[featureId]] += 1.0;
      break;

      case FeatureTemplate::SRC0_TGT0:
      featureId.type = FeatureTemplate::SRC0_TGT0;
      featureId.wordPair.srcWord = srcToken;
      featureId.wordPair.tgtWord = tgtToken;
      AddParam(featureId);
      activeFeatures[paramIndexes[featureId]] += 1.0;
      break;

      case FeatureTemplate::PRECOMPUTED:
        //assert(precomputedFeaturesWithTwoInputsPtr->size() > 0);
        if(precomputedFeatures) {
          for(auto precomputedIter = precomputedFeatures->begin();
              precomputedIter != precomputedFeatures->end();
              precomputedIter++) {
            AddParam(precomputedIter->first);
            activeFeatures[paramIndexes[precomputedIter->first]] += precomputedIter->second;
          }
        }
      break;

      case FeatureTemplate::ALIGNMENT_JUMP:
      featureId.type = FeatureTemplate::ALIGNMENT_JUMP;
      featureId.alignmentJump = yI - yIM1;
      AddParam(featureId);
      activeFeatures[paramIndexes[featureId]] += 1.0;
      break;
    
      case FeatureTemplate::SRC_BIGRAM:
      featureId.type = FeatureTemplate::SRC_BIGRAM;
      featureId.bigram.previous = prevSrcToken;
      featureId.bigram.current = srcToken;
      AddParam(featureId);
      activeFeatures[paramIndexes[featureId]] += 1.0;
      break;
      
      case FeatureTemplate::DIAGONAL_DEVIATION:
      if(yI + yIM1 > 0) {
        featureId.type = FeatureTemplate::DIAGONAL_DEVIATION;
        AddParam(featureId);
        double deviation = (yI > 0)?
          fabs(1.0 * (yI-1) / (x_s.size()-1) - 1.0 * i / x_t.size()):
          fabs(1.0 * (yIM1-1) / (x_s.size()-1) - 1.0 * i / x_t.size());
        activeFeatures[paramIndexes[featureId]] += deviation;
      }
      break;

      case FeatureTemplate::SYNC_START:
      if(i == 0 && yI == 0) {
        featureId.type = FeatureTemplate::SYNC_START;
        AddParam(featureId);
        activeFeatures[paramIndexes[featureId]] += 1.0;
      }
      break;
      
      case FeatureTemplate::SYNC_END:
      if(i == x_t.size() - 1 && yI == x_s.size() - 1) {
        featureId.type = FeatureTemplate::SYNC_END;
        AddParam(featureId);
        activeFeatures[paramIndexes[featureId]] += 1.0;
      }
      break;
    
      case FeatureTemplate::LABEL_BIGRAM:
      cerr << "this feature template is not implemented for word alignment" << endl;
      assert(false);
      break;
      
      default:
        assert(false);
    } // end of switch
  } // end of loop over enabled feature templates
}
/*
// features for the latent crf model
void LogLinearParams::FireFeatures(int yI, int yIM1, const vector<int> &x, int i, 
				   const std::vector<bool> &enabledFeatureTypes, 
				   FastSparseVector<double> &activeFeatures) {
  
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
*/

double LogLinearParams::Hash() {
  assert(sealed);
  double hash = 0.0;
  for(auto paramIter = paramWeightsPtr->begin(); paramIter != paramWeightsPtr->end(); paramIter++) {
    hash += *paramIter;
  }
  return hash;
} 

/*
void LogLinearParams::FireFeatures(int srcToken, int prevSrcToken, int tgtToken, int srcPos, int prevSrcPos, int tgtPos, 
				   int srcSentLength, int tgtSentLength, 
				   const std::vector<bool>& enabledFeatureTypes, 
				   boost::unordered_map<string, double>& activeFeatures) {
  
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
  const std::string& srcTokenString = types.Decode(srcToken);
  const std::string& tgtTokenString = types.Decode(tgtToken);
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

*/

// each line consists of: <featureStringId><space><featureWeight>\n
void LogLinearParams::PersistParams(const string &outputFilename, bool humanFriendly) {

  assert(sealed);

  ofstream paramsFile(outputFilename.c_str());
  
  assert(paramsFile.good());

  if(!humanFriendly) {
    // save data to archive
    archive::text_oarchive oa(paramsFile);
    // write class instance to archive
    oa << *this;
    // archive and stream closed when destructors are called
  } else {
    for (auto paramsIter = paramIndexes.begin(); paramsIter != paramIndexes.end(); paramsIter++) {
      paramsFile << paramsIter->first << " " << (*paramWeightsPtr)[paramsIter->second] << endl;
    }
  }
  paramsFile.close();
}

// each line consists of: <featureStringId><space><featureWeight>\n
void LogLinearParams::LoadParams(const string &inputFilename) {
  assert(!sealed);
  
  // master is in charge of laoding params
  if(learningInfo->mpiWorld->rank() == 0){ 
    assert(paramIndexes.size() == paramIdsTemp.size());
    ifstream paramsFile(inputFilename.c_str(), ios::in);
    boost::archive::text_iarchive oa(paramsFile);
    
    cerr << "rank " << learningInfo->mpiWorld->rank() << ": before deserializing parameters, paramWeightsTemp.size() = " << paramWeightsTemp.size() << ", paramIdsTemp.size() = " << paramIdsTemp.size() << ", paramIndexes.size() = " << paramIndexes.size() << endl;

    oa >> *this;

    int index = 0;
    for(auto paramIdIter = paramIdsTemp.begin(); paramIdIter != paramIdsTemp.end(); ++paramIdIter, ++index) {
      paramIndexes[*paramIdIter] = index;
    }
   
    cerr << "rank " << learningInfo->mpiWorld->rank() << ": after deserializing parameters, paramWeightsTemp.size() = " << paramWeightsTemp.size() << ", paramIdsTemp.size() = " << paramIdsTemp.size() << ", paramIndexes.size() = " << paramIndexes.size() << endl;

    paramsFile.close();
    assert(paramIndexes.size() == paramWeightsTemp.size() && paramIndexes.size() == paramIdsTemp.size());
  }
  
}
void LogLinearParams::PrintFirstNParams(unsigned n) {
  assert(sealed);
  for (auto paramsIter = paramIndexes.begin(); n-- > 0 && paramsIter != paramIndexes.end(); paramsIter++) {
    cerr << paramsIter->first << " " << (*paramWeightsPtr)[paramsIter->second] << " at " << paramsIter->second << endl;
  }
}

// each line consists of: <featureStringId><space><featureWeight>\n
void LogLinearParams::PrintParams() {
  assert(paramIndexes.size() == paramWeightsPtr->size());
  PrintFirstNParams(paramIndexes.size());
}

void LogLinearParams::PrintParams(unordered_map_featureId_double &tempParams) {
  for(auto paramsIter = tempParams.begin(); paramsIter != tempParams.end(); paramsIter++) {
    cerr << paramsIter->first << " " << paramsIter->second << endl;
  }
}


// use gradient based methods to update the model parameter weights
void LogLinearParams::UpdateParams(const unordered_map_featureId_double &gradient, const OptMethod& optMethod) {
  assert(sealed);
  switch(optMethod.algorithm) {
  case OptAlgorithm::GRADIENT_DESCENT:
    for(auto gradientIter = gradient.begin(); gradientIter != gradient.end();
        gradientIter++) {
      // in case this parameter does not exist in paramWeights/paramIndexes
      AddParam(gradientIter->first);
      // update the parameter weight
      (*paramWeightsPtr)[ paramIndexes[gradientIter->first] ] -= optMethod.learningRate * gradientIter->second;
    }
    break;
  default:
    assert(false);
    break;
  }
}

// override the member weights vector with this array
void LogLinearParams::UpdateParams(const double* array, const int arrayLength) {
  assert(sealed);
  cerr << "##################" << endl;
  cerr << "pointer to internal weights: " << paramWeightsPtr->data() << ". pointer to external weights: " << array << endl;
  assert(arrayLength == paramWeightsPtr->size());
  assert(paramWeightsPtr->size() == paramIndexes.size());
  for(int i = 0; i < arrayLength; i++) {
    (*paramWeightsPtr)[i] = array[i];
  }
}

// converts a map into an array. 
// when constrainedFeaturesCount is non-zero, length(valuesArray)  should be = valuesMap.size() - constrainedFeaturesCount, 
// we pretend as if the constrained features don't exist by subtracting the internal index - constrainedFeaturesCount  
void LogLinearParams::ConvertFeatureMapToFeatureArray(
    unordered_map_featureId_double& valuesMap, double* valuesArray, 
    unsigned constrainedFeaturesCount) { 
  // init to 0 
  for(int i = constrainedFeaturesCount; i < paramIndexes.size(); i++) { 
    valuesArray[i-constrainedFeaturesCount] = 0; 
  } 
  // set the active features 
  for(auto valuesMapIter = valuesMap.begin(); valuesMapIter != valuesMap.end(); valuesMapIter++) { 
    // skip constrained features 
    if(paramIndexes[valuesMapIter->first] < constrainedFeaturesCount) { 
      continue; 
    } 
    // set the modified index in valuesArray 
    valuesArray[ paramIndexes[valuesMapIter->first]-constrainedFeaturesCount ] = valuesMapIter->second; 
  } 
} 

// 1/2 * sum of the squares 
double LogLinearParams::ComputeL2Norm() { 
  assert(sealed);
  double l2 = 0; 
  for(int i = 0; i < paramWeightsPtr->size(); i++) { 
    l2 += (*paramWeightsPtr)[i] * (*paramWeightsPtr)[i]; 
  } 
  return l2/2; 
} 

// checks whether the "otherParams" have the same parameters and values as this object 
// disclaimer: pretty expensive, and also requires that the parameters have the same order in the underlying vectors 
bool LogLinearParams::LogLinearParamsIsIdentical(const LogLinearParams &otherParams) { 
  if(IsSealed() != otherParams.IsSealed())
    return false;
  if(paramIndexes.size() != otherParams.paramIndexes.size()) 
    return false; 
  if(paramWeightsPtr->size() != otherParams.paramWeightsPtr->size()) 
    return false; 
  if(paramWeightsTemp.size() != otherParams.paramWeightsTemp.size()) 
    return false; 
  if(paramIdsPtr->size() != otherParams.paramIdsPtr->size())  
    return false; 
  if(paramIdsTemp.size() != otherParams.paramIdsTemp.size())  
    return false; 
  for(auto paramIndexesIter = paramIndexes.begin();  
      paramIndexesIter != paramIndexes.end(); 
      ++paramIndexesIter) { 
    auto otherIter = otherParams.paramIndexes.find(paramIndexesIter->first); 
    if(paramIndexesIter->second != otherIter->second)  
      return false; 
  } 
  for(unsigned i = 0; i < paramWeightsPtr->size(); i++){ 
    if((*paramWeightsPtr)[i] != (*otherParams.paramWeightsPtr)[i]) 
      return false; 
  } 
  for(unsigned i = 0; i < paramWeightsTemp.size(); i++){ 
    if(paramWeightsTemp[i] != otherParams.paramWeightsTemp[i]) 
      return false; 
  } 
  for(unsigned i = 0; i < paramIdsPtr->size(); i++) { 
    if((*paramIdsPtr)[i] != (*otherParams.paramIdsPtr)[i])  
      return false; 
  } 
  for(unsigned i = 0; i < paramIdsTemp.size(); i++) { 
    if(paramIdsTemp[i] != otherParams.paramIdsTemp[i])  
      return false; 
  } 
  return true; 
}

// side effect: adds zero weights for parameter IDs present in values but not present in paramIndexes and paramWeights
double LogLinearParams::DotProduct(const unordered_map_featureId_double& values) {
  if(!sealed) {
    return 0.0;
  }
  assert(sealed);
  double dotProduct = 0;
  // for each active feature
  for(auto valuesIter = values.begin(); valuesIter != values.end(); valuesIter++) {
    // make sure there's a corresponding feature in paramIndexes and paramWeights
    bool newParam = AddParam(valuesIter->first);
    // then update the dot product
    dotProduct += valuesIter->second * (*paramWeightsPtr)[paramIndexes[valuesIter->first]];
    if(std::isnan(dotProduct) || std::isinf(dotProduct)){
      cerr << "problematic param: " << valuesIter->first << " with index " << paramIndexes[valuesIter->first] << endl;
      cerr << "value = " << valuesIter->second << endl;
      cerr << "weight = " << (*paramWeightsPtr)[paramIndexes[valuesIter->first]] << endl;
      if(newParam) { cerr << "newParam." << endl; } else { cerr << "old param." << endl;}
      assert(false);
    }
  }
  return dotProduct;
}

double LogLinearParams::DotProduct(const FastSparseVector<double> &values, const ShmemVectorOfDouble& weights) {
  if(!sealed) {
    return 0.0;
  }
  assert(sealed);
  double dotProduct = 0;
  for(auto valuesIter = values.begin(); valuesIter != values.end(); ++valuesIter) {
    assert(ParamExists(valuesIter->first));
    dotProduct += valuesIter->second * weights[valuesIter->first];
  }
  return dotProduct;
}

double LogLinearParams::DotProduct(const FastSparseVector<double> &values) {
  return DotProduct(values, *paramWeightsPtr);
}

// compute dot product of two vectors
// assumptions:
// -both vectors are of the same size
double LogLinearParams::DotProduct(const std::vector<double>& values, const ShmemVectorOfDouble& weights) {
  if(!sealed) {
    return 0.0;
  }
  assert(sealed);
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
double LogLinearParams::DotProduct(const std::vector<double>& values) {
  return DotProduct(values, *paramWeightsPtr);
}
  
