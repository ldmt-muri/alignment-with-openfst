#include "LogLinearParams.h"

VocabEncoder* FeatureId::vocabEncoder = 0;

using namespace std;
using namespace boost;

std::ostream& operator<<(std::ostream& os, const FeatureId& obj)
{
  switch(obj.type) {
  case FeatureTemplate::BOUNDARY_LABELS:
    os << "BOUNDARY_LABELS";
    os << "|" << obj.boundaryLabel.position;
    os << "|" << obj.boundaryLabel.label;
    break;
  case FeatureTemplate::EMISSION:
    os << "EMISSION";
    os << '|' << obj.emission.displacement << "|" << obj.emission.label << "->" << FeatureId::vocabEncoder->Decode(obj.emission.word);
    break;
  case FeatureTemplate::LABEL_BIGRAM:
    os << "LABEL_BIGRAM";
    os << '|' << obj.bigram.previous << "->" << obj.bigram.current;
    break;
  case FeatureTemplate::SRC_BIGRAM:
    os << "SRC_BIGRAM";
    os << '|' << FeatureId::vocabEncoder->Decode(obj.bigram.previous) << "->" << FeatureId::vocabEncoder->Decode(obj.bigram.current);
    break;
  case FeatureTemplate::ALIGNMENT_JUMP:
    os << "ALIGNMENT_JUMP";
    os << '|' << obj.alignmentJump;
    break;
  case FeatureTemplate::LOG_ALIGNMENT_JUMP:
    os << "LOG_ALIGNMENT_JUMP";
    os << '|' << obj.biasedAlignmentJump.alignmentJump;
    os << '|' << FeatureId::vocabEncoder->Decode(obj.biasedAlignmentJump.wordBias);
    break;
  case FeatureTemplate::ALIGNMENT_JUMP_IS_ZERO:
    os << "ALIGNMENT_JUMP_IS_ZERO";
    os << '|' << obj.alignmentJump;
    break;
  case FeatureTemplate::SRC0_TGT0:
    os << "SRC0_TGT0";
    os << '|' << FeatureId::vocabEncoder->Decode(obj.wordPair.srcWord) << "->" << FeatureId::vocabEncoder->Decode(obj.wordPair.tgtWord);
    break;
  case FeatureTemplate::HEAD_CHILD_TOKEN:
    os << "HEAD_CHILD_TOKEN";
    os << '|' << FeatureId::vocabEncoder->Decode(obj.wordPair.srcWord) << "->" << FeatureId::vocabEncoder->Decode(obj.wordPair.tgtWord);
    break;
  case FeatureTemplate::HEAD_CHILD_POS:
    os << "HEAD_CHILD_POS";
    os << '|' << FeatureId::vocabEncoder->Decode(obj.wordPair.srcWord) << "->" << FeatureId::vocabEncoder->Decode(obj.wordPair.tgtWord);
    break;
  case FeatureTemplate::PRECOMPUTED:
    os << "PRECOMPUTED";
    os << '|' << FeatureId::vocabEncoder->Decode(obj.precomputed);
    break;
  case FeatureTemplate::DIAGONAL_DEVIATION:
    os << "DIAGONAL_DEVIATION";
    os << '|' << obj.wordBias;
    break;
  case FeatureTemplate::SRC_WORD_BIAS:
    os << "SRC_WORD_BIAS";
    os << '|' << FeatureId::vocabEncoder->Decode(obj.wordBias);
    break;
  case FeatureTemplate::HEAD_POS:
    os << "HEAD_POS";
    os << '|' << FeatureId::vocabEncoder->Decode(obj.wordBias);
    break;
  case FeatureTemplate::CHILD_POS:
    os << "CHILD_POS";
    os << '|' << FeatureId::vocabEncoder->Decode(obj.wordBias);
    break;
  case FeatureTemplate::HCX_POS:
    os << "HCX_POS";
    os << '|' << FeatureId::vocabEncoder->Decode(obj.wordBias);
    break;
  case FeatureTemplate::CHX_POS:
    os << "CHX_POS";
    os << '|' << FeatureId::vocabEncoder->Decode(obj.wordBias);
    break;
  case FeatureTemplate::XHC_POS:
    os << "XHC_POS";
    os << '|' << FeatureId::vocabEncoder->Decode(obj.wordBias);
    break;
  case FeatureTemplate::XCH_POS:
    os << "XCH_POS";
    os << '|' << FeatureId::vocabEncoder->Decode(obj.wordBias);
    break;
  case FeatureTemplate::HXC_POS:
    os << "HXC_POS";
    os << '|' << FeatureId::vocabEncoder->Decode(obj.wordBias);
    break;
  case FeatureTemplate::CXH_POS:
    os << "CXH_POS";
    os << '|' << FeatureId::vocabEncoder->Decode(obj.wordBias);
    break;
  case FeatureTemplate::SYNC_START:
    os << "SYNC_START";
    break;
  case FeatureTemplate::SYNC_END:
    os << "SYNC_END";
    break;
  case FeatureTemplate::OTHER_ALIGNERS:
    os << "OTHER_ALIGNERS";
    os << "|" << obj.otherAligner.alignerId << "|" << obj.otherAligner.compatible;
    break;
  case FeatureTemplate::NULL_ALIGNMENT:
    os << "NULL_ALIGNMENT";
    break;
  case FeatureTemplate::NULL_ALIGNMENT_LENGTH_RATIO:
    os << "NULL_ALIGNMENT_LENGTH_RATIO";
    break;
  default:
    assert(false);
  }
  return os;
}

LogLinearParams::LogLinearParams(VocabEncoder &types, 
                                 double gaussianStdDev) :
  types(types) {
  learningInfo = 0;
  gaussianSampler = new GaussianSampler(0.0, gaussianStdDev);
  FeatureId::vocabEncoder = &types;
  sealed = false;
  paramIdsPtr = 0;
  paramWeightsPtr = 0;
}

bool LogLinearParams::IsSealed() const {
  return sealed;
}

void* LogLinearParams::MapToSharedMemory(bool create, const string objectNickname) {
  if(string(objectNickname) == string("paramWeights")) {
    ShmemDoubleAllocator sharedMemoryDoubleAllocator(learningInfo->sharedMemorySegment->get_segment_manager()); 
    if(create) {
      auto ptr = learningInfo->sharedMemorySegment->find_or_construct<ShmemVectorOfDouble> (objectNickname.c_str()) (sharedMemoryDoubleAllocator);
      return ptr;
    } else {
      return learningInfo->sharedMemorySegment->find<ShmemVectorOfDouble> (objectNickname.c_str()).first;
    }

  } else if (string(objectNickname) == string("paramIds")) {
    ShmemFeatureIdAllocator sharedMemoryFeatureIdAllocator(learningInfo->sharedMemorySegment->get_segment_manager());
    if(create) {
      auto ptr = learningInfo->sharedMemorySegment->find_or_construct<ShmemVectorOfFeatureId> (objectNickname.c_str()) (sharedMemoryFeatureIdAllocator);
      return ptr;
    } else {
      return learningInfo->sharedMemorySegment->find<ShmemVectorOfFeatureId> (objectNickname.c_str()).first;
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

OuterMappedType* LogLinearParams::MapWordPairFeaturesToSharedMemory(bool create, const std::pair<int64_t, int64_t> &wordPair) {
  //cerr << "ttt0:" << wordPair.first << ","<<wordPair.second << endl;
  if(cacheWordPairFeatures.count(wordPair) == 0) {
    stringstream ss;
    ss << wordPair.first << " " << wordPair.second;
    string wordPairStr = ss.str();
    auto features = MapWordPairFeaturesToSharedMemory(create, wordPairStr);
    cacheWordPairFeatures[wordPair] = features;
    return features;
  } else {
    return cacheWordPairFeatures[wordPair];
  }
}

// disclaimer: may return zeros
OuterMappedType* LogLinearParams::MapWordPairFeaturesToSharedMemory(bool create, const string& objectNickname) {
  ShmemInnerValueAllocator sharedMemorySimpleMapAllocator(learningInfo->sharedMemorySegment->get_segment_manager()); 
  if(create) {
    try {
      auto temp = learningInfo->sharedMemorySegment->find< OuterMappedType > (objectNickname.c_str()).first;
      if(temp) {
        return temp;
      }
    } catch(std::exception const&  ex) {
      assert(false);
    }

    try {
      auto temp = learningInfo->sharedMemorySegment->construct< OuterMappedType > (objectNickname.c_str()) (std::less<InnerKeyType>(), sharedMemorySimpleMapAllocator);
      return temp;
    } catch(std::exception const& ex) {
      cerr << "sharedMemorySegment->construct( " << objectNickname << " ) threw " << ex.what() << endl;
      assert(false);
    }
  } else {
    try {
      auto temp = learningInfo->sharedMemorySegment->find< OuterMappedType > (objectNickname.c_str()).first;
      if(temp) { 
        return temp;
      } else {
        return 0;
      }
    } catch (std::exception const& ex) {
      cerr << "create == false, sharedMemorySegment->find( " << objectNickname << " ) threw " << ex.what() << endl;
      assert(false);
    }
  }
}

void LogLinearParams::Seal() {
  assert(!sealed);
  assert(paramIdsPtr == 0 && paramWeightsPtr == 0);
  if(learningInfo->mpiWorld->rank() == 0) {
    paramWeightsPtr = (ShmemVectorOfDouble *) MapToSharedMemory(true, "paramWeights");
    assert(paramWeightsPtr != 0);
    paramIdsPtr = (ShmemVectorOfFeatureId *) MapToSharedMemory(true, "paramIds");
    assert(paramIdsPtr != 0);
    
    // sync
    bool dummy = true;
    boost::mpi::broadcast<bool>(*learningInfo->mpiWorld, dummy, 0);
    
    // copy paramIdsTemp and paramWeightsTemp, and wipe off temporary parameters you had
    assert(paramWeightsTemp.size() == paramIdsTemp.size());
    for(unsigned i = 0; i < paramWeightsTemp.size(); ++i) {
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

    // sync
    bool dummy = true;
    boost::mpi::broadcast<bool>(*learningInfo->mpiWorld, dummy, 0);

    // map paramIds and paramWeights to shared memory
    paramWeightsPtr = (ShmemVectorOfDouble *)MapToSharedMemory(false, "paramWeights");
    paramIdsPtr = (ShmemVectorOfFeatureId *)MapToSharedMemory(false, "paramIds");
  }
  assert(paramIdsPtr != 0 && paramWeightsPtr != 0);
  sealed = true;
}

void LogLinearParams::Unseal() {
  assert(sealed);
  assert(paramWeightsTemp.size() == 0);
  assert(paramIdsTemp.size() == 0);
  paramIndexes.clear();
  if(learningInfo->mpiWorld->rank() == 0) {
    // sync
    double dummy=1.0;
    boost::mpi::all_reduce<double>(*learningInfo->mpiWorld, dummy, dummy, std::plus<double>());
    // now, that all processes reached this method, you can delete the shared objects
    paramIdsPtr = 0;
    paramWeightsPtr = 0;
  } else {
    // this is done by the slaves, not the master
    // sync
    double dummy = 0.5;
    boost::mpi::all_reduce<double>(*learningInfo->mpiWorld, dummy, dummy, std::plus<double>());
    paramWeightsPtr = 0;
    paramIdsPtr = 0;
  }
  assert(paramIdsPtr == 0 && paramWeightsPtr == 0);
  sealed = false;
}

// by two inputs, i mean that a precomputed feature value is a function of two strings
// example line in the precomputed features file:
// madrasa ||| school ||| F52:editdistance=7 F53:capitalconsistency=1
void LogLinearParams::LoadPrecomputedFeaturesWith2Inputs(const string &wordPairFeaturesFilename) {
  assert(learningInfo->mpiWorld->rank() == 0);

  cerr << "rank " << learningInfo->mpiWorld->rank() << " is going to read the word pair features file..." << endl;
  ifstream wordPairFeaturesFile(wordPairFeaturesFilename.c_str(), ios::in);
  string line;
  
  while( getline(wordPairFeaturesFile, line) ) {
    if(line.size() == 0) {
      continue;
    }
    //cerr << line << endl;
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
    std::pair<int64_t, int64_t> srcTgtPair(input1, input2);
    //cerr << "ttt-2" << endl;
    auto tempMap = MapWordPairFeaturesToSharedMemory(true, srcTgtPair);
    assert(tempMap);
    // the remaining elements are precomputed features for (input1, input2)
    while(splitsIter != splits.end()) {
      // read feature id
      std::vector<string> featureIdAndValue;
      StringUtils::SplitString(*(splitsIter++), '=', featureIdAndValue);
      if(featureIdAndValue.size() != 2) {
        cerr << "UH-OH: featureIdAndValue.size() == " << featureIdAndValue.size() << endl;
        cerr << "elements:" << endl; 
        for(auto field = featureIdAndValue.begin(); field != featureIdAndValue.end(); field++) {
          cerr << *field << endl;
        }
      }
      assert(featureIdAndValue.size() == 2);
      FeatureId featureId;
      featureId.type = FeatureTemplate::PRECOMPUTED;
      featureId.precomputed = types.Encode(featureIdAndValue[0]);

      // read feature value
      double featureValue;
      stringstream temp;
      temp << featureIdAndValue[1];
      temp >> featureValue;
      
      try {
        tempMap->insert( std::pair<FeatureId, double>( featureId, featureValue ));
       
      } catch(std::exception const&  ex) {
        cerr << "tempMap->insert( pair (" << featureId << ", " << featureValue << ") ) threw exception: "  << ex.what() << endl;
        assert(false);
      }
    }
  }
  
  wordPairFeaturesFile.close();
  cerr << "rank " << learningInfo->mpiWorld->rank() << " finished reading the word pair features file." << endl;
  
}

void LogLinearParams::SetLearningInfo(LearningInfo &learningInfo) {
  this->learningInfo = &learningInfo;

  // load word alignments of other aligners
  LoadOtherAlignersOutput();
}

void LogLinearParams::LoadOtherAlignersOutput() {
  // each process independently reads the output of other word aligners
  if(learningInfo->otherAlignersOutputFilenames.size() > 0) {
    for(auto filenameIter = learningInfo->otherAlignersOutputFilenames.begin();
        filenameIter != learningInfo->otherAlignersOutputFilenames.end();
        ++filenameIter) {
      cerr << "aligner filename: " << *filenameIter << endl;
      auto alignerOutput = new vector< vector< set<int>* >* >();
      //cerr << "adding this aligner to the list of aligners" << endl;
      otherAlignersOutput.push_back(alignerOutput);
      std::ifstream infile(filenameIter->c_str());
      std::string line;
      while (std::getline(infile, line)) {
        auto sentAlignments = new vector< set<int>* >();
        //cerr << "adding this sentence to the list of sentences for this aligner" << endl;
        alignerOutput->push_back(sentAlignments);
        // each line consists of a number of word-to-word alignments
        vector<std::string> srcpos_tgtpos_pairs;
        StringUtils::SplitString(line, ' ', srcpos_tgtpos_pairs);
        for(auto pairIter = srcpos_tgtpos_pairs.begin(); 
            pairIter != srcpos_tgtpos_pairs.end();
            ++pairIter) {
          // read this pair
          int srcpos, tgtpos; 
          char del;
          std::istringstream ss(*pairIter);
          ss >> srcpos >> del >> tgtpos;
          assert(del == '-');
          // if this is a 'reverse' training, swap srcpos with tgtpos
          if(learningInfo->reverse) {
            int temp = srcpos;
            srcpos = tgtpos;
            tgtpos = temp;
          }
          // increment srcpos because we insert the NULL src word at the beginning of each sentence
          srcpos++;
          // make room in the sentAlignments vector for this pair. tgt positions not mentioned are aligned to NULL
          assert(tgtpos >= 0);
          while(sentAlignments->size() <= (unsigned) tgtpos) { 
            auto tgtPosAlignments = new set<int>();
            sentAlignments->push_back(tgtPosAlignments); 
          }
          // memorize this pair
          //cerr << "adding this word pair to this sentence" << endl;
          (*sentAlignments)[tgtpos]->insert(srcpos);
          //cerr << "tgtpos=" << tgtpos << " aligns to srcpos=" << srcpos << endl;
        }
        //cerr << endl;
      }
    }
  }
}

int LogLinearParams::AddParams(const std::vector< FeatureId > &paramIds) {
  paramIndexes.reserve(paramIndexes.size() + paramIds.size());
  paramIdsTemp.reserve(paramIdsTemp.size() + paramIds.size());
  paramWeightsTemp.reserve(paramWeightsTemp.size() + paramIds.size());
  int newParamsCount = 0;
  for(unsigned i = 0; i < paramIds.size(); ++i) {
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
    
    if(sealed) {
      cerr << "trying to add the followign paramId after object is sealed: " << paramId << endl;
      throw LogLinearParamsException("adding new parameter after object is sealed!");
    }
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

// for dependency parsing
void LogLinearParams::FireFeatures(const ObservationDetails &headDetails, 
                                   const ObservationDetails &childDetails,
                                   const vector<ObservationDetails> & sentDetails,
                                   FastSparseVector<double> &activeFeatures) {
  FeatureId featureId;

  unsigned earlierIndex = min(headDetails.details[ObservationDetailsHeader::ID]-1, 
                              childDetails.details[ObservationDetailsHeader::ID]-1);
  unsigned laterIndex = max(headDetails.details[ObservationDetailsHeader::ID]-1, 
                            childDetails.details[ObservationDetailsHeader::ID]-1);
  

  for(auto featTemplateIter = learningInfo->featureTemplates.begin();
      featTemplateIter != learningInfo->featureTemplates.end(); ++featTemplateIter) {
    
    switch(*featTemplateIter) {
    case FeatureTemplate::HEAD_CHILD_TOKEN:
      featureId.type = FeatureTemplate::HEAD_CHILD_TOKEN;
      featureId.wordPair.srcWord = headDetails.details[ObservationDetailsHeader::FORM];
      featureId.wordPair.tgtWord = childDetails.details[ObservationDetailsHeader::FORM];  
      AddParam(featureId);
      activeFeatures[paramIndexes[featureId]] += 1.0;
      break;
      
    case FeatureTemplate::HEAD_CHILD_POS:
      featureId.type = FeatureTemplate::HEAD_CHILD_POS;
      featureId.wordPair.srcWord = headDetails.details[ObservationDetailsHeader::CPOSTAG];
      featureId.wordPair.tgtWord = childDetails.details[ObservationDetailsHeader::CPOSTAG];
      AddParam(featureId);
      activeFeatures[paramIndexes[featureId]] += 1.0;
      break;
      
    case FeatureTemplate::HEAD_POS:
      featureId.type = FeatureTemplate::HEAD_POS;
      featureId.wordBias = headDetails.details[ObservationDetailsHeader::CPOSTAG];
      AddParam(featureId);
      activeFeatures[paramIndexes[featureId]] += 1.0;
      break;
      
    case FeatureTemplate::CHILD_POS:
      featureId.type = FeatureTemplate::CHILD_POS;
      featureId.wordBias = headDetails.details[ObservationDetailsHeader::CPOSTAG];
      AddParam(featureId);
      activeFeatures[paramIndexes[featureId]] += 1.0;
      break;
      
      // inbetween
    case FeatureTemplate::CXH_POS:
    case FeatureTemplate::HXC_POS:
      if(headDetails.details[ObservationDetailsHeader::ID] == 0) break;
      featureId.type = headDetails.details[ObservationDetailsHeader::ID] < childDetails.details[ObservationDetailsHeader::ID]?
        FeatureTemplate::HXC_POS: FeatureTemplate::CXH_POS;
      if(featureId.type != *featTemplateIter) break;
      for(unsigned inbetweenIndex = 1 + earlierIndex; inbetweenIndex < laterIndex; ++inbetweenIndex) {
        assert(inbetweenIndex >= 0 && inbetweenIndex < sentDetails.size());
        assert(ObservationDetailsHeader::CPOSTAG < sentDetails[inbetweenIndex].details.size());
        featureId.wordBias = sentDetails[inbetweenIndex].details[ObservationDetailsHeader::CPOSTAG];
        AddParam(featureId);
        activeFeatures[paramIndexes[featureId]] += 1.0;
      }
      break;
      
      // outside to the left 
    case FeatureTemplate::XHC_POS:
    case FeatureTemplate::XCH_POS:
      if(headDetails.details[ObservationDetailsHeader::ID] == 0) {
        break;
      }
      featureId.type = headDetails.details[ObservationDetailsHeader::ID] < childDetails.details[ObservationDetailsHeader::ID]?
        FeatureTemplate::XHC_POS: FeatureTemplate::XCH_POS;
      if(featureId.type != *featTemplateIter) break;
      featureId.wordBias = 
          earlierIndex == 0?
        -1:
        sentDetails[earlierIndex-1].details[ObservationDetailsHeader::CPOSTAG];
      AddParam(featureId);
      activeFeatures[paramIndexes[featureId]] += 1.0;
      break;

      // outside to the right 
    case FeatureTemplate::CHX_POS:
    case FeatureTemplate::HCX_POS:
      if(headDetails.details[ObservationDetailsHeader::ID] == 0) break;
      featureId.type = headDetails.details[ObservationDetailsHeader::ID] < childDetails.details[ObservationDetailsHeader::ID]?
        FeatureTemplate::HCX_POS: FeatureTemplate::CHX_POS;
      if(featureId.type != *featTemplateIter) break;
      featureId.wordBias = 
        laterIndex == sentDetails.size() - 1?
        -1:
        sentDetails[laterIndex+1].details[ObservationDetailsHeader::CPOSTAG];
      AddParam(featureId);
      activeFeatures[paramIndexes[featureId]] += 1.0;
      break;
      
    // log alignment jump (two versions below, with and without conjoining the head pos tag)
    case FeatureTemplate::LOG_ALIGNMENT_JUMP:
      if(headDetails.details[ObservationDetailsHeader::ID] == 0) break; 
      featureId.type = FeatureTemplate::LOG_ALIGNMENT_JUMP;
      featureId.biasedAlignmentJump.alignmentJump = 
        headDetails.details[ObservationDetailsHeader::ID] < childDetails.details[ObservationDetailsHeader::ID]?
        log(1 + 2.0 * (laterIndex - earlierIndex)):
        -1 * log(1 + 2.0 * (laterIndex - earlierIndex));
      
      // biased version:
      //featureId.biasedAlignmentJump.wordBias = headDetails.details[ObservationDetailsHeader::CPOSTAG];
      //AddParam(featureId);
      //activeFeatures[paramIndexes[featureId]] += 1.0;
      
      // unbiased version:
      featureId.biasedAlignmentJump.wordBias = -1;
      AddParam(featureId);
      activeFeatures[paramIndexes[featureId]] += 1.0;
      break;
      
      // alignment jump
    case FeatureTemplate::ALIGNMENT_JUMP:
      if(headDetails.details[ObservationDetailsHeader::ID] == 0) break;
      featureId.type = FeatureTemplate::ALIGNMENT_JUMP;
      featureId.alignmentJump = headDetails.details[ObservationDetailsHeader::ID] - childDetails.details[ObservationDetailsHeader::ID];
      AddParam(featureId);
      activeFeatures[paramIndexes[featureId]] += 1.0;
      break;
      
    default:
      cerr << "feature not implemented for dependency parsing: " << *featTemplateIter << endl;
      assert(false);
    }
  }
}

// for word alignment
// x_t is the tgt sentence, and x_s is the src sentence (which has a null token at position 0)
void LogLinearParams::FireFeatures(int yI, int yIM1, const vector<int64_t> &x_t, const vector<int64_t> &x_s, unsigned i, 
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
  auto srcToken = x_s[yI];
  auto prevSrcToken = yIM1 >= 0? x_s[yI] : START_OF_SENTENCE_Y_VALUE;
  auto tgtToken = x_t[i];
  //auto prevTgtToken = i > 0? x_t[i-1] : -1;
  //auto nextTgtToken = (i < x_t.size() - 1)? x_t[i+1] : (int64_t) -1;
  std::pair<int64_t, int64_t> srcTgtPair(srcToken, tgtToken);
  auto precomputedFeatures = MapWordPairFeaturesToSharedMemory(false, srcTgtPair);
  
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
      featureId.biasedAlignmentJump.alignmentJump = 
        yI >= yIM1? 
        log(1 + 2.0 * (yI - yIM1)):
        -1 * log(1 + 2.0 * (yIM1 - yI));
      // biased version:
      featureId.biasedAlignmentJump.wordBias = srcToken;
      AddParam(featureId);
      activeFeatures[paramIndexes[featureId]] += 1.0;
      // unbiased version:
      featureId.biasedAlignmentJump.wordBias = -1;
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
            try {
              AddParam(precomputedIter->first);
            } catch (LogLinearParamsException &ex) {
              cerr << "LogLinearParamsException " << ex.what() << " -- thrown at LogLinearParams::FireFeatures(int yI, int yIM1, const vector<int64_t> &x_t, const vector<int64_t> &x_s, int i, int START_OF_SENTENCE_Y_VALUE, int FIRST_POS, FastSparseVector<double> &activeFeatures) where yI = " << yI << ", yIM1 = " << yIM1 << ", i = " << i << ", x_t[i] = " << x_t[i] << ", x_s[yI] = " << x_s[yI] << ", types.Decode(x_t[i]) = " << types.Decode(x_t[i]) << ", types.Decode(x_s[yI]) = " << types.Decode(x_s[yI]) << ", precomputedFeatures->size() = " << precomputedFeatures->size() << ", precomputedIter->first = " << precomputedIter->first << ", precomputedIter->second = " << precomputedIter->second << ", types.Decode(precomputedIter->first.precomputed) = " << types.Decode(precomputedIter->first.precomputed) << endl;
              throw;
            }
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
        featureId.wordBias = srcToken;
        double deviation = (yI > 0)?
          fabs(1.0 * (yI-1) / (x_s.size()-1) - 1.0 * i / x_t.size()):
          fabs(1.0 * (yIM1-1) / (x_s.size()-1) - 1.0 * i / x_t.size());
        AddParam(featureId);
        activeFeatures[paramIndexes[featureId]] += deviation;

        // this feature is not specific to the srcToken
        featureId.wordBias = -1; 
        AddParam(featureId);
        activeFeatures[paramIndexes[featureId]] += deviation;
      }
      break;

      case FeatureTemplate::SRC_WORD_BIAS:
      featureId.type = FeatureTemplate::SRC_WORD_BIAS;
      featureId.wordBias = srcToken;
      AddParam(featureId);
      activeFeatures[paramIndexes[featureId]] += 1.0;
      break;

      case FeatureTemplate::SYNC_START:
      if(i == 0 && yI == 1) {
        featureId.type = FeatureTemplate::SYNC_START;
        AddParam(featureId);
        activeFeatures[paramIndexes[featureId]] += 1.0;
      }
      break;
      
      case FeatureTemplate::SYNC_END:
        if(i == x_t.size() - 1 && yI == (int)x_s.size() - 1) {
        featureId.type = FeatureTemplate::SYNC_END;
        AddParam(featureId);
        activeFeatures[paramIndexes[featureId]] += 1.0;
      }
      break;
    
      case FeatureTemplate::EMISSION:
      cerr << "this feature template is not implemented for word alignment" << endl;
      assert(false);
      break;

      case FeatureTemplate::LABEL_BIGRAM:
      cerr << "this feature template is not implemented for word alignment" << endl;
      assert(false);
      break;

    case FeatureTemplate::OTHER_ALIGNERS:
      for(unsigned alignerId = 0; alignerId < otherAlignersOutput.size(); alignerId++) {
        assert(learningInfo->currentSentId < (int)otherAlignersOutput[alignerId]->size());
	if( (*(*otherAlignersOutput[alignerId])[learningInfo->currentSentId]).size() <= i ) {continue;}
	auto woodAlignments = (*(*otherAlignersOutput[alignerId])[learningInfo->currentSentId])[i];
	featureId.type = FeatureTemplate::OTHER_ALIGNERS;
	featureId.otherAligner.compatible = woodAlignments->count(yI) == 1 || \
	  (yI == 0 && woodAlignments->size() == 0);
	featureId.otherAligner.alignerId = alignerId;
	AddParam(featureId);
	activeFeatures[paramIndexes[featureId]] += 1.0;
      }
      break;

    case FeatureTemplate::NULL_ALIGNMENT:
      if(yI == 0) {
	featureId.type = FeatureTemplate::NULL_ALIGNMENT;
	AddParam(featureId);
	activeFeatures[paramIndexes[featureId]] += 1.0;
      }
      break;

    case FeatureTemplate::NULL_ALIGNMENT_LENGTH_RATIO:
      if(yI == 0) {
	featureId.type = FeatureTemplate::NULL_ALIGNMENT_LENGTH_RATIO;
	AddParam(featureId);
	activeFeatures[paramIndexes[featureId]] += 1.0 * (x_t.size() - x_s.size() - 1) / x_t.size();
      }
      break;

    default:
      assert(false);
    } // end of switch
  } // end of loop over enabled feature templates
}

// for pos induction
// features for the latent crf model
void LogLinearParams::FireFeatures(int yI, int yIM1, const vector<int64_t> &x, unsigned i, 
				   FastSparseVector<double> &activeFeatures) {
  
  const int64_t &xI = x[i];
  const int64_t &xIM1 = i-1 >= 0? x[i-1] : -1;
  const int64_t &xIM2 = i-2 >= 0? x[i-2] : -1;
  const int64_t &xIP1 = i+1 < x.size()? x[i+1] : -1;
  const int64_t &xIP2 = i+2 < x.size()? x[i+2] : -1; 

  // return cached features for this factor id
  PosFactorId factorId;
  if(learningInfo->cacheActiveFeatures) {
    factorId.yI = yI;
    factorId.yIM1 = yIM1;
    factorId.xIM2 = xIM2;
    factorId.xIM1 = xIM1;
    factorId.xI = xI;
    factorId.xIP1 = xIP1;
    factorId.xIP2 = xIP2;
    if(posFactorIdToFeatures.count(factorId) == 1) {
      activeFeatures = posFactorIdToFeatures[factorId];
      // logging
      //factorId.Print();
      //cerr << "activeFeatures.size() = " << activeFeatures.size() << endl;
      return;
    }
  }
  
  FeatureId featureId;

  std::vector<int> kValues;
 
  for(auto featTemplateIter = learningInfo->featureTemplates.begin();
      featTemplateIter != learningInfo->featureTemplates.end(); 
      ++featTemplateIter) {
    
    featureId.type = *featTemplateIter;
    
    switch(featureId.type) {

    case FeatureTemplate::PRECOMPUTED:
      // override the feature type because we need to conjoin the precomputed feature with label id in pos tagging
      featureId.type = FeatureTemplate::EMISSION;
      // set the conjoined label
      featureId.emission.label = yI;
      
      // a moving window of tokens y[i]:Precomputed(x[i+k])
      kValues.clear();
      if(learningInfo->firePrecomputedFeaturesForXIM2) { kValues.push_back(-2); }
      if(learningInfo->firePrecomputedFeaturesForXIM1) { kValues.push_back(-1); }
      if(learningInfo->firePrecomputedFeaturesForXI) { kValues.push_back(0); }
      if(learningInfo->firePrecomputedFeaturesForXIP1) { kValues.push_back(1); }
      if(learningInfo->firePrecomputedFeaturesForXIP2) { kValues.push_back(2); }
      for(auto kIter = kValues.begin(); kIter != kValues.end(); ++kIter) {
        int k = *kIter;
        std::pair<int64_t, int64_t> wordPair(k==-2? xIM2: k==-1? xIM1: k==0? xI: k==1? xIP1: xIP2,
                                             k==-2? xIM2: k==-1? xIM1: k==0? xI: k==1? xIP1: xIP2);
        if(wordPair.first == -1) { continue; }
        
        auto precomputedFeatures = MapWordPairFeaturesToSharedMemory(false, wordPair);
        if(!precomputedFeatures) { continue; }

        // set the relative position of this token to the label being considered
        featureId.emission.displacement = k;

        // now, for each precomputed feature of this token:
        for(auto precomputedIter = precomputedFeatures->begin();
            precomputedIter != precomputedFeatures->end();
            precomputedIter++) {
          // now set the emission.word field to the precomputed feature. 
          // TODO-REFACTOR: this is a misuse of the field names.
          featureId.emission.word = precomputedIter->first.precomputed;
          // now, all necessary fields of this featureId has been set
          try {
            AddParam(featureId);
          } catch(LogLinearParamsException &ex) {
            cerr << "been here"<< endl;
            throw;
          }
          activeFeatures[paramIndexes[featureId]] += precomputedIter->second;
        }
      }
    
      break;

      case FeatureTemplate::LABEL_BIGRAM:
        featureId.bigram.current = yI;
        featureId.bigram.previous = yIM1;
        AddParam(featureId);
        activeFeatures[paramIndexes[featureId]] += 1.0;
        // label bias features
        //featureId.bigram.previous = -1;
        //AddParam(featureId);
        //activeFeatures[paramIndexes[featureId]] += 1.0;
      break;

      case FeatureTemplate::EMISSION:
        featureId.emission.label = yI;
        // y[i]:x[i+k]
        kValues.clear();
        kValues.push_back(0);
        //kValues.push_back(1);
        //kValues.push_back(2);
        for(auto kIter = kValues.begin(); kIter != kValues.end(); ++kIter) {
          int k = *kIter;
          featureId.emission.word = k==-2? xIM2: k==-1? xIM1: k==0? xI: k==1? xIP1: xIP2;
          featureId.emission.displacement = k;
          AddParam(featureId);
          activeFeatures[paramIndexes[featureId]] += 1.0;
        }
      break;

      case FeatureTemplate::OTHER_ALIGNERS:
      for(unsigned alignerId = 0; alignerId < otherAlignersOutput.size(); alignerId++) {
        assert(learningInfo->currentSentId < (int)otherAlignersOutput[alignerId]->size());
        if( (*(*otherAlignersOutput[alignerId])[learningInfo->currentSentId]).size() <= i ) {
          continue;
        }
        auto woodAlignments = (*(*otherAlignersOutput[alignerId])[learningInfo->currentSentId])[i];
        featureId.type = FeatureTemplate::OTHER_ALIGNERS;
        featureId.otherAligner.compatible = woodAlignments->count(yI) == 1 || \
          (yI == 0 && woodAlignments->size() == 0);
        featureId.otherAligner.alignerId = alignerId;
        AddParam(featureId);
        activeFeatures[paramIndexes[featureId]] += 1.0;
      }
      break;

    case FeatureTemplate::BOUNDARY_LABELS:
      if(i <= 0 || i >= x.size() - 1) {
        featureId.type = FeatureTemplate::BOUNDARY_LABELS;
        featureId.boundaryLabel.position = i < 2? i : i - x.size(); 
        featureId.boundaryLabel.label = yI;
        AddParam(featureId);
        activeFeatures[paramIndexes[featureId]] += 1.0;
      }
      break;
      
    default:
      assert(false);
      
    }
  }
  
  // save the active features in the cache
  if(learningInfo->cacheActiveFeatures) {
    assert(posFactorIdToFeatures.count(factorId) == 0);
    posFactorIdToFeatures[factorId] = activeFeatures;
    if(posFactorIdToFeatures.size() % 1000000 == 0) {
      cerr << "|factorIds| is now " << posFactorIdToFeatures.size() << endl;
    } 
    // logging
    //factorId.Print();
    //cerr << "activeFeatures.size() = " << activeFeatures.size() << endl;
  }
}


double LogLinearParams::Hash() {
  assert(sealed);
  double hash = 0.0;
  for(auto paramIter = paramWeightsPtr->begin(); paramIter != paramWeightsPtr->end(); paramIter++) {
    hash += *paramIter;
  }
  return hash;
}

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
  assert((unsigned)arrayLength == paramWeightsPtr->size());
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
  for(unsigned i = constrainedFeaturesCount; i < paramIndexes.size(); i++) { 
    valuesArray[i-constrainedFeaturesCount] = 0; 
  } 
  // set the active features 
  for(auto valuesMapIter = valuesMap.begin(); valuesMapIter != valuesMap.end(); valuesMapIter++) { 
    // skip constrained features 
    if(paramIndexes[valuesMapIter->first] < (int)constrainedFeaturesCount) { 
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
  for(unsigned i = 0; i < paramWeightsPtr->size(); i++) { 
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
  for(unsigned i = 0; i < values.size(); i++) {
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
  
