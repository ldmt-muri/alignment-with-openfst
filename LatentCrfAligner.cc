#include "LatentCrfAligner.h"

// singleton
LatentCrfModel& LatentCrfAligner::GetInstance(const string &textFilename, 
					      const string &outputPrefix, 
					      LearningInfo &learningInfo, 
					      unsigned FIRST_LABEL_ID) {
  if(!LatentCrfAligner::instance) {
    LatentCrfAligner::instance = new LatentCrfAligner(textFilename, outputPrefix, learningInfo, FIRST_LABEL_ID);
  }
  return *LatentCrfAligner::instance;
}

LatentCrfModel& LatentCrfAligner::GetInstance() {
  if(!instance) {
    assert(false);
  }
  return *instance;
}

LatentCrfAligner::LatentCrfAligner(const string &textFilename,
				   const string &outputPrefix,
				   LearningInfo &learningInfo,
				   unsigned FIRST_LABEL_ID) : LatentCrfModel(textFilename,
									     outputPrefix,
									     learningInfo,
									     FIRST_LABEL_ID) {

  // set constants
  this->START_OF_SENTENCE_Y_VALUE = FIRST_LABEL_ID - 1;
  this->FIRST_ALLOWED_LABEL_VALUE = FIRST_LABEL_ID;
  this->NULL_POSITION = FIRST_LABEL_ID;
  this->FIRST_SRC_POSITION = FIRST_LABEL_ID + 1;
  assert(START_OF_SENTENCE_Y_VALUE > 0);

  // unlike POS tagging, yDomain depends on the src sentence length. we will set it on a per-sentence basis.
  this->yDomain.clear();

  // populate zDomain (tgt language vocabulary) and x_sDomain (src language vocabulary), by reading the parallel-data file
  std::ifstream textFile(textFilename.c_str(), std::ios::in);
  std::string line;
  while(getline(textFile, line)) {
    // skip empty lines
    if(line.size() == 0) {
      continue;
    }
    // convert "the src sent ||| the tgt sent" into one string vector
    std::vector<string> splits;
    StringUtils::SplitString(line, ' ', splits);
    // each line starts with the source sentence
    bool src = true;
    // for each token in the line
    for(std::vector<string>::const_iterator tokenIter = splits.begin(); 
	tokenIter != splits.end();
	tokenIter++) {
      if(*tokenIter == "|||") {
	// then the target sentence
	src = false;
	continue;
      }
      // either add it to source vocab or target vocab
      if(src) {
	x_sDomain.insert(vocabEncoder[*tokenIter]);
      } else {
	zDomain.insert(vocabEncoder[*tokenIter]);
      }
    }
  }
  
  // zero is reserved for FST epsilon (I don't think this is a problem since we don't use the x_i nor the z_i as fst labels)
  assert(this->x_sDomain.count(0) == 0 && this->zDomain.count(0) == 0);

  // encode the null token which is conventionally added to the beginning of the src sentnece. 
  NULL_TOKEN_STR = "__null__token__";
  bool explicitUseUnk = false;
  NULL_TOKEN = vocabEncoder.Encode(NULL_TOKEN_STR, explicitUseUnk);
  assert(NULL_TOKEN != vocabEncoder.UnkInt);

  // read and encode data
  srcSents.clear();
  tgtSents.clear();
  vocabEncoder.ReadParallelCorpus(textFilename, srcSents, tgtSents, NULL_TOKEN);

  // bool vectors indicating which feature types to use
  assert(enabledFeatureTypes.size() == 0);
  // features 1-50 are reserved for wordalignment
  for(int i = 0; i <= 50; i++) {
    enabledFeatureTypes.push_back(false);
  }
  // features 51-100 are reserved for latentCrfPosTagger model
  for(int i = 51; i < 100; i++) {
    enabledFeatureTypes.push_back(false);
  }
  // features 101-150 are reserved for latentCrfAligner model
  for(int i = 51; i < 100; i++) {
    enabledFeatureTypes.push_back(false);
  }
  enabledFeatureTypes[101] = true;  // I( y_i-y_{i-1} == 0 )
  enabledFeatureTypes[102] = true;  // I( floor( ln(y_i - y_{i-1}) ) )
  enabledFeatureTypes[103] = true;  // I( tgt[i] aligns_to src[y_i] )

  // initialize (and normalize) the log theta params to gaussians
  InitTheta();

  // make sure all slaves have the same theta values
  BroadcastTheta(0);

  // persist initial parameters
  assert(learningInfo.iterationsCount == 0);
  if(learningInfo.iterationsCount % learningInfo.persistParamsAfterNIteration == 0 && learningInfo.mpiWorld->rank() == 0) {
    stringstream thetaParamsFilename;
    thetaParamsFilename << outputPrefix << ".initial.theta";
    PersistTheta(thetaParamsFilename.str());
  }
  
  // initialize the lambda parameters
  // add all features in this data set to lambda.params
  InitLambda();
}

void LatentCrfAligner::InitTheta() {

  if(learningInfo.mpiWorld->rank() == 0 && learningInfo.debugLevel >= DebugLevel::CORPUS) {
    cerr << "master" << learningInfo.mpiWorld->rank() << ": initializing thetas...";
  }

  // this feature of the model is not supported for the word alignment model yet
  assert(!learningInfo.zIDependsOnYIM1);

  assert(srcSents.size() == tgtSents.size());

  // first initialize nlogthetas to unnormalized gaussians
  nLogThetaGivenOneLabel.params.clear();
  for(unsigned sentId = 0; sentId < srcSents.size(); ++sentId) {
    vector<int> &srcSent = srcSents[sentId];
    vector<int> &tgtSent = tgtSents[sentId];
    for(unsigned i = 0; i < srcSent.size(); ++i) {
      int srcToken = srcSent[i];
      for(unsigned j = 0; j < tgtSent.size(); ++j) {
	int tgtToken = tgtSent[j];
	nLogThetaGivenOneLabel.params[srcToken][tgtToken] = abs(gaussianSampler.Draw());
      }
    }
  }

  // then normalize them
  MultinomialParams::NormalizeParams(nLogThetaGivenOneLabel);
  
  if(learningInfo.mpiWorld->rank() == 0) {
    cerr << "done" << endl;
  }
}

void LatentCrfAligner::PrepareExample(unsigned exampleId) {
  yDomain.clear();
  // this length includes the null token that was inserted at the begging of all source sentences
  srcSentLength = srcSents[exampleId].size();
  // each position in the src sentence, including null, should have an entry in yDomain
  for(unsigned i = NULL_POSITION; i < NULL_POSITION + srcSentLength; ++i) {
    yDomain.insert(i);
  }
}

vector<int> LatentCrfAligner::GetObservableSequence(int exampleId) {
  if(testingMode) {
    assert(exampleId < testTgtSents.size());
    return testTgtSents[exampleId];
  } else {
    assert(exampleId < tgtSents.size());
    return tgtSents[exampleId];
  }
}

vector<int>& LatentCrfAligner::GetObservableContext(int exampleId) { 
  if(testingMode) {
    assert(exampleId < testSrcSents.size());
    return testSrcSents[exampleId];
  } else {
    assert(exampleId < srcSents.size());
    return srcSents[exampleId];
  }   
}
