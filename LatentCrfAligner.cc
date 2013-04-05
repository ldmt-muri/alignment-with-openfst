#include "LatentCrfAligner.h"

string LatentCrfAligner::NULL_TOKEN_STR = "REDICULOUS";
int LatentCrfAligner::NULL_TOKEN = -1000000;
unsigned LatentCrfAligner::FIRST_SRC_POSITION =  100000;

// singleton
LatentCrfModel* LatentCrfAligner::GetInstance(const string &textFilename, 
					      const string &outputPrefix, 
					      LearningInfo &learningInfo, 
					      unsigned FIRST_LABEL_ID,
					      const string &initialLambdaParamsFilename, 
					      const string &initialThetaParamsFilename,
					      const string &wordPairFeaturesFilename) {
  if(!instance) {
    instance = new LatentCrfAligner(textFilename, 
				    outputPrefix,
				    learningInfo, 
				    FIRST_LABEL_ID, 
				    initialLambdaParamsFilename, 
				    initialThetaParamsFilename,
				    wordPairFeaturesFilename);
  }
  return instance;
}

LatentCrfModel* LatentCrfAligner::GetInstance() {
  if(!instance) {
    assert(false);
  }
  return instance;
}

LatentCrfAligner::LatentCrfAligner(const string &textFilename,
				   const string &outputPrefix,
				   LearningInfo &learningInfo,
				   unsigned FIRST_LABEL_ID,
				   const string &initialLambdaParamsFilename, 
				   const string &initialThetaParamsFilename,
				   const string &wordPairFeaturesFilename) : LatentCrfModel(textFilename,
											    outputPrefix,
											    learningInfo,
											    FIRST_LABEL_ID,
											    LatentCrfAligner::Task::WORD_ALIGNMENT) {
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
	x_sDomain.insert(vocabEncoder.Encode(*tokenIter));
      } else {
	zDomain.insert(vocabEncoder.Encode(*tokenIter));
      }
    }
  }
  
  // zero is reserved for FST epsilon (I don't think this is a problem since we don't use the x_i nor the z_i as fst labels)
  assert(this->x_sDomain.count(0) == 0 && this->zDomain.count(0) == 0);

  // encode the null token which is conventionally added to the beginning of the src sentnece. 
  NULL_TOKEN_STR = "__null__token__";
  bool explicitUseUnk = false;
  NULL_TOKEN = vocabEncoder.Encode(NULL_TOKEN_STR, explicitUseUnk);
  assert(NULL_TOKEN != vocabEncoder.UnkInt());
  
  // read and encode data
  srcSents.clear();
  tgtSents.clear();
  vocabEncoder.ReadParallelCorpus(textFilename, srcSents, tgtSents, NULL_TOKEN_STR);
  assert(srcSents.size() == tgtSents.size());
  assert(srcSents.size() > 0);
  examplesCount = srcSents.size();

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
  enabledFeatureTypes[101] = true;  // I( y_i - y_{i-1} == 0 )
  enabledFeatureTypes[102] = true;  // I( floor( ln(y_i - y_{i-1}) ) )
  enabledFeatureTypes[103] = true;  // I( tgt[i] aligns_to src[y_i] )
  enabledFeatureTypes[104] = true;  // precomputed features (src-tgt word pair)
  enabledFeatureTypes[105] = true;  // I( y_i - y_{i-1} )
  enabledFeatureTypes[106] = true;  // I( src[y_{i-1}]:src[y_{i}] )   captures the intuition that certain src side transitions are more common than others
  enabledFeatureTypes[107] = true;  // |y_i/len(src) - i/len(tgt)|

  // initialize (and normalize) the log theta params to gaussians
  if(initialThetaParamsFilename.size() == 0) {
    InitTheta();
    BroadcastTheta(0);
  } else {
    assert(nLogThetaGivenOneLabel.params.size() == 0);
    MultinomialParams::LoadParams(initialThetaParamsFilename, nLogThetaGivenOneLabel, vocabEncoder);
    assert(nLogThetaGivenOneLabel.params.size() > 0);
  }
  
  // populate the map of precomputed feature ids and feature values
  lambda->LoadPrecomputedFeaturesWith2Inputs(wordPairFeaturesFilename);

  // initialize the lambda parameters
  if(initialLambdaParamsFilename.size() == 0) {
    // add all features in this data set to lambda.params
    InitLambda();
    BroadcastLambdas(0);
  } else {
    assert(lambda->GetParamsCount() == 0);
    lambda->LoadParams(initialLambdaParamsFilename);
    assert(lambda->GetParamsCount() > 0);
  }
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
  this->yDomain.insert(LatentCrfAligner::START_OF_SENTENCE_Y_VALUE); // always insert the conceptual yValue of word at position -1 in a sentence
  // this length includes the null token that was inserted at the begging of all source sentences
  unsigned srcSentLength = testingMode? testSrcSents[exampleId].size() : srcSents[exampleId].size();
  // each position in the src sentence, including null, should have an entry in yDomain
  for(unsigned i = NULL_POSITION; i < NULL_POSITION + srcSentLength; ++i) {
    yDomain.insert(i);
  }
}

vector<int>& LatentCrfAligner::GetObservableSequence(int exampleId) {
  if(testingMode) {
    assert(exampleId < testTgtSents.size());
    return testTgtSents[exampleId];
  } else {
    if(exampleId >= tgtSents.size()) {
      cerr << exampleId << " < " << tgtSents.size() << endl;
    }
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

// tokens = target sentence
void LatentCrfAligner::Label(vector<int> &tokens, vector<int> &context, vector<int> &labels) {

  // set up
  assert(labels.size() == 0); 
  assert(tokens.size() > 0);
  testingMode = true;
  SetTestExample(tokens, context);

  // build an fst in which each path is a complete word alignment of the target sentence, weighted according to the model
  unsigned sentId = 0;
  fst::VectorFst<FstUtils::LogArc> fst;
  std::vector<FstUtils::LogWeight> alphas, betas;
  BuildThetaLambdaFst(sentId, tokens, fst, alphas, betas);
  
  // map to the tropical semiring which enjoys the path property (in order to find the best alignment)
  fst::VectorFst<FstUtils::StdArc> pathFst, shortestPathFst;
  fst::ArcMap(fst, &pathFst, FstUtils::LogToTropicalMapper());
  fst::ShortestPath(pathFst, &shortestPathFst);
  std::vector<int> dummy;
  FstUtils::LinearFstToVector(shortestPathFst, dummy, labels);

  // set down ;)
  testingMode = false;

  assert(labels.size() == tokens.size());  
}
  
void LatentCrfAligner::AddConstrainedFeatures() { 
  // no constrained features to add
}

void LatentCrfAligner::SetTestExample(vector<int> &x_t, vector<int> &x_s) {
  testSrcSents.clear();
  testSrcSents.push_back(x_s);
  testTgtSents.clear();
  testTgtSents.push_back(x_t);
}
