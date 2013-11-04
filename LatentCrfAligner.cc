#include "LatentCrfAligner.h"

string LatentCrfAligner::NULL_TOKEN_STR = "REDICULOUS";
int64_t LatentCrfAligner::NULL_TOKEN = -1000000;
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
  if(learningInfo.allowNullAlignments) {
    vocabEncoder.ReadParallelCorpus(textFilename, srcSents, tgtSents, 
                                    NULL_TOKEN_STR, learningInfo.reverse);
  } else {
    vocabEncoder.ReadParallelCorpus(textFilename, srcSents, tgtSents, learningInfo.reverse);
  }
  assert(srcSents.size() == tgtSents.size());
  assert(srcSents.size() > 0);
  examplesCount = srcSents.size();
  
  // initialize (and normalize) the log theta params to gaussians
  if(initialThetaParamsFilename.size() == 0) {
    InitTheta();
    BroadcastTheta(0);
  } else {
    assert(nLogThetaGivenOneLabel.params.size() == 0);
    if(learningInfo.mpiWorld->rank() == 0) {
      cerr << "initializing theta params from " << initialThetaParamsFilename << endl;
    }
    MultinomialParams::LoadParams(initialThetaParamsFilename, nLogThetaGivenOneLabel, vocabEncoder, true, true);
    assert(nLogThetaGivenOneLabel.params.size() > 0);
  }
  
  if(learningInfo.mpiWorld->rank() == 0) {
    lambda->LoadPrecomputedFeaturesWith2Inputs(wordPairFeaturesFilename);
  }

  // load saved parameters
  if(initialLambdaParamsFilename.size() > 0) {
    lambda->LoadParams(initialLambdaParamsFilename);
    assert(lambda->paramWeightsTemp.size() == lambda->paramIndexes.size());
    assert(lambda->paramIdsTemp.size() == lambda->paramIndexes.size());
  }

  // add all features in this data set to lambda
  InitLambda();

  assert(lambda->paramWeightsTemp.size() == 0 && lambda->paramIdsTemp.size() == 0);
  assert(lambda->paramWeightsPtr->size() == lambda->paramIndexes.size());
  assert(lambda->paramIdsPtr->size() == lambda->paramIndexes.size());
}

void LatentCrfAligner::InitTheta() {

  if(learningInfo.mpiWorld->rank() == 0 && learningInfo.debugLevel >= DebugLevel::CORPUS) {
    cerr << "master" << learningInfo.mpiWorld->rank() << ": initializing thetas...";
  }

  assert(srcSents.size() == tgtSents.size());

  // first initialize nlogthetas to unnormalized gaussians
  nLogThetaGivenOneLabel.params.clear();
  for(unsigned sentId = 0; sentId < srcSents.size(); ++sentId) {
    vector<int64_t> &srcSent = srcSents[sentId];
    vector<int64_t> &tgtSent = tgtSents[sentId];
    for(unsigned i = 0; i < srcSent.size(); ++i) {
      auto srcToken = srcSent[i];
      for(unsigned j = 0; j < tgtSent.size(); ++j) {
	auto tgtToken = tgtSent[j];
	if(learningInfo.initializeThetasWithGaussian) {
	  nLogThetaGivenOneLabel.params[srcToken][tgtToken] = abs(gaussianSampler.Draw());
	} else if (learningInfo.initializeThetasWithUniform || learningInfo.initializeThetasWithModel1) {
	  nLogThetaGivenOneLabel.params[srcToken][tgtToken] = 1;
	}
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
  // if null alignments are enabled, this length will include the null token that was inserted at the begging of all source sentences
  unsigned srcSentLength = testingMode? testSrcSents[exampleId].size() : srcSents[exampleId].size();
  // each position in the src sentence, including null, should have an entry in yDomain
  unsigned firstPossibleYValue = learningInfo.allowNullAlignments? NULL_POSITION : NULL_POSITION + 1;
  for(unsigned i = firstPossibleYValue; i < firstPossibleYValue + srcSentLength; ++i) {
    yDomain.insert(i);
  }
}

vector<int64_t>& LatentCrfAligner::GetObservableSequence(int exampleId) {
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

vector<int64_t>& LatentCrfAligner::GetObservableContext(int exampleId) { 
  if(testingMode) {
    assert(exampleId < testSrcSents.size());
    return testSrcSents[exampleId];
  } else {
    assert(exampleId < srcSents.size());
    return srcSents[exampleId];
  }   
}

void LatentCrfAligner::SetTestExample(vector<int64_t> &x_t, vector<int64_t> &x_s) {
  testSrcSents.clear();
  testSrcSents.push_back(x_s);
  testTgtSents.clear();
  testTgtSents.push_back(x_t);
}

// tokens = target sentence
void LatentCrfAligner::Label(vector<int64_t> &tokens, vector<int64_t> &context, vector<int> &labels) {

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

void LatentCrfAligner::Label(const string &labelsFilename) {
  // run viterbi (and write alignments in giza format)
  ofstream labelsFile(labelsFilename.c_str());
  assert(learningInfo.firstKExamplesToLabel <= examplesCount);
  for(unsigned exampleId = 0; exampleId < learningInfo.firstKExamplesToLabel; ++exampleId) {
    if(exampleId % learningInfo.mpiWorld->size() != learningInfo.mpiWorld->rank()) {
      if(learningInfo.mpiWorld->rank() == 0){
        string labelSequence;
        learningInfo.mpiWorld->recv(exampleId % learningInfo.mpiWorld->size(), 0, labelSequence);
        labelsFile << labelSequence;
      }
      continue;
    }
    std::vector<int64_t> &srcSent = GetObservableContext(exampleId);
    std::vector<int64_t> &tgtSent = GetObservableSequence(exampleId);
    std::vector<int> labels;
    // run viterbi
    Label(tgtSent, srcSent, labels);
    
    stringstream ss;
    for(unsigned i = 0; i < labels.size(); ++i) {
      // dont write null alignments
      if(labels[i] == NULL_POSITION) {
        continue;
      }
      // determine the alignment (i.e. src position) for this tgt position (i)
      int alignment = labels[i] - FIRST_SRC_POSITION;
      assert(alignment >= 0);
      if(learningInfo.reverse) {
        ss << i << "-" << alignment << " ";
      } else {
        ss << alignment << "-" << i << " ";
      }
      
    }
    ss << endl;
    if(learningInfo.mpiWorld->rank() == 0){
      labelsFile << ss.str();
    }else{
      //cerr << "rank" << learningInfo.mpiWorld->rank() << " will send exampleId " << exampleId << " to master" << endl; 
      learningInfo.mpiWorld->send(0, 0, ss.str());
      //cerr << "rank" << learningInfo.mpiWorld->rank() << " sending done." << endl;
    }
    
  }
  labelsFile.close();
}

int64_t LatentCrfAligner::GetContextOfTheta(unsigned sentId, int y) {
  vector<int64_t> &srcSent = GetObservableContext(sentId);
  if(y == NULL_POSITION) {
    return NULL_TOKEN;
  } else {
    assert(y - FIRST_SRC_POSITION < srcSent.size());
    assert(y - FIRST_SRC_POSITION >= 0);
    return srcSent[y - NULL_POSITION];
  }
}
