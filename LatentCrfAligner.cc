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
    cerr << "no instance was found!" << endl;
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
    vocabEncoder.ReadParallelCorpus(textFilename, srcSents, tgtSents, NULL_TOKEN_STR);
  } else {
    vocabEncoder.ReadParallelCorpus(textFilename, srcSents, tgtSents);
  }
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
  enabledFeatureTypes[101] = true;  // (bool) // id = y_i-y_{i-1} 
  enabledFeatureTypes[102] = true;  // (bool) // id = floor( ln(y_i - y_{i-1}) )
  enabledFeatureTypes[103] = true;  // (bool) // id = tgt[i] aligns_to src[y_i]
  enabledFeatureTypes[104] = true;  // precomputed features (src-tgt word pair), including model1.   this is the features chris used in his 2011 paper with Gimpel
  enabledFeatureTypes[105] = true;  // (bool) // id = y_i - y_{i-1} captures the intuition that certain jump distances are more common than others
  enabledFeatureTypes[106] = true;  // (bool) // id = src[y_{i-1}]:src[y_{i}]  captures the intuition that certain src side transitions are more common than others
  enabledFeatureTypes[107] = true;  // (real) // value = |y_i/len(src) - i/len(tgt)| the positional features used by dyer in several WA papers
  enabledFeatureTypes[108] = true;  // (bool) // value = I( i==0 && y_i==0 )
                                    // (bool) // value = I( i==len(tgt) && y_i==len(src) ) captures the intuition that the first and last word in the target sentence usually aligns to the first and last word in the src sentence, respectively.
  enabledFeatureTypes[109] = true;

  // initialize (and normalize) the log theta params to gaussians
  if(initialThetaParamsFilename.size() == 0) {
    InitTheta();
    BroadcastTheta(0);
  } else {
    assert(nLogThetaGivenOneLabel.params.size() == 0);
    MultinomialParams::LoadParams(initialThetaParamsFilename, nLogThetaGivenOneLabel, vocabEncoder, true, true);
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

void LatentCrfAligner::AddConstrainedFeatures() { 
  // no constrained features to add
}

void LatentCrfAligner::SetTestExample(vector<int> &x_t, vector<int> &x_s) {
  testSrcSents.clear();
  testSrcSents.push_back(x_s);
  testTgtSents.clear();
  testTgtSents.push_back(x_t);
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
  
void LatentCrfAligner::Label(const string &labelsFilename) {
  // run viterbi (and write alignments in giza format)
  ofstream labelsFile(labelsFilename.c_str());
  assert(learningInfo.firstKExamplesToLabel <= examplesCount);
  cerr << "labeling the first " << examplesCount << " in the corpus" << endl;
  for(unsigned exampleId = 0; exampleId < learningInfo.firstKExamplesToLabel; ++exampleId) {
    cerr << "rank" << learningInfo.mpiWorld->rank() << ": started processing exampleId " << exampleId << endl;
    // if this example does not belong to this process, skip it (except for the master who receives its output)
    if(exampleId % learningInfo.mpiWorld->size() != learningInfo.mpiWorld->rank()) {
      if(learningInfo.mpiWorld->rank() == 0){
        string labelSequence;
        cerr << "master is waiting for the result of exampleId " << exampleId << " from rank " << exampleId % learningInfo.mpiWorld->size() << endl;
        learningInfo.mpiWorld->recv(exampleId % learningInfo.mpiWorld->size(), 0, labelSequence);
        labelsFile << labelSequence;
        cerr << "master received the reuslt of exampleId " << exampleId << " and printed it to disk" << endl;
      }
      cerr << "rank" << learningInfo.mpiWorld->rank() << " is done with exampleId" << endl;
      continue;
    }
    std::vector<int> &srcSent = GetObservableContext(exampleId);
    std::vector<int> &tgtSent = GetObservableSequence(exampleId);
    std::vector<int> labels;
    // run viterbi
    Label(tgtSent, srcSent, labels);
    //
    stringstream ss;
    for(unsigned i = 0; i < labels.size(); ++i) {
      // dont write null alignments
      if(labels[i] == NULL_POSITION) {
        continue;
      }
      // determine the alignment (i.e. src position) for this tgt position (i)
      int alignment = labels[i] - FIRST_SRC_POSITION;
      assert(alignment >= 0);
      ss << alignment << "-" << i << " ";
    }
    ss << endl;
    if(learningInfo.mpiWorld->rank() == 0){
      labelsFile << ss.str();
    }else{
      cerr << "rank" << learningInfo.mpiWorld->rank() << " will send exampleId " << exampleId << " to master" << endl; 
      learningInfo.mpiWorld->send(0, 0, ss.str());
      cerr << "rank" << learningInfo.mpiWorld->rank() << " sending done." << endl;
    }
    
  }
  labelsFile.close();
}

int LatentCrfAligner::GetContextOfTheta(unsigned sentId, int y) {
  vector<int> &srcSent = GetObservableContext(sentId);
  if(y == NULL_POSITION) {
    return NULL_TOKEN;
  } else {
    assert(y - FIRST_SRC_POSITION < srcSent.size());
    assert(y - FIRST_SRC_POSITION >= 0);
    return srcSent[y - NULL_POSITION];
  }
}
