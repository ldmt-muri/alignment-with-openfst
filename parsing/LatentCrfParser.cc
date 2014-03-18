#include "LatentCrfParser.h"

string LatentCrfParser::HEAD_STR = "__HEAD__";
int64_t LatentCrfParser::HEAD_ID = -1000000;
int LatentCrfParser::HEAD_POSITION = 3;

// singleton
LatentCrfModel* LatentCrfParser::GetInstance(const string &textFilename, 
					     const string &outputPrefix, 
					     LearningInfo &learningInfo, 
					     const string &initialLambdaParamsFilename, 
					     const string &initialThetaParamsFilename,
					     const string &wordPairFeaturesFilename) {
  
  if(!instance) {
    instance = new LatentCrfParser(textFilename, 
                                    outputPrefix,
                                    learningInfo, 
                                    initialLambdaParamsFilename, 
                                    initialThetaParamsFilename,
                                    wordPairFeaturesFilename);
  }
  return instance;
}

LatentCrfModel* LatentCrfParser::GetInstance() {
  if(!instance) {
    assert(false);
  }

  return instance;
}

LatentCrfParser::LatentCrfParser(const string &textFilename,
				   const string &outputPrefix,
				   LearningInfo &learningInfo,
				   const string &initialLambdaParamsFilename, 
				   const string &initialThetaParamsFilename,
				   const string &wordPairFeaturesFilename) : LatentCrfModel(textFilename,
											    outputPrefix,
											    learningInfo,
											    LatentCrfParser::HEAD_POSITION,
											    LatentCrfParser::Task::DEPENDENCY_PARSING) {

  // unlike POS tagging, yDomain depends on the src sentence length. we will set it on a per-sentence basis.
  this->yDomain.clear();
  
  // slaves wait for master
  if(learningInfo.mpiWorld->rank() != 0) {
    bool vocabEncoderIsReady;
    boost::mpi::broadcast<bool>(*learningInfo.mpiWorld, vocabEncoderIsReady, 0);
  }

  // encode the null token which is conventionally added to the beginning of the src sentnece. 
  HEAD_STR = "__HEAD__";
  HEAD_ID = vocabEncoder.Encode(HEAD_STR);
  assert(HEAD_ID != vocabEncoder.UnkInt());
  
  // read and encode data
  sents.clear();
  vocabEncoder.ReadCorpus(textFilename, sents);
  assert(sents.size() > 0);
  examplesCount = sents.size();

  if(learningInfo.mpiWorld->rank() == 0 && wordPairFeaturesFilename.size() > 0) {
    lambda->LoadPrecomputedFeaturesWith2Inputs(wordPairFeaturesFilename);
  }

  // master signals to slaves that he's done
  if(learningInfo.mpiWorld->rank() == 0) {
    bool vocabEncoderIsReady;
    boost::mpi::broadcast<bool>(*learningInfo.mpiWorld, vocabEncoderIsReady, 0);
  }

  // initialize (and normalize) the log theta params to gaussians
  InitTheta();
  if(initialThetaParamsFilename.size() > 0) {
    //assert(nLogThetaGivenOneLabel.params.size() == 0);
    if(learningInfo.mpiWorld->rank() == 0) {
      cerr << "initializing theta params from " << initialThetaParamsFilename << endl;
    }
    MultinomialParams::LoadParams(initialThetaParamsFilename, nLogThetaGivenOneLabel, vocabEncoder, true, true);
    assert(nLogThetaGivenOneLabel.params.size() > 0);
  } else {
    BroadcastTheta(0);
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

  if(learningInfo.mpiWorld->rank() == 0) {
    vocabEncoder.PersistVocab(outputPrefix + string(".vocab"));
  }

}

void LatentCrfParser::InitTheta() {

  if(learningInfo.mpiWorld->rank() == 0 && learningInfo.debugLevel >= DebugLevel::CORPUS) {
    cerr << "master" << learningInfo.mpiWorld->rank() << ": initializing thetas...";
  }

  assert(sents.size() > 0);

  // first initialize nlogthetas to unnormalized gaussians
  nLogThetaGivenOneLabel.params.clear();
  for(unsigned sentId = 0; sentId < sents.size(); ++sentId) {
    vector<int64_t> &sent = sents[sentId];
    vector<int64_t> &reconstructedSent = sents[sentId];
    for(unsigned i = 0; i < sent.size(); ++i) {
      auto parentToken = sent[i];
      for(unsigned j = 0; j < reconstructedSent.size(); ++j) {
        if(i == j) { continue; }
	auto childToken = reconstructedSent[j];
	if(learningInfo.initializeThetasWithGaussian) {
          nLogThetaGivenOneLabel.params[parentToken][childToken] = abs(gaussianSampler.Draw());
        } else if (learningInfo.initializeThetasWithUniform || learningInfo.initializeThetasWithModel1) {
          nLogThetaGivenOneLabel.params[parentToken][childToken] = 1;
        }
      }
    }
  }
  
  // then normalize them
  MultinomialParams::NormalizeParams(nLogThetaGivenOneLabel);

  stringstream thetaParamsFilename;
  thetaParamsFilename << outputPrefix << ".init.theta";
  PersistTheta(thetaParamsFilename.str());

  if(learningInfo.mpiWorld->rank() == 0) {
    cerr << "done" << endl;
  }
}

void LatentCrfParser::PrepareExample(unsigned exampleId) {
  yDomain.clear();
  this->yDomain.insert(LatentCrfParser::HEAD_POSITION);
  unsigned sentLength = testingMode? testSents[exampleId].size() : sents[exampleId].size();
  // each position in the src sentence, including null, should have an entry in yDomain
  for(unsigned i = LatentCrfParser::HEAD_POSITION + 1; i < LatentCrfParser::HEAD_POSITION + sentLength + 1; ++i) {
    yDomain.insert(i);
  }
}

vector<int64_t>& LatentCrfParser::GetReconstructedObservableSequence(int exampleId) {
  if(testingMode) {
    return testSents[exampleId];
  } else {
    // refactor: this following line does not logically belong here
    lambda->learningInfo->currentSentId = exampleId;

    assert(exampleId < sents.size());
    return sents[exampleId];
  }
}

vector<int64_t>& LatentCrfParser::GetObservableSequence(int exampleId) {
  if(testingMode) {
    assert(exampleId < testSents.size());
    return testSents[exampleId];
  } else {
    lambda->learningInfo->currentSentId = exampleId;
    assert(exampleId < sents.size());
    return sents[exampleId];
  }
}

vector<int64_t>& LatentCrfParser::GetObservableContext(int exampleId) { 
  if(testingMode) {
    assert(exampleId < testSents.size());
    return testSents[exampleId];
  } else {
    assert(exampleId < sents.size());
    return sents[exampleId];
  }
}

void LatentCrfParser::SetTestExample(vector<int64_t> &sent) {
  testSents.clear();
  testSents.push_back(sent);
}

void LatentCrfParser::Label(vector<int64_t> &tokens, vector<int> &labels) {

  // set up
  assert(labels.size() == 0); 
  assert(tokens.size() > 0);
  testingMode = true;
  SetTestExample(tokens);

  // do the actual labeling
  // TODO: depparse. now, label() always return a parse in which all words are direct children of HEAD
  for(auto tokenIter = tokens.begin(); tokenIter != tokens.end(); ++tokenIter) {
    labels.push_back(HEAD_ID);
  }

  // set down ;)
  testingMode = false;

  assert(labels.size() == tokens.size());  
}

void LatentCrfParser::Label(const string &labelsFilename) {
  ofstream labelsFile(labelsFilename.c_str());
  assert(learningInfo.firstKExamplesToLabel <= examplesCount);
  for(unsigned exampleId = 0; exampleId < learningInfo.firstKExamplesToLabel; ++exampleId) {
    lambda->learningInfo->currentSentId = exampleId;
    if(exampleId % learningInfo.mpiWorld->size() != learningInfo.mpiWorld->rank()) {
      if(learningInfo.mpiWorld->rank() == 0){
        string labelSequence;
        learningInfo.mpiWorld->recv(exampleId % learningInfo.mpiWorld->size(), 0, labelSequence);
        labelsFile << labelSequence;
      }
      continue;
    }

    std::vector<int64_t> &sent = GetObservableSequence(exampleId);
    std::vector<int> labels;
    // run viterbi
    Label(sent, labels);
    
    // TODO: depparse. fix how the viterbi parses are written to file
    stringstream ss;
    for(unsigned i = 0; i < labels.size(); ++i) {
      // determine the alignment (i.e. src position) for this tgt position (i)
      int parent = labels[i] - HEAD_POSITION;
      ss << parent << " ";
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

int64_t LatentCrfParser::GetContextOfTheta(unsigned sentId, int y) {
  vector<int64_t> &srcSent = GetObservableContext(sentId);
  if(y == NULL_POSITION) {
    return NULL_TOKEN;
  } else {
    assert(y - FIRST_SRC_POSITION < srcSent.size());
    assert(y - FIRST_SRC_POSITION >= 0);
    return srcSent[y - NULL_POSITION];
  }
}
