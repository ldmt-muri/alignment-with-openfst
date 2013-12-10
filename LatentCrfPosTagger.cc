#include "LatentCrfPosTagger.h"

using namespace std;

vector<int64_t>& LatentCrfPosTagger::GetReconstructedObservableSequence(int exampleId) {
  return GetObservableSequence(exampleId);
}

vector<int64_t>& LatentCrfPosTagger::GetObservableSequence(int exampleId) {
  if(testingMode) {
    assert(exampleId < testData.size());
    return testData[exampleId];
  } else {
    assert(exampleId < data.size());
    return data[exampleId];
  }
}

// singleton
LatentCrfModel* LatentCrfPosTagger::GetInstance(const string &textFilename, 
						const string &outputPrefix, 
						LearningInfo &learningInfo, 
						unsigned NUMBER_OF_LABELS, 
						unsigned FIRST_LABEL_ID) {
  if(!instance) {
    instance = new LatentCrfPosTagger(textFilename, outputPrefix, learningInfo, NUMBER_OF_LABELS, FIRST_LABEL_ID);
  } else {
    cerr << "A LatentCrfPosTagger object has already been initialized" << endl;
  }
  return instance;
}

LatentCrfModel* LatentCrfPosTagger::GetInstance() {
  if(!instance) {
    assert(false);
  }
  return instance;
}

LatentCrfPosTagger::LatentCrfPosTagger(const string &textFilename, 
				       const string &outputPrefix, 
				       LearningInfo &learningInfo, 
				       unsigned NUMBER_OF_LABELS, 
				       unsigned FIRST_LABEL_ID) : LatentCrfModel(textFilename, 
										 outputPrefix, 
										 learningInfo, 
										 FIRST_LABEL_ID,
										 LatentCrfModel::Task::POS_TAGGING) {
  // set constants
  LatentCrfModel::START_OF_SENTENCE_Y_VALUE = FIRST_LABEL_ID - 1;
  this->FIRST_ALLOWED_LABEL_VALUE = FIRST_LABEL_ID;
  assert(START_OF_SENTENCE_Y_VALUE > 0);

  // POS tag yDomain
  unsigned latentClasses = NUMBER_OF_LABELS;
  assert(latentClasses > 1);
  this->yDomain.insert(LatentCrfModel::START_OF_SENTENCE_Y_VALUE); // the conceptual yValue of word at position -1 in a sentence
  for(unsigned i = 0; i < latentClasses; i++) {
    this->yDomain.insert(LatentCrfModel::START_OF_SENTENCE_Y_VALUE + i + 1);
  }
  // zero is reserved for FST epsilon
  assert(this->yDomain.count(0) == 0);

  // read and encode data
  data.clear();
  vocabEncoder.Read(textFilename, data);
  examplesCount = data.size();

  // initialize (and normalize) the log theta params to gaussians
  InitTheta();

  // make sure all slaves have the same theta values
  BroadcastTheta(0);

  // persist initial parameters
  assert(learningInfo.iterationsCount == 0);
  if(learningInfo.iterationsCount % learningInfo.persistParamsAfterNIteration == 0 && learningInfo.mpiWorld->rank() == 0) {
    stringstream thetaParamsFilename;
    thetaParamsFilename << outputPrefix << ".initial.theta";
    if(learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
      cerr << "persisting theta parameters after iteration " << learningInfo.iterationsCount << " at " << thetaParamsFilename.str() << endl;
    }
    PersistTheta(thetaParamsFilename.str());
  }
  
  // initialize the lambda parameters
  // add all features in this data set to lambda.params
  InitLambda();
}

LatentCrfPosTagger::~LatentCrfPosTagger() {}

void LatentCrfPosTagger::InitTheta() {
  if(learningInfo.mpiWorld->rank() == 0 && learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
    cerr << "master" << learningInfo.mpiWorld->rank() << ": initializing thetas...";
  }

  // create a vector of all word types in the corpus
  set<int64_t> wordTypes;
  for(unsigned sentId = 0; sentId < data.size(); ++sentId) {
    vector<int64_t> &sent = data[sentId];
    for(auto tokenIter = sent.begin(); tokenIter != sent.end(); ++tokenIter) {
      wordTypes.insert(*tokenIter);
    }
  }

  // first initialize nlogthetas to unnormalized gaussians
  nLogThetaGivenOneLabel.params.clear();
  for(auto yDomainIter = yDomain.begin(); 
      yDomainIter != yDomain.end(); yDomainIter++) {
    for(auto wordTypeIter = wordTypes.begin(); wordTypeIter != wordTypes.end(); ++wordTypeIter) {
      nLogThetaGivenOneLabel.params[*yDomainIter][*wordTypeIter] = abs(gaussianSampler.Draw());
    }
  }
  
  // then normalize them
  MultinomialParams::NormalizeParams(nLogThetaGivenOneLabel);
  if(learningInfo.mpiWorld->rank() == 0) {
    cerr << "done" << endl;
  }
}

void LatentCrfPosTagger::SetTestExample(vector<int64_t> &tokens) {
  testData.clear();
  testData.push_back(tokens);
}

void LatentCrfPosTagger::Label(vector<int64_t> &tokens, vector<int> &labels) {
  assert(labels.size() == 0); 
  assert(tokens.size() > 0);
  testingMode = true;

  // hack to reuse the code that manipulates the fst
  SetTestExample(tokens);
  unsigned sentId = 0;
  
  fst::VectorFst<FstUtils::LogArc> fst;
  vector<FstUtils::LogWeight> alphas, betas;
  BuildThetaLambdaFst(sentId, tokens, fst, alphas, betas);
  fst::VectorFst<FstUtils::StdArc> fst2, shortestPath;
  fst::ArcMap(fst, &fst2, FstUtils::LogToTropicalMapper());
  fst::ShortestPath(fst2, &shortestPath);
  std::vector<int> dummy;
  FstUtils::LinearFstToVector(shortestPath, dummy, labels);
  assert(labels.size() == tokens.size());
}
