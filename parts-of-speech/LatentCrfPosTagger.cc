#include "LatentCrfPosTagger.h"

using namespace std;

vector<int64_t>& LatentCrfPosTagger::GetReconstructedObservableSequence(int exampleId) {
  if(testingMode) {
    if(testClassTgtSents.size() > 0) {
      return testClassTgtSents[exampleId];
    } else {
      return testData[exampleId];
    }
  } else {
    // refactor: this following line does not logically belong here
    lambda->learningInfo->currentSentId = exampleId;

    if(exampleId >= data.size()) {
      cerr << exampleId << " < " << data.size() << endl;
    }
    assert(exampleId < data.size());
    if(classTgtSents.size() > 0) {
      return classTgtSents[exampleId];
    } else {
      return data[exampleId];
    }
  }
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
                                                unsigned FIRST_LABEL_ID,
                                                const string &wordPairFeaturesFilename,
                                                const string &initLambdaFilename,
                                                const string &initThetaFilename) {
  if(!instance) {
    instance = new LatentCrfPosTagger(textFilename, outputPrefix, learningInfo, NUMBER_OF_LABELS, 
                                      FIRST_LABEL_ID, wordPairFeaturesFilename,
                                      initLambdaFilename,
                                      initThetaFilename);
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
                                       unsigned FIRST_LABEL_ID,
                                       const string &wordPairFeaturesFilename,
                                       const string &initLambdaFilename,
                                       const string &initThetaFilename) : LatentCrfModel(textFilename, 
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

  // slaves wait for master
  if(learningInfo.mpiWorld->rank() != 0) {
    bool vocabEncoderIsReady;
    boost::mpi::broadcast<bool>(*learningInfo.mpiWorld, vocabEncoderIsReady, 0);
  }


  // read and encode tgt words and their classes (e.g. brown clusters)
  if(learningInfo.mpiWorld->rank() == 0) {
    EncodeTgtWordClasses();
  } 

  // read and encode data
  data.clear();
  vocabEncoder.Read(textFilename, data);
  examplesCount = data.size();

  // read and encode tagging dictionary
  vector<vector<int64_t> > rawTagDict;
  int wordClassCounter = FIRST_ALLOWED_LABEL_VALUE;
  if(learningInfo.tagDictFilename.size() > 0) {
    vocabEncoder.Read(learningInfo.tagDictFilename, rawTagDict);
    for(auto wordTags = rawTagDict.begin(); wordTags != rawTagDict.end(); ++wordTags) {
      for(int i = 1; i < wordTags->size(); ++i) {
        if(posTagVocabIdToClassId.count( (*wordTags)[i]) == 0) {
          posTagVocabIdToClassId[(*wordTags)[i]] = wordClassCounter++;
        } 
        tagDict[(*wordTags)[0]].insert(posTagVocabIdToClassId[(*wordTags)[i]]);
      }
    }
    if(learningInfo.mpiWorld->rank() == 0) {
      cerr << "|tagDict| = " << tagDict.size() << endl;
      assert(wordClassCounter - FIRST_ALLOWED_LABEL_VALUE < yDomain.size());
    }
  }

  if(learningInfo.mpiWorld->rank() == 0 && wordPairFeaturesFilename.size() > 0) {
    cerr << "vocabEncoder.Count() = " << vocabEncoder.Count() << endl;
    lambda->LoadPrecomputedFeaturesWith2Inputs(wordPairFeaturesFilename);
    cerr << "vocabEncoder.Count() = " << vocabEncoder.Count() << endl;
  }

  // master signals to slaves that he's done
  if(learningInfo.mpiWorld->rank() == 0) {
    bool vocabEncoderIsReady;
    boost::mpi::broadcast<bool>(*learningInfo.mpiWorld, vocabEncoderIsReady, 0);
  }

  // load the mapping from each target word to its word class (e.g. brown clusters)
  LoadTgtWordClasses(data);

    
  // initialize (and normalize) the log theta params to gaussians
  InitTheta();
  if(initThetaFilename.size() > 0) {
    if(learningInfo.mpiWorld->rank() == 0) {
      cerr << "initializing theta params from " << initThetaFilename << endl;
    }
    MultinomialParams::LoadParams(initThetaFilename, nLogThetaGivenOneLabel, vocabEncoder, true, true);
    assert(nLogThetaGivenOneLabel.params.size() > 0);
  } else {
    BroadcastTheta(0);
  }

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
  
  // load saved parameters
  if(initLambdaFilename.size() > 0) {
    lambda->LoadParams(initLambdaFilename);
    assert(lambda->paramWeightsTemp.size() == lambda->paramIndexes.size());
    assert(lambda->paramIdsTemp.size() == lambda->paramIndexes.size());
  }

  // initialize the lambda parameters
  // add all features in this data set to lambda.params
  InitLambda();

  if(learningInfo.mpiWorld->rank() == 0) {
    vocabEncoder.PersistVocab(outputPrefix + string(".vocab"));
  }

}

LatentCrfPosTagger::~LatentCrfPosTagger() {}

void LatentCrfPosTagger::InitTheta() {
  if(learningInfo.mpiWorld->rank() == 0 && learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
    cerr << "master" << learningInfo.mpiWorld->rank() << ": initializing thetas...";
  }

  // create a vector of all word types in the corpus
  set<int64_t> wordTypes;
  for(unsigned sentId = 0; sentId < data.size(); ++sentId) {

    vector<int64_t> &reconstructedSent = classTgtSents.size() > 0?
      classTgtSents[sentId] : data[sentId];
    for(auto tokenIter = reconstructedSent.begin(); tokenIter != reconstructedSent.end(); ++tokenIter) {
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
  if(learningInfo.tgtWordClassesFilename.size() > 0) {
    testClassTgtSents.clear();
    testClassTgtSents.push_back( GetTgtWordClassSequence(tokens) );
  }
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
  if(learningInfo.testWithCrfOnly) {
    BuildLambdaFst(sentId, fst, alphas, betas);
  } else {
    BuildThetaLambdaFst(sentId, GetReconstructedObservableSequence(sentId), fst, alphas, betas);
  }  
  fst::VectorFst<FstUtils::StdArc> fst2, shortestPath;
  fst::ArcMap(fst, &fst2, FstUtils::LogToTropicalMapper());
  fst::ShortestPath(fst2, &shortestPath);
  std::vector<int> dummy;
  FstUtils::LinearFstToVector(shortestPath, dummy, labels);
  assert(labels.size() == tokens.size());

  testingMode = false;
}

void LatentCrfPosTagger::Label(const string &labelsFilename) {
  cerr << learningInfo.mpiWorld->rank() << ": inside LatentCrfPosTagger::Label(const string &labelsFilename)" << endl;
  // run viterbi (and write the classes to file)
  ofstream labelsFile(labelsFilename.c_str());
  if(learningInfo.firstKExamplesToLabel == 0) {
    learningInfo.firstKExamplesToLabel = examplesCount;
  }
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

    //std::vector<int64_t> &srcSent = GetObservableContext(exampleId);
    std::vector<int64_t> &tokens = GetObservableSequence(exampleId);
    std::vector<int> labels;
    // run viterbi
    Label(tokens, labels);

    stringstream ss;
    for(unsigned i = 0; i < labels.size(); ++i) {
      ss << labels[i] << " ";
    }
    ss << endl;
    if(learningInfo.mpiWorld->rank() == 0){
      labelsFile << ss.str();
    }else{
      learningInfo.mpiWorld->send(0, 0, ss.str());
    }
    
  }
  labelsFile.close();
}

