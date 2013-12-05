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

void LatentCrfAligner::EncodeTgtWordClasses() {
  if(learningInfo.mpiWorld != 0 || learningInfo.tgtWordClassesFilename.size() == 0) { return; }
  std::ifstream infile(learningInfo.tgtWordClassesFilename.c_str());
  string classString, wordString;
  int frequency;
  while(infile >> classString >> wordString >> frequency) {
    cerr << "reading line: " << classString << ", " << wordString << ", " << frequency << endl; 
    int64_t wordClass = vocabEncoder.Encode(classString);
    int64_t wordType = vocabEncoder.Encode(wordString);
  }
  vocabEncoder.Encode("?");
}

vector<int64_t> LatentCrfAligner::GetTgtWordClassSequence(vector<int64_t> &x_t) {
  vector<int64_t> classSequence;
  for(auto tgtToken = x_t.begin(); tgtToken != x_t.end(); tgtToken++) {
    if( tgtWordToClass.count(*tgtToken) == 0 ) {
      classSequence.push_back( vocabEncoder.ConstEncode("?") );
    } else {
      classSequence.push_back( tgtWordToClass[*tgtToken] );
    }
  }
  return classSequence;
}

void LatentCrfAligner::LoadTgtWordClasses() {
  // read the word class file and store it in a map
  if(learningInfo.tgtWordClassesFilename.size() == 0) { return; }
  tgtWordToClass.clear();
  std::ifstream infile(learningInfo.tgtWordClassesFilename.c_str());
  string classString, wordString;
  int frequency;
  while(infile >> classString >> wordString >> frequency) {
    cerr << "reading line: " << classString << ", " << wordString << ", " << frequency << endl; 
    int64_t wordClass = vocabEncoder.ConstEncode(classString);
    int64_t wordType = vocabEncoder.ConstEncode(wordString);
    tgtWordToClass[wordType] = wordClass;
  }
  infile.close();
  
  // now read each tgt sentence and create a corresponding sequence of tgt word clusters
  for(auto tgtSent = tgtSents.begin(); tgtSent != tgtSents.end(); tgtSent++) {
    classTgtSents.push_back( GetTgtWordClassSequence(*tgtSent) );
  }
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
  
  // slaves wait for master
  if(learningInfo.mpiWorld->rank() != 0) {
    bool vocabEncoderIsReady;
    boost::mpi::broadcast<bool>(*learningInfo.mpiWorld, vocabEncoderIsReady, 0);
  }

  // encode the null token which is conventionally added to the beginning of the src sentnece. 
  NULL_TOKEN_STR = "__null__token__";
  NULL_TOKEN = vocabEncoder.Encode(NULL_TOKEN_STR);
  assert(NULL_TOKEN != vocabEncoder.UnkInt());
  
  // read and encode tgt words and their classes (e.g. brown clusters)
  if(learningInfo.mpiWorld->rank() == 0) {
    EncodeTgtWordClasses();
  }

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

  if(learningInfo.mpiWorld->rank() == 0) {
    lambda->LoadPrecomputedFeaturesWith2Inputs(wordPairFeaturesFilename);
  }

  // master signals to slaves that he's done
  if(learningInfo.mpiWorld->rank() == 0) {
    bool vocabEncoderIsReady;
    boost::mpi::broadcast<bool>(*learningInfo.mpiWorld, vocabEncoderIsReady, 0);
  }

  // load the mapping from each target word to its word class (e.g. brown clusters)
  LoadTgtWordClasses();

  // initialize (and normalize) the log theta params to gaussians
  InitTheta();
  if(initialThetaParamsFilename.size() == 0) {
    BroadcastTheta(0);
  } else {
    //assert(nLogThetaGivenOneLabel.params.size() == 0);
    if(learningInfo.mpiWorld->rank() == 0) {
      cerr << "initializing theta params from " << initialThetaParamsFilename << endl;
    }
    MultinomialParams::LoadParams(initialThetaParamsFilename, nLogThetaGivenOneLabel, vocabEncoder, true, true);
    assert(nLogThetaGivenOneLabel.params.size() > 0);
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
    vector<int64_t> &reconstructedSent = classTgtSents.size() > 0?
      classTgtSents[sentId] : tgtSents[sentId];
    for(unsigned i = 0; i < srcSent.size(); ++i) {
      auto srcToken = srcSent[i];
      for(unsigned j = 0; j < reconstructedSent.size(); ++j) {
	auto tgtToken = reconstructedSent[j];
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

vector<int64_t>& LatentCrfAligner::GetReconstructedObservableSequence(int exampleId) {
  if(testingMode) {
    if(testClassTgtSents.size() > 0) {
      return testClassTgtSents[exampleId];
    } else {
      return testTgtSents[exampleId];
    }
  } else {
    // refactor: this following line does not logically belong here
    lambda->learningInfo->currentSentId = exampleId;

    if(exampleId >= tgtSents.size()) {
      cerr << exampleId << " < " << tgtSents.size() << endl;
    }
    assert(exampleId < tgtSents.size());
    if(classTgtSents.size() > 0) {
      return classTgtSents[exampleId];
    } else {
      return tgtSents[exampleId];
    }
  }
}

vector<int64_t>& LatentCrfAligner::GetObservableSequence(int exampleId) {
  if(testingMode) {
    assert(exampleId < testTgtSents.size());
    return testTgtSents[exampleId];
  } else {
    lambda->learningInfo->currentSentId = exampleId;
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
  testClassTgtSents.clear();  
  testClassTgtSents.push_back( GetTgtWordClassSequence(x_t) );
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
  if(learningInfo.testWithCrfOnly) {
    BuildLambdaFst(sentId, fst, alphas, betas);
  } else {
    BuildThetaLambdaFst(sentId, GetReconstructedObservableSequence(sentId), fst, alphas, betas);
  }

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
    lambda->learningInfo->currentSentId = exampleId;
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
