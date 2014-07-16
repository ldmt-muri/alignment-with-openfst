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
  this->yDomain.push_back(LatentCrfModel::START_OF_SENTENCE_Y_VALUE); // the conceptual yValue of word at position -1 in a sentence
  for(unsigned i = 0; i < latentClasses; i++) {
    this->yDomain.push_back(LatentCrfModel::START_OF_SENTENCE_Y_VALUE + i + 1);
  }
  // zero is reserved for FST epsilon
  assert(LatentCrfModel::START_OF_SENTENCE_Y_VALUE > 0);
  
  // if gold labels are provided, establish a one-to-one mapping between elements in yDomain and the label strings
  if(learningInfo.goldFilename.size() > 0) {
    auto yDomainIter = yDomain.begin();
    // skip the START-OF-SENTENCE label
    if(*yDomainIter == LatentCrfModel::START_OF_SENTENCE_Y_VALUE) { 
      yDomainIter++;
    }
    std::vector<std::vector<std::string> > labelStringSequences;
    goldLabelSequences.clear();
    StringUtils::ReadTokens(learningInfo.goldFilename, labelStringSequences);
    for(auto labelStringSequencesIter = labelStringSequences.begin(); 
        labelStringSequencesIter != labelStringSequences.end();
        ++labelStringSequencesIter) {
      vector<int> goldLabelSequence;
      for(auto labelStringIter = labelStringSequencesIter->begin(); 
          labelStringIter != labelStringSequencesIter->end(); 
          ++labelStringIter) {
        // is it a new label?
        if(labelStringToInt.count(*labelStringIter) == 0) {
          // yes!
          // are there any unused elements in yDomain (excluding the start of sentence label)?
          if(yDomainIter == yDomain.end() || yDomain.size() - 1 <= labelStringToInt.size()) {
            cerr << "ERROR: the number of unique label strings in the gold file is greater than the predefined number of classes = " << NUMBER_OF_LABELS << " (fyi: yDomain.size() = " << yDomain.size() << ")" << endl;
            assert(false);
          } 
	  if(false) {
	    auto trimmed = StringUtils::Trim(*labelStringIter);
	    if (trimmed.length() == 0) {
	      cerr << "ERROR: label has zero length when trimmed\n";
	      assert(false);
	    }
          }
          labelStringToInt[*labelStringIter] = *yDomainIter;
          labelIntToString[*yDomainIter] = *labelStringIter;
          yDomainIter++;
          if(yDomainIter != yDomain.end() && *yDomainIter == LatentCrfModel::START_OF_SENTENCE_Y_VALUE) { 
            yDomainIter++;
          }
        }
        goldLabelSequence.push_back(labelStringToInt[*labelStringIter]);
      }
      goldLabelSequences.push_back(goldLabelSequence);
    }
    if(yDomainIter != yDomain.end()) {
      cerr << "The unique gold labels are fewer than the possible values in yDomain. Therefore, we will remove the unused elements in yDomain. " << endl;
    }
    while(yDomainIter != yDomain.end()) {
      yDomainIter = yDomain.erase(yDomainIter);
    }
    if(learningInfo.mpiWorld->rank() == 0) {
      cerr << "now, |yDomain| = " << yDomain.size() << endl;
      cerr << "the mapping between string labels and yDomain elements is as follows: " << endl;
      for(auto labelIntToStringIter = labelIntToString.begin();
          labelIntToStringIter != labelIntToString.end();
          ++labelIntToStringIter) {
        cerr << labelIntToStringIter->first << " -> " << labelIntToStringIter->second << endl;
      }
      for(auto labelStringToIntIter = labelStringToInt.begin();
          labelStringToIntIter != labelStringToInt.end();
          ++labelStringToIntIter) {
        cerr << labelStringToIntIter->first << " -> " << labelStringToIntIter->second << endl;
      }
      cerr << goldLabelSequences.size() << " gold label sequences read." << endl;
    }
  }
  
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

// -loglikelihood is the return value
// nll = - log ( exp( \lambda . f(y-gold, x) ) / \sum_y exp( \lambda.f(y,x) )
//     = - \lambda . f(y-gold, x) + log \sum_y exp( \lambda . f(y, x) )
//     = - \lambda . f(y-gold, x) - nLogZ
// nDerivative_k = - f_k(y-gold, x) + [ \sum_y f_k(y,x) * exp (\lambda . f(y,x)) ] / [ \sum_y exp( \lambda . f(y, x)) ]
//               = - f_k(y-gold, x) + E_{p(y|x)}[f_k(y,x)]
double LatentCrfPosTagger::ComputeNllYGivenXAndLambdaGradient(
							  vector<double> &derivativeWRTLambda, int fromSentId, int toSentId) {
  
  // this method is used for supervised training. if we don't have any gold labels then we can't do supervised training.
  assert(goldLabelSequences.size() > 0);

  double objective = 0;

  assert(derivativeWRTLambda.size() == lambda->GetParamsCount());
  
  // for each training example
  for(int sentId = fromSentId; sentId < toSentId; sentId++) {

    // only process sentences for which there are gold labels
    if(sentId >= goldLabelSequences.size()) {
      break;
    }
   
    // sentId is assigned to the process with rank = sentId % world.size()
    if(sentId % learningInfo.mpiWorld->size() != learningInfo.mpiWorld->rank()) {
      continue;
    }
    
    vector<int64_t> &tokens = GetObservableSequence(sentId);
    /*if(sentId == 383) {
      cerr << endl << "sentId=" << sentId << ", tokens=";
      for(auto tokensIter = tokens.begin(); tokensIter != tokens.end(); ++tokensIter) {
        cerr << vocabEncoder.Decode(*tokensIter) << " ";
      }
    }
    */
    vector<int> &labels = goldLabelSequences[sentId];
    if(tokens.size() != labels.size()) {
      cerr << "ERROR: the number of tokens = " << tokens.size() << " is different than the number of labels = " << labels.size() << " in sentId = " << sentId << endl;
    }
    
    // prune long sequences
    if( learningInfo.maxSequenceLength > 0 && 
        tokens.size() > learningInfo.maxSequenceLength ) {
      continue;
    }
    
    // build the FSTs
    fst::VectorFst<FstUtils::LogArc> lambdaFst;
    vector<FstUtils::LogWeight> lambdaAlphas, lambdaBetas;
    BuildLambdaFst(sentId, lambdaFst, lambdaAlphas, lambdaBetas, &derivativeWRTLambda, &objective);
    
    // compute the F map fro this sentence
    FastSparseVector<LogVal<double> > FSparseVector;
    ComputeF(sentId, lambdaFst, lambdaAlphas, lambdaBetas, FSparseVector);

    // compute the Z value for this sentence
    double nLogZ = ComputeNLogZ_lambda(lambdaFst, lambdaBetas);
    
    // keep an eye on bad numbers
    if(std::isinf(nLogZ)) {
      cerr << "WARNING: nLogZ = " << nLogZ << ". ";
      cerr << "I will just ignore this sentence ..." << endl;
      assert(false);
      continue;
    } else if(std::isnan(nLogZ)) {
      cerr << "ERROR: nLogZ = " << nLogZ << endl;
      assert(false);
    }

    // the denominator
    double sentLevelObjective = -nLogZ; // this term should always be greater than the (supervision) other term
    objective -= nLogZ;

    // subtract F/Z from the gradient
    for(FastSparseVector<LogVal<double> >::iterator fIter = FSparseVector.begin(); 
        fIter != FSparseVector.end(); ++fIter) {
      double nLogf = fIter->second.s_? fIter->second.v_ : -fIter->second.v_; // multiply the inner logF representation by -1.
      double fOverZ = MultinomialParams::nExp(nLogf - nLogZ);
      if(std::isnan(fOverZ) || std::isinf(fOverZ)) {
        if(learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
          cerr << "ERROR: fOverZ = " << nLogZ << ", nLogf = " << nLogf << ". my mistake. will halt!" << endl;
        }
        assert(false);
      }
      assert(fIter->first < derivativeWRTLambda.size());

      /*if(sentId == 383) {
        //cerr << "feature #" << fIter->first << " = " << (*lambda->paramIdsPtr)[fIter->first] << ", fOverZ = " << fOverZ << endl;
        }*/
      
      derivativeWRTLambda[fIter->first] += fOverZ;
      if(std::isnan(derivativeWRTLambda[fIter->first]) || 
         std::isinf(derivativeWRTLambda[fIter->first])) {
        cerr << "rank #" << learningInfo.mpiWorld->rank()               \
             << ": ERROR: fOverZ = " << nLogZ << ", nLogf = " << nLogf  \
             << ". my mistake. will halt!" << endl;
        assert(false);
      }
    }
    
    double goldSequenceScore = 0.0;
    // now, add terms of the gold label
    int prevLabel = LatentCrfModel::START_OF_SENTENCE_Y_VALUE;
    for(int tokenIndex = 0; tokenIndex < tokens.size(); ++tokenIndex) {
      FastSparseVector<double> activeFeatures;
      FireFeatures(labels[tokenIndex], prevLabel, sentId, tokenIndex, activeFeatures);
      prevLabel = labels[tokenIndex];
      // update objective
      goldSequenceScore += lambda->DotProduct(activeFeatures);
      sentLevelObjective -= lambda->DotProduct(activeFeatures);
      objective -= lambda->DotProduct(activeFeatures);
      // update gradient
      for(auto activeFeatureIndex = activeFeatures.begin(); 
          activeFeatureIndex != activeFeatures.end();
          ++activeFeatureIndex) {
        derivativeWRTLambda[activeFeatureIndex->first] -= activeFeatureIndex->second;
      }
    }
    if(goldSequenceScore > -nLogZ) {
      cerr << "sentId=" << sentId << ": is gold score=" << goldSequenceScore << " <= logZ=" << -nLogZ << "?" << (goldSequenceScore < -nLogZ) << endl;
    }
    // debug info
    if(sentId % learningInfo.nSentsPerDot == 0) {
      cerr << ".";
    }
  } // end of training examples 

  cerr << learningInfo.mpiWorld->rank() << "|";
  
  if(learningInfo.mpiWorld->rank() == 0) {
    cerr << "at the end of some lbfgs iteration " << endl;
    lambda->PersistParams(learningInfo.outputFilenamePrefix + ".current.lambda", false);
    lambda->PersistParams(learningInfo.outputFilenamePrefix + ".current.lambda.humane", true);
    cerr << "parameters can be found at " << learningInfo.outputFilenamePrefix << ".current.lambda" << endl;
  }
  
  return objective;
}

void LatentCrfPosTagger::SetTestExample(vector<int64_t> &tokens) {
  testData.clear();
  testData.push_back(tokens);
  if(learningInfo.tgtWordClassesFilename.size() > 0) {
    testClassTgtSents.clear();
    testClassTgtSents.push_back( GetTgtWordClassSequence(tokens) );
  }
}

void LatentCrfPosTagger::Label(string &inputFilename, string &outputFilename, bool parallelize=true) {
  std::vector<std::vector<std::string> > tokens;
  StringUtils::ReadTokens(inputFilename, tokens);
  vector<vector<int> > labels;
  Label(tokens, labels, parallelize);
  if(!parallelize ||
     parallelize && learningInfo.mpiWorld->rank() == 0) {
    if(labelIntToString.size() > 0) {
      StringUtils::WriteTokens(outputFilename, labels, labelIntToString);
    } else {
      StringUtils::WriteTokens(outputFilename, labels);
    }
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
    if(labelIntToString.size()>0) {
      for(unsigned i = 0; i < labels.size(); ++i) {
        ss << labelIntToString[labels[i]] << " ";
      }
    } else {
      for(unsigned i = 0; i < labels.size(); ++i){
        ss << labels[i] << " ";
      }
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

void LatentCrfPosTagger::FireFeatures(int yI, int yIM1, unsigned sentId, int i, 
				  FastSparseVector<double> &activeFeatures) { 
    // fire the pos tagger features
    lambda->FireFeatures(yI, yIM1, sentId, GetObservableSequence(sentId), i, activeFeatures);

}

