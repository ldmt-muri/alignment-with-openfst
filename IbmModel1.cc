#include "IbmModel1.h"

using namespace std;
using namespace fst;

// initialize model 1 scores
IbmModel1::IbmModel1(const string& srcIntCorpusFilename, const string& tgtIntCorpusFilename, const string& outputFilenamePrefix, const LearningInfo& learningInfo) {
  // set member variables
  this->srcCorpusFilename = srcIntCorpusFilename;
  this->tgtCorpusFilename = tgtIntCorpusFilename;
  this->outputPrefix = outputFilenamePrefix;
  this->learningInfo = learningInfo;

  // encode and memorize training data
  vocabEncoder.useUnk = false;
  vocabEncoder.Read(srcCorpusFilename, srcSents);
  vocabEncoder.Read(tgtCorpusFilename, tgtSents);
  assert(srcSents.size() > 0 && srcSents.size() == tgtSents.size());

  // initialize the model parameters
  cerr << "init model1 params" << endl;
  stringstream initialModelFilename;
  initialModelFilename << outputPrefix << ".param.init";
  InitParams();
  PersistParams(initialModelFilename.str());

  // create the initial grammar FST
  cerr << "create grammar fst" << endl;
  CreateGrammarFst();

}

void IbmModel1::Train() {

  // create tgt fsts
  cerr << "create tgt fsts" << endl;
  vector< VectorFst <LogArc> > tgtFsts;
  CreateTgtFsts(tgtFsts);

  // training iterations
  cerr << "train!" << endl;
  LearnParameters(tgtFsts);

  // persist parameters
  cerr << "persist" << endl;
  PersistParams(outputPrefix + ".param.final");
}

void IbmModel1::CreateTgtFsts(vector< VectorFst< LogArc > >& targetFsts) {

  for(unsigned i = 0; i < tgtSents.size(); i++) {
    // read the list of integers representing target tokens
    vector< int > &intTokens  =tgtSents[i];
    
    // create the fst
    VectorFst< LogArc > tgtFst;
    int statesCount = intTokens.size() + 1;
    for(int stateId = 0; stateId < intTokens.size()+1; stateId++) {
      int temp = tgtFst.AddState();
      assert(temp == stateId);
      if(stateId == 0) continue;
      tgtFst.AddArc(stateId-1, LogArc(intTokens[stateId-1], intTokens[stateId-1], 0, stateId));
    }
    tgtFst.SetStart(0);
    tgtFst.SetFinal(intTokens.size(), 0);
    ArcSort(&tgtFst, ILabelCompare<LogArc>());
    targetFsts.push_back(tgtFst);
    
    // for debugging
    // PrintFstSummary(tgtFst);
  }
}

// normalizes the parameters such that \sum_t p(t|s) = 1 \forall s
void IbmModel1::NormalizeParams() {
  MultinomialParams::NormalizeParams<int>(params);
}

void IbmModel1::PrintParams() {
  MultinomialParams::PrintParams<int>(params);
}

void IbmModel1::PersistParams(const string& outputFilename) {
  MultinomialParams::PersistParams1(outputFilename, params, vocabEncoder);
}

// finds out what are the parameters needed by reading hte corpus, and assigning initial weights based on the number of co-occurences
void IbmModel1::InitParams() {
  for(unsigned sentId = 0; sentId < srcSents.size(); sentId++) {
    // read the list of integers representing target tokens
    vector< int > &tgtTokens = tgtSents[sentId], &srcTokens = srcSents[sentId];
    // we want to allow target words to align to NULL (which has srcTokenId = 1).
    if(srcTokens[0] != NULL_SRC_TOKEN_ID) {
      srcTokens.insert(srcTokens.begin(), NULL_SRC_TOKEN_ID);
    } 
    
    // for each srcToken
    for(int i=0; i<srcTokens.size(); i++) {
      int srcToken = srcTokens[i];
      // get the corresponding map of tgtTokens (and the corresponding probabilities)
      map<int, double> &translations = params.params[srcToken];
      
      // for each tgtToken
      for (int j=0; j<tgtTokens.size(); j++) {
	int tgtToken = tgtTokens[j];
	// if this the first time the pair(tgtToken, srcToken) is experienced, give it a value of 1 (i.e. prob = exp(-1) ~= 1/3)
	if( translations.count(tgtToken) == 0) {
	  translations[tgtToken] = FstUtils::nLog(1/3.0);
	} else {
	  // otherwise, add nLog(1/3) to the original value, effectively counting the number of times 
	  // this srcToken-tgtToken pair appears in the corpus
	  translations[tgtToken] = Plus( LogWeight(translations[tgtToken]), LogWeight(FstUtils::nLog(1/3.0)) ).Value();
	}
      }
    }
  }
    
  NormalizeParams();
}

void IbmModel1::CreateGrammarFst() {
  // clear grammar
  if (grammarFst.NumStates() > 0) {
    grammarFst.DeleteArcs(grammarFst.Start());
    grammarFst.DeleteStates();
  }
  
  // create the only state in this fst, and make it initial and final
  LogArc::StateId dummy = grammarFst.AddState();
  assert(dummy == 0);
  grammarFst.SetStart(0);
  grammarFst.SetFinal(0, 0);
  int fromState = 0, toState = 0;
  for(map<int, MultinomialParams::MultinomialParam>::const_iterator srcIter = params.params.begin(); srcIter != params.params.end(); srcIter++) {
    for(MultinomialParams::MultinomialParam::const_iterator tgtIter = srcIter->second.begin(); tgtIter != srcIter->second.end(); tgtIter++) {
      int tgtToken = tgtIter->first;
      int srcToken = srcIter->first;
      double paramValue = tgtIter->second;
      grammarFst.AddArc(fromState, LogArc(tgtToken, srcToken, paramValue, toState));
    }
  }
  ArcSort(&grammarFst, ILabelCompare<LogArc>());
  //  PrintFstSummary(grammarFst);
}

void IbmModel1::CreatePerSentGrammarFsts(vector< VectorFst< LogArc > >& perSentGrammarFsts) {
  
  for(unsigned sentId = 0; sentId < srcSents.size(); sentId++) {
    vector<int> &srcTokens = srcSents[sentId];
    vector<int> &tgtTokensVector = tgtSents[sentId];
    set<int> tgtTokens(tgtTokensVector.begin(), tgtTokensVector.end());

    // allow null alignments
    assert(srcTokens[0] == NULL_SRC_TOKEN_ID);
    
    // create the fst
    VectorFst< LogArc > grammarFst;
    int stateId = grammarFst.AddState();
    assert(stateId == 0);
    for(vector<int>::const_iterator srcTokenIter = srcTokens.begin(); srcTokenIter != srcTokens.end(); srcTokenIter++) {
      for(set<int>::const_iterator tgtTokenIter = tgtTokens.begin(); tgtTokenIter != tgtTokens.end(); tgtTokenIter++) {
	grammarFst.AddArc(stateId, LogArc(*tgtTokenIter, *srcTokenIter, params[*srcTokenIter][*tgtTokenIter], stateId));	
      }
    }
    grammarFst.SetStart(stateId);
    grammarFst.SetFinal(stateId, 0);
    ArcSort(&grammarFst, ILabelCompare<LogArc>());
    perSentGrammarFsts.push_back(grammarFst);
    
  }
}

// zero all parameters
void IbmModel1::ClearParams() {
  for (map<int, MultinomialParams::MultinomialParam>::iterator srcIter = params.params.begin(); srcIter != params.params.end(); srcIter++) {
    for (MultinomialParams::MultinomialParam::iterator tgtIter = srcIter->second.begin(); tgtIter != srcIter->second.end(); tgtIter++) {
      tgtIter->second = FstUtils::LOG_ZERO;
    }
  }
}

void IbmModel1::LearnParameters(vector< VectorFst< LogArc > >& tgtFsts) {
  clock_t compositionClocks = 0, forwardBackwardClocks = 0, updatingFractionalCountsClocks = 0, grammarConstructionClocks = 0, normalizationClocks = 0;
  clock_t t00 = clock();
  do {
    clock_t t05 = clock();
    vector< VectorFst< LogArc > > perSentGrammarFsts;
    CreatePerSentGrammarFsts(perSentGrammarFsts);
    grammarConstructionClocks += clock() - t05;

    clock_t t10 = clock();
    float logLikelihood = 0, validationLogLikelihood = 0;
    //    cout << "iteration's loglikelihood = " << logLikelihood << endl;
    
    // this vector will be used to accumulate fractional counts of parameter usages
    ClearParams();
    
    // iterate over sentences
    int sentsCounter = 0;
    for( vector< VectorFst< LogArc > >::const_iterator tgtIter = tgtFsts.begin(), grammarIter = perSentGrammarFsts.begin(); 
	 tgtIter != tgtFsts.end() && grammarIter != perSentGrammarFsts.end(); 
	 tgtIter++, grammarIter++) {
      
      // build the alignment fst
      clock_t t20 = clock();
      VectorFst< LogArc > tgtFst = *tgtIter, perSentGrammarFst = *grammarIter, alignmentFst;
      Compose(tgtFst, perSentGrammarFst, &alignmentFst);
      compositionClocks += clock() - t20;
      //FstUtils::PrintFstSummary(alignmentFst);
      
      // run forward/backward for this sentence
      clock_t t30 = clock();
      vector<LogWeight> alphas, betas;
      ShortestDistance(alignmentFst, &alphas, false);
      ShortestDistance(alignmentFst, &betas, true);
      float fSentLogLikelihood = betas[alignmentFst.Start()].Value();
      forwardBackwardClocks += clock() - t30;
      //      cout << "sent's shifted log likelihood = " << fShiftedSentLogLikelihood << endl;
      //      float fSentLikelihood = exp(-1.0 * fShiftedSentLogLikelihood) / alignmentsCount;
      //      cout << "sent's likelihood = " << fSentLikelihood << endl;
      //      float fSentLogLikelihood = nLog(fSentLikelihood);
      //      cout << "nLog(alignmentsCount) = nLog(" << alignmentsCount << ") = " << nLog(alignmentsCount) << endl;
      //      cout << "sent's loglikelihood = " << fSentLogLikelihood << endl;
      
      // compute and accumulate fractional counts for model parameters
      clock_t t40 = clock();
      bool excludeFractionalCountsInThisSent = 
	learningInfo.useEarlyStopping && 
	sentsCounter % learningInfo.trainToDevDataSize == 0;
      for (int stateId = 0; !excludeFractionalCountsInThisSent && stateId < alignmentFst.NumStates() ;stateId++) {
	for (ArcIterator<VectorFst< LogArc > > arcIter(alignmentFst, stateId);
	     !arcIter.Done();
	     arcIter.Next()) {
	  int srcToken = arcIter.Value().olabel, tgtToken = arcIter.Value().ilabel;
	  int fromState = stateId, toState = arcIter.Value().nextstate;
	  
	  // probability of using this parameter given this sentence pair and the previous model
	  LogWeight currentParamLogProb = arcIter.Value().weight;
	  LogWeight unnormalizedPosteriorLogProb = Times(Times(alphas[fromState], currentParamLogProb), betas[toState]);
	  //float fUnnormalizedPosteriorProb = exp(-1.0 * unnormalizedPosteriorLogProb.Value());
	  //float fNormalizedPosteriorProb = (fUnnormalizedPosteriorProb / alignmentsCount) / fSentLikelihood;
	  //float fNormalizedPosteriorLogProb = -1.0 * log(fNormalizedPosteriorProb);
	  float fNormalizedPosteriorLogProb = unnormalizedPosteriorLogProb.Value() - fSentLogLikelihood;
	  
	  // logging
	  /*
	    if(srcToken == 2 && tgtToken == 2){
	    
	    cout << "The alignment FST looks like this: " << endl;
	    cout << "================================== " << endl;
	    PrintFstSummary(alignmentFst);
	    
	    cout << endl << "Updates on prob(2|2): " << endl;
	    cout << "===================== " << endl;
	    cout << " before: " << params[srcToken][tgtToken] << endl;
	    cout << " before: logProb(2|2) = " << currentParamLogProb.Value() << endl;
	    cout << " alpha[fromState] = " << alphas[fromState] << endl;
	    cout << " beta[toState] = " << betas[toState] << endl;
	    cout << " unnormalized logProb(2-2|sent) = alpha[fromState] logTimes logProb(2|2) logTimes beta[toState] = " << unnormalizedPosteriorLogProb.Value() << endl;
	    cout << " unnormalized prob(2-2|sent) = " << fUnnormalizedPosteriorProb << endl;
	    cout << " prob(2-2|sent) = " << fNormalizedPosteriorProb << endl;
	    cout << " logProb(2-2|sent) = " << fNormalizedPosteriorLogProb << endl;
	    cout << " adding: " << fNormalizedPosteriorLogProb << endl;
	    }
	  */
	    
	  // append the fractional count for this parameter
	  params[srcToken][tgtToken] = Plus(LogWeight(params[srcToken][tgtToken]), LogWeight(fNormalizedPosteriorLogProb)).Value();
	  
	  // logging
	  /*
	    if(srcToken == 2 && tgtToken == 2){
	    cout << " after: " << params[srcToken][tgtToken] << endl << endl;
	    }
	  */
	  
	}
      }
      updatingFractionalCountsClocks += clock() - t40;
      
      // update the iteration log likelihood with this sentence's likelihod
      if(excludeFractionalCountsInThisSent) {
	validationLogLikelihood += fSentLogLikelihood;
      } else {
	logLikelihood += fSentLogLikelihood;
      }
      //	cout << "iteration's loglikelihood = " << logLikelihood << endl;
      
      // logging
      if (++sentsCounter % 1000 == 0) {
	cerr << sentsCounter << " sents processed. iterationLoglikelihood = " << logLikelihood <<  endl;
      }
    }
    
    // normalize fractional counts such that \sum_t p(t|s) = 1 \forall s
    clock_t t50 = clock();
    NormalizeParams();
    normalizationClocks += clock() - t50;
    
    // persist parameters
    if(false) {
      stringstream filename;
      filename << outputPrefix << ".param." << learningInfo.iterationsCount;
      PersistParams(filename.str());
    }
    
    // create the new grammar
    clock_t t60 = clock();
    CreateGrammarFst();
    grammarConstructionClocks += clock() - t60;

    // logging
    cerr << "iterations # " << learningInfo.iterationsCount << " - total loglikelihood = " << logLikelihood << endl;
    
    // update learningInfo
    learningInfo.logLikelihood.push_back(logLikelihood);
    learningInfo.validationLogLikelihood.push_back(validationLogLikelihood);
    learningInfo.iterationsCount++;
    
    // check for convergence
  } while(!learningInfo.IsModelConverged());

  // logging
  cerr << endl;
  cerr << "trainTime        = " << (float) (clock() - t00) / CLOCKS_PER_SEC << " sec." << endl;
  cerr << "compositionTime  = " << (float) compositionClocks / CLOCKS_PER_SEC << " sec." << endl;
  cerr << "forward/backward = " << (float) forwardBackwardClocks / CLOCKS_PER_SEC << " sec." << endl;
  cerr << "fractionalCounts = " << (float) updatingFractionalCountsClocks / CLOCKS_PER_SEC << " sec." << endl;
  cerr << "normalizeClocks  = " << (float) normalizationClocks / CLOCKS_PER_SEC << " sec." << endl;
  cerr << "grammarConstruct = " << (float) grammarConstructionClocks / CLOCKS_PER_SEC << " sec." << endl;
  cerr << endl;
}

// given the current model, align the corpus
void IbmModel1::Align() {
  Align(outputPrefix + ".train.align");
}

void IbmModel1::Align(const string &alignmentsFilename) {
  ofstream outputAlignments;
  outputAlignments.open(alignmentsFilename.c_str(), ios::out);

  vector< VectorFst< LogArc > > perSentGrammarFsts;
  CreatePerSentGrammarFsts(perSentGrammarFsts);
  vector< VectorFst <LogArc> > tgtFsts;
  CreateTgtFsts(tgtFsts);

  assert(tgtFsts.size() == srcSents.size());
  assert(perSentGrammarFsts.size() == srcSents.size());
  assert(tgtSents.size() == srcSents.size());

  for(unsigned sentId = 0; sentId < srcSents.size(); sentId++) {
    vector<int> &srcSent = srcSents[sentId], &tgtSent = tgtSents[sentId];
    VectorFst< LogArc > &perSentGrammarFst = perSentGrammarFsts[sentId], &tgtFst = tgtFsts[sentId], alignmentFst;
    
    // given a src token id, what are the possible src position (in this sentence)
    map<int, set<int> > srcTokenToSrcPos;
    for(unsigned srcPos = 0; srcPos < srcSent.size(); srcPos++) {
      srcTokenToSrcPos[ srcSent[srcPos] ].insert(srcPos);
    }
    
    // build alignment fst and compute potentials
    Compose(tgtFst, perSentGrammarFst, &alignmentFst);
    vector<LogWeight> alphas, betas;
    ShortestDistance(alignmentFst, &alphas, false);
    ShortestDistance(alignmentFst, &betas, true);
    double fSentLogLikelihood = betas[alignmentFst.Start()].Value();
    
    // tropical has the path property. we need this property to compute the shortest path
    VectorFst< StdArc > alignmentFstProbsWithPathProperty, bestAlignment, corrected;
    ArcMap(alignmentFst, &alignmentFstProbsWithPathProperty, LogToTropicalMapper());
    ShortestPath(alignmentFstProbsWithPathProperty, &bestAlignment);
   
    // fix labels
    // - the input labels are tgt positions
    // - the output labels are the corresponding src positions according to the alignment
    // traverse the transducer beginning with the start state
    stringstream alignmentsLine;
    int startState = bestAlignment.Start();
    int currentState = startState;
    int tgtPos = 0;
    while(bestAlignment.Final(currentState) == LogWeight::Zero()) {
      // get hold of the arc
      ArcIterator< VectorFst< StdArc > > aiter(bestAlignment, currentState);
      // identify the next state
      int nextState = aiter.Value().nextstate;
      // skip epsilon arcs
      if(aiter.Value().ilabel == FstUtils::EPSILON && aiter.Value().olabel == FstUtils::EPSILON) {
	currentState = nextState;
	continue;
      }
      // update tgt pos
      tgtPos++;
      // find src pos
      int srcPos = 0;
      assert(srcTokenToSrcPos[aiter.Value().olabel].size() > 0);
      if(srcTokenToSrcPos[aiter.Value().olabel].size() == 1) {
	srcPos = *(srcTokenToSrcPos[aiter.Value().olabel].begin());
      } else {
	float distortion = 100;
	for(set<int>::iterator srcPosIter = srcTokenToSrcPos[aiter.Value().olabel].begin(); srcPosIter != srcTokenToSrcPos[aiter.Value().olabel].end(); srcPosIter++) {
	  if(*srcPosIter - tgtPos < distortion) {
	    srcPos = *srcPosIter;
	    distortion = abs(*srcPosIter - tgtPos);
	  }
	}
      }
      // print the alignment giza-style
      // giza++ does not write null alignments
      if(srcPos != 0) {
	// giza++ uses zero-based src and tgt positions, and writes the src position first
	alignmentsLine << (srcPos - 1) << "-" << (tgtPos - 1) << " ";
      }
      // this state shouldn't have other arcs!
      aiter.Next();
      assert(aiter.Done());
      // move forward to the next state
      currentState = nextState;
    }
    alignmentsLine << endl;

    // write the best alignment to file
    outputAlignments << alignmentsLine.str();
  }
  outputAlignments.close();
}


// TODO: not implemented 
// given the current model, align a test set
void IbmModel1::AlignTestSet(const string &srcTestSetFilename, const string &tgtTestSetFilename, const string &outputAlignmentsFilename) {
  assert(false);
}
