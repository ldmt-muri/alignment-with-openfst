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

  // TODO: this only works because I'm working on a small corpus. if we have millions of sentence pairs, 
  // we probably need to create tgtFsts and srcFsts on the fly.

  // create tgt fsts
  cerr << "create tgt fsts" << endl;
  vector< VectorFst <LogArc> > tgtFsts;
  CreateTgtFsts(tgtFsts);

  // create src fsts
  cerr << "create src fsts" << endl;
  vector< VectorFst <LogArc> > srcFsts;
  CreateSrcFsts(srcFsts);

  // training iterations
  cerr << "train!" << endl;
  LearnParameters(srcFsts, tgtFsts);

  // persist parameters
  cerr << "persist" << endl;
  PersistParams(outputPrefix + ".param.final");
}

void IbmModel1::CreateTgtFsts(vector< VectorFst< LogArc > >& targetFsts) {
  ifstream tgtCorpus(tgtCorpusFilename.c_str(), ios::in); 
  
  // for each line
  string line;
  while(getline(tgtCorpus, line)) {
    
    // read the list of integers representing target tokens
    vector< int > intTokens;
    StringUtils::ReadIntTokens(line, intTokens);
    
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
  tgtCorpus.close();
}

// normalizes the parameters such that \sum_t p(t|s) = 1 \forall s
void IbmModel1::NormalizeParams() {
  // iterate over src tokens in the model
  for(Model1Param::iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
    //cout << "=======================" << endl << "normalizing srcToken = " << (*srcIter).first << endl;
    map< int, float > *translations = &(*srcIter).second;
    float fTotalProb = 0.0;
    //cout << "totalProb = " << fTotalProb << endl;
    // iterate over tgt tokens logsumming over the logprob(tgt|src) 
    for(map< int, float >::iterator tgtIter = translations->begin(); tgtIter != translations->end(); tgtIter++) {
      LogWeight temp = (*tgtIter).second;
      fTotalProb += exp(-1.0 * temp.Value());
      //cout << "fTotalProb += " << exp(-1.0 * temp.Value()) << "(i.e. e^-" << temp.Value() << ") ==> " << fTotalProb << endl;
    }
    // exponentiate to find p(*|src) before normalization
    //    float fLogTotalProb = (float) logTotalProb.Value();
    //cout << "totalProb = " << fTotalProb << endl << endl;
    // iterate again over tgt tokens dividing p(tgt|src) by p(*|src)
    float fVerifyTotalProb = 0.0;
    for(map< int, float >::iterator tgtIter = translations->begin(); tgtIter != translations->end(); tgtIter++) {
      float fUnnormalized = exp( -1.0 * (*tgtIter).second );
      float fNormalized = fUnnormalized / fTotalProb;
      fVerifyTotalProb += fNormalized;
      float fLogNormalized = -1 * log(fNormalized);
      //cout << "prob(" << (*tgtIter).first << "|" << (*srcIter).first << ") = " << fUnnormalized << " ==> " << fNormalized << endl;
      //cout << "-logprob(" << (*tgtIter).first << "|" << (*srcIter).first << ") = " << (*tgtIter).second << " ==> ";
      (*tgtIter).second = fLogNormalized;
      //cout << (*tgtIter).second << endl;
    }
    //cout << "verify totalProb = " << fVerifyTotalProb << endl << endl;
  }
}

void IbmModel1::PrintParams() {
  // iterate over src tokens in the model
  int counter =0;
  for(Model1Param::const_iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
    map< int, float > translations = (*srcIter).second;
    // iterate over tgt tokens 
    for(map< int, float >::const_iterator tgtIter = translations.begin(); tgtIter != translations.end(); tgtIter++) {
      cerr << "-logp(" << (*tgtIter).first << "|" << (*srcIter).first << ")=log(" << exp(-1.0 * (*tgtIter).second) << ")=" << (*tgtIter).second << endl;
    }
  } 
}

void IbmModel1::PersistParams(const string& outputFilename) {
  ofstream paramsFile(outputFilename.c_str());
  cerr << "writing model params at " << outputFilename << endl;
  
  for (Model1Param::const_iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
    for (map<int, float>::const_iterator tgtIter = srcIter->second.begin(); tgtIter != srcIter->second.end(); tgtIter++) {
      // line format: 
      // srcTokenId tgtTokenId logP(tgtTokenId|srcTokenId) p(tgtTokenId|srcTokenId)
      paramsFile << srcIter->first << " " << tgtIter->first << " " << tgtIter->second << " " << exp(-1.0 * tgtIter->second) << endl;
    }
  }
  paramsFile.close();
}

// finds out what are the parameters needed by reading hte corpus, and assigning initial weights based on the number of co-occurences
void IbmModel1::InitParams() {
  ifstream srcCorpus(srcCorpusFilename.c_str(), ios::in);
  ifstream tgtCorpus(tgtCorpusFilename.c_str(), ios::in); 
  
  // for each line
  string srcLine, tgtLine;
  while(getline(tgtCorpus, tgtLine) && getline(srcCorpus, srcLine)) {
    
    // read the list of integers representing target tokens
    vector< int > tgtTokens, srcTokens;
    StringUtils::ReadIntTokens(srcLine, srcTokens);
    // we want to allow target words to align to NULL (which has srcTokenId = 1).
    srcTokens.push_back(NULL_SRC_TOKEN_ID); 
    StringUtils::ReadIntTokens(tgtLine, tgtTokens);
    
    // for each srcToken
    for(int i=0; i<srcTokens.size(); i++) {
      int srcToken = srcTokens[i];
      // get the corresponding map of tgtTokens (and the corresponding probabilities)
      map<int, float> *translations = &(params[srcToken]);
      
      // for each tgtToken
      for (int j=0; j<tgtTokens.size(); j++) {
	int tgtToken = tgtTokens[j];
	// if this the first time the pair(tgtToken, srcToken) is experienced, give it a value of 1 (i.e. prob = exp(-1) ~= 1/3)
	if( translations->count(tgtToken) == 0) {
	  (*translations)[tgtToken] = FstUtils::nLog(1/3.0);
	} else {
	  // otherwise, add nLog(1/3) to the original value, effectively counting the number of times 
	  // this srcToken-tgtToken pair appears in the corpus
	  (*translations)[tgtToken] = Plus( LogWeight((*translations)[tgtToken]), LogWeight(FstUtils::nLog(1/3.0)) ).Value();
	}
      }
    }
  }
  
  srcCorpus.close();
  tgtCorpus.close();
    
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
  for(Model1Param::const_iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
    for(map<int,float>::const_iterator tgtIter = (*srcIter).second.begin(); tgtIter != (*srcIter).second.end(); tgtIter++) {
      int tgtToken = (*tgtIter).first;
      int srcToken = (*srcIter).first;
      float paramValue = (*tgtIter).second;
      grammarFst.AddArc(fromState, LogArc(tgtToken, srcToken, paramValue, toState));
    }
  }
  ArcSort(&grammarFst, ILabelCompare<LogArc>());
  //  PrintFstSummary(grammarFst);
}

  // zero all parameters
void IbmModel1::ClearParams() {
  for (Model1Param::iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
    for (map<int, float>::iterator tgtIter = srcIter->second.begin(); tgtIter != srcIter->second.end(); tgtIter++) {
      tgtIter->second = FstUtils::LOG_ZERO;
    }
  }
}

void IbmModel1::LearnParameters(vector< VectorFst< LogArc > >& srcFsts, vector< VectorFst< LogArc > >& tgtFsts) {
  clock_t compositionClocks = 0, forwardBackwardClocks = 0, updatingFractionalCountsClocks = 0, grammarConstructionClocks = 0, normalizationClocks = 0;
  clock_t t00 = clock();
  do {
    clock_t t10 = clock();
    float logLikelihood = 0, validationLogLikelihood = 0;
    //    cout << "iteration's loglikelihood = " << logLikelihood << endl;
    
    // this vector will be used to accumulate fractional counts of parameter usages
    ClearParams();
    
    // iterate over sentences
    int sentsCounter = 0;
    for( vector< VectorFst< LogArc > >::const_iterator tgtIter = tgtFsts.begin(), srcIter = srcFsts.begin(); 
	 tgtIter != tgtFsts.end() && srcIter != srcFsts.end(); 
	 tgtIter++, srcIter++) {
      
      // build the alignment fst
      clock_t t20 = clock();
      VectorFst< LogArc > tgtFst = *tgtIter, srcFst = *srcIter, temp, alignmentFst;
      Compose(tgtFst, grammarFst, &temp);
      ArcSort(&temp, ILabelCompare<LogArc>());
      Compose(temp, srcFst, &alignmentFst);
      compositionClocks += clock() - t20;
      //      PrintFstSummary(alignmentFst);
      
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
	cerr << sentsCounter << " sents processed.." << endl;
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
    cerr << "iterations # " << learningInfo.iterationsCount << " - total loglikelihood = " << logLikelihood << endl << endl;
    
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

// returns a list of acceptors of the source sentences in any order. 
// Each acceptor has a single state with arcs representing src tokens in addition to NULL (srcTokenId = 0)
void IbmModel1::CreateSrcFsts(vector< VectorFst< LogArc > >& srcFsts) {
  ifstream srcCorpus(srcCorpusFilename.c_str(), ios::in); 
  
  // for each line
  string line;
  while(getline(srcCorpus, line)) {
    
    // read the list of integers representing source tokens
    vector< int > intTokens;
    StringUtils::ReadIntTokens(line, intTokens);
    // allow null alignments
    intTokens.push_back(NULL_SRC_TOKEN_ID);
    
    // create the fst
    VectorFst< LogArc > srcFst;
    int stateId = srcFst.AddState();
    assert(stateId == 0);
    for(vector<int>::const_iterator tokenIter = intTokens.begin(); tokenIter != intTokens.end(); tokenIter++) {
      srcFst.AddArc(stateId, LogArc(*tokenIter, *tokenIter, 0, stateId));
    }
    srcFst.SetStart(stateId);
    srcFst.SetFinal(stateId, 0);
    ArcSort(&srcFst, ILabelCompare<LogArc>());
    srcFsts.push_back(srcFst);
    
    // for debugging
    //    PrintFstSummary(srcFst);
  }
  srcCorpus.close();
}

// TODO: not implemented
// given the current model, align the corpus
void IbmModel1::Align() {
  
}
