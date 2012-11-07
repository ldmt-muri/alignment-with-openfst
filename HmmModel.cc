#include "HmmModel.h"

using namespace std;
using namespace fst;

// initialize model 1 scores
HmmModel::HmmModel(const string& srcIntCorpusFilename, 
		   const string& tgtIntCorpusFilename, 
		   const string& outputFilenamePrefix, 
		   const LearningInfo& learningInfo) {

  // TODO: use a constant for reproducible results
  srand(time(0));

  // set member variables
  this->srcCorpusFilename = srcIntCorpusFilename;
  this->tgtCorpusFilename = tgtIntCorpusFilename;
  this->outputPrefix = outputFilenamePrefix;
  this->learningInfo = learningInfo;

  // initialize the model parameters
  cerr << "init hmm params" << endl;
  stringstream initialModelFilename;
  initialModelFilename << outputPrefix << ".param.init";
  InitParams();
  PersistParams(initialModelFilename.str());

  // create the initial grammar FST
  cerr << "create grammar fst" << endl;
  CreateGrammarFst();
}

void HmmModel::Train() {

  // create tgt fsts
  cerr << "create tgt fsts" << endl;
  vector< VectorFst <LogTripleArc> > tgtFsts;
  CreateTgtFsts(tgtFsts);

  // training iterations
  cerr << "train!" << endl;
  LearnParameters(tgtFsts);

  // persist parameters
  cerr << "persist" << endl;
  PersistParams(outputPrefix + ".param.final");
}

// src fsts are 1st order markov models
void HmmModel::CreateSrcFsts(vector< VectorFst< LogTripleArc > >& srcFsts) {
  ifstream srcCorpus(srcCorpusFilename.c_str(), ios::in); 
  
  // for each line
  string line;
  while(getline(srcCorpus, line)) {
    
    // read the list of integers representing target tokens
    vector< int > intTokens;
    intTokens.push_back(NULL_SRC_TOKEN_ID);
    StringUtils::ReadIntTokens(line, intTokens);
    
    // create the fst
    VectorFst< LogTripleArc > srcFst;
    Create1stOrderSrcFst(intTokens, srcFst);
    srcFsts.push_back(srcFst);
    
    // for debugging
    // PrintFstSummary(tgtFst);
  }
  srcCorpus.close();
}


void HmmModel::CreateTgtFsts(vector< VectorFst< LogTripleArc > >& targetFsts) {
  ifstream tgtCorpus(tgtCorpusFilename.c_str(), ios::in); 
  
  // for each line
  string line;
  while(getline(tgtCorpus, line)) {
    
    // read the list of integers representing target tokens
    vector< int > intTokens;
    StringUtils::ReadIntTokens(line, intTokens);
    
    // create the fst
    VectorFst< LogTripleArc > tgtFst;
    int statesCount = intTokens.size() + 1;
    for(int stateId = 0; stateId < intTokens.size()+1; stateId++) {
      int temp = tgtFst.AddState();
      assert(temp == stateId);
      if(stateId == 0) continue;
      tgtFst.AddArc(stateId-1, 
		    LogTripleArc(intTokens[stateId-1], 
				 intTokens[stateId-1], 
				 LogTripleWeight::One(), 
				 stateId));
    }
    tgtFst.SetStart(0);
    tgtFst.SetFinal(intTokens.size(), LogTripleWeight::One());
    ArcSort(&tgtFst, ILabelCompare<LogTripleArc>());
    targetFsts.push_back(tgtFst);
    
    // for debugging
    // PrintFstSummary(tgtFst);
  }
  tgtCorpus.close();
}

void HmmModel::NormalizeFractionalCounts() {
  NormalizeParams(aFractionalCounts);
  NormalizeParams(tFractionalCounts);
}

// refactor variable names here (e.g. translations)
// normalizes ConditionalMultinomialParam parameters such that \sum_t p(t|s) = 1 \forall s
void HmmModel::NormalizeParams(ConditionalMultinomialParam& params) {
  // iterate over src tokens in the model
  for(ConditionalMultinomialParam::iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
    map< int, float > &translations = (*srcIter).second;
    float fTotalProb = 0.0;
    // iterate over tgt tokens logsumming over the logprob(tgt|src) 
    for(map< int, float >::iterator tgtIter = translations.begin(); tgtIter != translations.end(); tgtIter++) {
      float temp = (*tgtIter).second;
      fTotalProb += FstUtils::nExp(temp);
    }
    // exponentiate to find p(*|src) before normalization
    // iterate again over tgt tokens dividing p(tgt|src) by p(*|src)
    float fVerifyTotalProb = 0.0;
    for(map< int, float >::iterator tgtIter = translations.begin(); tgtIter != translations.end(); tgtIter++) {
      float fUnnormalized = FstUtils::nExp((*tgtIter).second);
      float fNormalized = fUnnormalized / fTotalProb;
      fVerifyTotalProb += fNormalized;
      float fLogNormalized = FstUtils::nLog(fNormalized);
      (*tgtIter).second = fLogNormalized;
    }
  }
}

void HmmModel::PrintParams() {
  PrintParams(aParams);
  PrintParams(tFractionalCounts);
}

// refactor variable names here (e.g. translations)
void HmmModel::PrintParams(const ConditionalMultinomialParam& params) {
  // iterate over src tokens in the model
  int counter = 0;
  for(ConditionalMultinomialParam::const_iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
    const map< int, float > &translations = (*srcIter).second;
    // iterate over tgt tokens 
    for(map< int, float >::const_iterator tgtIter = translations.begin(); tgtIter != translations.end(); tgtIter++) {
      cerr << "-logp(" << (*tgtIter).first << "|" << (*srcIter).first << ")=log(" << FstUtils::nExp((*tgtIter).second) << ")=" << (*tgtIter).second << endl;
    }
  } 
}

void HmmModel::PersistParams(ofstream& paramsFile, const ConditionalMultinomialParam& params) {
  for (ConditionalMultinomialParam::const_iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
    for (map<int, float>::const_iterator tgtIter = srcIter->second.begin(); tgtIter != srcIter->second.end(); tgtIter++) {
      // line format: 
      // srcTokenId tgtTokenId logP(tgtTokenId|srcTokenId) p(tgtTokenId|srcTokenId)
      paramsFile << srcIter->first << " " << tgtIter->first << " " << tgtIter->second << " " << FstUtils::nExp(tgtIter->second) << endl;
    }
  }
}

void HmmModel::PersistParams(const string& outputFilename) {
  ofstream paramsFile(outputFilename.c_str());
  cerr << "writing model params at " << outputFilename << endl;
  paramsFile << "=============== translation parameters p(tgtWord|srcWord) ============" << endl;
  PersistParams(paramsFile, tFractionalCounts);
  paramsFile << endl << "=============== alignment parameters p(a_i|a_{i-1}) ==================" << endl;
  PersistParams(paramsFile, aParams);
  paramsFile.close();
}

// finds out what are the parameters needed by reading hte corpus, and assigning initial weights based on the number of co-occurences
void HmmModel::InitParams() {
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

      // INITIALIZE TRANSLATION PARAMETERS
      int srcToken = srcTokens[i];
      // get the corresponding map of tgtTokens (and the corresponding probabilities)
      map<int, float> &tParamsGivenS_i = tFractionalCounts[srcToken];
      // for each tgtToken
      for (int j=0; j<tgtTokens.size(); j++) {
	int tgtToken = tgtTokens[j];
	// TODO: consider initializing these parameters with a uniform distribution instead of reflecting co-occurences. EM should figure it out on its own.
	// if this the first time the pair(tgtToken, srcToken) is experienced, give it a value of 1 (i.e. prob = exp(-1) ~= 1/3)
	if( tParamsGivenS_i.count(tgtToken) == 0) {
	  tParamsGivenS_i[tgtToken] = FstUtils::nLog(1/3.0);
	} else {
	  // otherwise, add nLog(1/3) to the original value, effectively counting the number of times 
	  // this srcToken-tgtToken pair appears in the corpus
	  tParamsGivenS_i[tgtToken] = Plus( LogWeight(tParamsGivenS_i[tgtToken]), LogWeight(FstUtils::nLog(1/3.0)) ).Value();
	}
      }

      // INITIALIZE ALIGNMENT PARAMETERS
      // TODO: It *might* be a good idea to initialize those parameters reflecting co-occurence statistics such that p(a=50|prev_a=30) < p(a=10|prev_a=30).
      //       EM should be able to figure it out on its own, though.
      for(int k=-1; k<srcTokens.size(); k++) {
      // assume that previous alignment = k, initialize p(i|k)
	aFractionalCounts[k][i] = FstUtils::nLog(1/3.0);
      }
      // also initialize aFractionalCounts[-1][i]
      aFractionalCounts[INITIAL_SRC_POS][i] = FstUtils::nLog(1/3.0);

    }
  }
  
  srcCorpus.close();
  tgtCorpus.close();
    
  NormalizeFractionalCounts();
  DeepCopy(aFractionalCounts, aParams);
}

// make a deep copy of parameters
void HmmModel::DeepCopy(const ConditionalMultinomialParam& original, 
			ConditionalMultinomialParam& duplicate) {
  // zero duplicate
  ClearParams(duplicate);

  // copy original into duplicate
  for(ConditionalMultinomialParam::const_iterator contextIter = original.begin(); 
      contextIter != original.end();
      contextIter ++) {
    for(MultinomialParam::const_iterator multIter = contextIter->second.begin();
	multIter != contextIter->second.end();
	multIter ++) {
      duplicate[contextIter->first][multIter->first] = multIter->second;
    }
  }
}

void HmmModel::CreateGrammarFst() {
  // clear grammar
  if (grammarFst.NumStates() > 0) {
    grammarFst.DeleteArcs(grammarFst.Start());
    grammarFst.DeleteStates();
    assert(grammarFst.NumStates() == 0);
  }
  
  // create the only state in this fst, and make it initial and final
  LogTripleArc::StateId dummy = grammarFst.AddState();
  assert(dummy == 0);
  grammarFst.SetStart(0);
  grammarFst.SetFinal(0, LogTripleWeight::One());
  int fromState = 0, toState = 0;
  for(ConditionalMultinomialParam::const_iterator srcIter = tFractionalCounts.begin(); srcIter != tFractionalCounts.end(); srcIter++) {
    for(MultinomialParam::const_iterator tgtIter = (*srcIter).second.begin(); tgtIter != (*srcIter).second.end(); tgtIter++) {
      int tgtToken = (*tgtIter).first;
      int srcToken = (*srcIter).first;
      float paramValue = (*tgtIter).second;
      grammarFst.AddArc(fromState, 
			LogTripleArc(tgtToken, 
				     srcToken, 
				     FstUtils::EncodeTriple(0,0,paramValue), 
				     toState));
    }
  }
  ArcSort(&grammarFst, ILabelCompare<LogTripleArc>());
  //  PrintFstSummary(grammarFst);
}

// assumptions:
// - first token in srcTokens is the NULL token (to represent null-alignments)
// - srcFst is assumed to be empty
//
// notes:
// - the structure of this FST is laid out such that each state encodes the previous non-null 
//   src position. the initial state is unique: it represents both the starting state the state
//   where all previous alignments are null-alignments.
// - if a source type is repeated, it will have multiple states corresponding to the different positions
// - the "1stOrder" part of the function name indicates this FST represents a first order markov process
//   for alignment transitions.
//
void HmmModel::Create1stOrderSrcFst(const vector<int>& srcTokens, VectorFst<LogTripleArc>& srcFst) {
  // enforce assumptions
  assert(srcTokens.size() > 0 && srcTokens[0] == NULL_SRC_TOKEN_ID);
  assert(srcFst.NumStates() == 0);

  // create one state per src position
  for(int i = 0; i < srcTokens.size(); i++) {
    int stateId = srcFst.AddState();
    // assumption that AddState() first returns a zero then increment ones
    assert(i == stateId);
  }

  // for each state
  for(int i = 0; i < srcTokens.size(); i++) {

    // for debugging only
    //    cerr << "srcTokens[" << i << "] = " << srcTokens[i] << endl;
    
    // set the initial/final states
    if(i == 0) {
      srcFst.SetStart(i);
    } else {
      srcFst.SetFinal(i, LogTripleWeight::One());
    }

    // we don't allow prevAlignment to be null alignment in our markov model. if a null alignment happens after alignment = 5, we use 5 as prevAlignment, not the null alignment. if null alignment happens before any non-null alignment, we use a special src position INITIAL_SRC_POS to indicate the prevAlignment
    int prevAlignment = i == 0? INITIAL_SRC_POS : i;

    // each state can go to itself with the null src token
    srcFst.AddArc(i, LogTripleArc(srcTokens[0], srcTokens[0], FstUtils::EncodeTriple(i, prevAlignment, aParams[prevAlignment][i]), i));

    // each state can go to states representing non-null alignments
    for(int j = 1; j < srcTokens.size(); j++) {
      srcFst.AddArc(i, LogTripleArc(srcTokens[j], srcTokens[j], FstUtils::EncodeTriple(j, prevAlignment, aParams[prevAlignment][j]), j));
    }
  }
 
  // arc sort to enable composition
  ArcSort(&srcFst, ILabelCompare<LogTripleArc>());

  // for debugging
  //  cerr << "=============SRC FST==========" << endl;
  //  cerr << FstUtils::PrintFstSummary(srcFst);
}

void HmmModel::ClearFractionalCounts() {
  ClearParams(tFractionalCounts);
  ClearParams(aFractionalCounts);
}

// zero all parameters
void HmmModel::ClearParams(ConditionalMultinomialParam& params) {
  for (ConditionalMultinomialParam::iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
    for (map<int, float>::iterator tgtIter = srcIter->second.begin(); tgtIter != srcIter->second.end(); tgtIter++) {
      tgtIter->second = FstUtils::LOG_ZERO;
    }
  }
}

void HmmModel::LearnParameters(vector< VectorFst< LogTripleArc > >& tgtFsts) {
  clock_t compositionClocks = 0, forwardBackwardClocks = 0, updatingFractionalCountsClocks = 0, normalizationClocks = 0;
  clock_t t00 = clock();
  do {
    clock_t t05 = clock();

    // create src fsts (these encode the aParams as weights on their arcs)
    cerr << "create src fsts" << endl;
    vector< VectorFst <LogTripleArc> > srcFsts;
    CreateSrcFsts(srcFsts);

    clock_t t10 = clock();
    float logLikelihood = 0, validationLogLikelihood = 0;
    //    cerr << "iteration's loglikelihood = " << logLikelihood << endl;
    
    // this vector will be used to accumulate fractional counts of parameter usages
    ClearFractionalCounts();
    
    // iterate over sentences
    int sentsCounter = 0;
    for( vector< VectorFst< LogTripleArc > >::const_iterator tgtIter = tgtFsts.begin(), srcIter = srcFsts.begin(); 
	 tgtIter != tgtFsts.end() && srcIter != srcFsts.end(); 
	 tgtIter++, srcIter++) {

      // build the alignment fst
      clock_t t20 = clock();
      const VectorFst< LogTripleArc > &tgtFst = *tgtIter, &srcFst = *srcIter;
      VectorFst< LogTripleArc > alignmentFst, temp;
      Compose(tgtFst, grammarFst, &temp);
      Compose(temp, srcFst, &alignmentFst);
      compositionClocks += clock() - t20;
      
      // run forward/backward for this sentence
      clock_t t30 = clock();
      vector<LogTripleWeight> alphas, betas;
      ShortestDistance(alignmentFst, &alphas, false);
      ShortestDistance(alignmentFst, &betas, true);
      float fSentLogLikelihood, dummy;
      FstUtils::DecodeTriple(betas[alignmentFst.Start()], 
			     dummy, dummy, fSentLogLikelihood);
      forwardBackwardClocks += clock() - t30;
      
      // compute and accumulate fractional counts for model parameters
      clock_t t40 = clock();
      bool excludeFractionalCountsInThisSent = 
	learningInfo.useEarlyStopping && 
	sentsCounter % learningInfo.trainToDevDataSize == 0;
      for (int stateId = 0; !excludeFractionalCountsInThisSent && stateId < alignmentFst.NumStates() ;stateId++) {
	for (ArcIterator<VectorFst< LogTripleArc > > arcIter(alignmentFst, stateId);
	     !arcIter.Done();
	     arcIter.Next()) {

	  // decode arc information
	  int srcToken = arcIter.Value().olabel, tgtToken = arcIter.Value().ilabel;
	  int fromState = stateId, toState = arcIter.Value().nextstate;
	  float fCurrentSrcPos, fPrevSrcPos, arcLogProb;
	  FstUtils::DecodeTriple(arcIter.Value().weight, fCurrentSrcPos, fPrevSrcPos, arcLogProb);
	  int currentSrcPos = (int) fCurrentSrcPos, prevSrcPos = (int) fPrevSrcPos;

	  // probability of using this parameter given this sentence pair and the previous model
	  float alpha, beta, dummy;
	  FstUtils::DecodeTriple(alphas[fromState], dummy, dummy, alpha);
	  FstUtils::DecodeTriple(betas[toState], dummy, dummy, beta);
	  float fNormalizedPosteriorLogProb = (alpha + arcLogProb + beta) - fSentLogLikelihood;
	    
	  // update tFractionalCounts
	  tFractionalCounts[srcToken][tgtToken] = 
	    Plus(LogWeight(tFractionalCounts[srcToken][tgtToken]), 
		 LogWeight(fNormalizedPosteriorLogProb)).Value();
	  // update aFractionalCounts
	  aFractionalCounts[prevSrcPos][currentSrcPos] = 
	    Plus(LogWeight(aFractionalCounts[prevSrcPos][currentSrcPos]),
		 LogWeight(fNormalizedPosteriorLogProb)).Value();
	  
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
    NormalizeFractionalCounts();
    DeepCopy(aFractionalCounts, aParams);
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
  cerr << endl;
}

// sample an integer from a multinomial
int HmmModel::SampleFromMultinomial(const MultinomialParam params) {
  // generate a pseudo random number between 0 and 1
  double randomProb = ((double) rand() / (RAND_MAX));

  // find the lucky value
  for(MultinomialParam::const_iterator paramIter = params.begin(); 
      paramIter != params.end(); 
      paramIter++) {
    double valueProb = FstUtils::nExp(paramIter->second);
    if(randomProb <= valueProb) {
      return paramIter->first;
    } else {
      randomProb -= valueProb;
    }
  }

  // if you get here, one of the following two things happened: \sum valueProb_i > 1 OR randomProb > 1
  assert(false);
}

// assumptions:
// - both aParams and tFractionalCounts are properly normalized logProbs
// sample both an alignment and a translation, given src sentence and tgt length
void HmmModel::SampleAT(const vector<int>& srcTokens, int tgtLength, vector<int>& tgtTokens, vector<int>& alignments, double& hmmLogProb) {

  // intialize
  int prevAlignment = INITIAL_SRC_POS;
  hmmLogProb = 0;

  // for each target position,
  for(; tgtLength > 0; tgtLength--) {

    // sample a src position (i.e. an alignment)
    int currentAlignment;
    do {
      currentAlignment = SampleFromMultinomial(aParams[prevAlignment]);
    } while(currentAlignment >= srcTokens.size());
    alignments.push_back(currentAlignment);
    
    // sample a translation
    int currentTranslation = SampleFromMultinomial(tFractionalCounts[srcTokens[currentAlignment]]);
    tgtTokens.push_back(currentTranslation);

    // update the sample probability according to the model
    hmmLogProb += aParams[prevAlignment][currentAlignment];
    hmmLogProb += tFractionalCounts[srcTokens[currentAlignment]][currentTranslation];

    // update prevAlignment
    if(currentAlignment != NULL_SRC_TOKEN_ID) {
      prevAlignment = currentAlignment;
    }
  }

  assert(tgtTokens.size() == 3 && alignments.size() == 3);
}

// TODO: not implemented
// given the current model, align the corpus
void HmmModel::Align() {
  
}
