
// working code available at https://github.com/ldmt-muri/alignment-with-openfst

// data structure used to hold the parameters of IBM model1. 
// Outer keys are source token IDs
// Inside keys are target token IDs
// Inside values are p(target token|source token)
typedef map<int, map< int, float > >  Model1Param;

class IbmModel1 {

  // fractional counts
  Model1Param params;

  // model parameters (fst version)
  VectorFst<LogArc> grammarFst;

  // stopping criteria and iteration stats
  LearningInfo learningInfo;  

  ////
  // construct a linear-chain FSA which accepts a target sentence
  ////
  void CreateTgtFst(const vector<int>& intTokens, VectorFst<LogArc>& targetFst) {
    // targetFst is an empty mutable FST which uses the log-semiring, defined as:
    // logplus(x,y) := log(e^-x + e^-y), logtimes(x,y) := x + y, logzero = inf, logone = 0 }
    
    // build the FST states and arcs
    for(int stateId = 0; stateId < intTokens.size()+1; stateId++) {
      tgtFst.AddState();
      if(stateId == 0) continue;
      tgtFst.AddArc(stateId-1, LogArc(intTokens[stateId-1], intTokens[stateId-1], 0, stateId));
    }
    
    // set start and final states
    tgtFst.SetStart(0);
    tgtFst.SetFinal(intTokens.size(), 0); 
  }

  ////
  // constructs a single-state FSA which accepts any sequence of tokens in a source sentence
  ////
  void IbmModel1::CreateSrcFsts(const vector<int>& srcTokens, VectorFst<LogArc>& srcFst) {
    
    // allow null alignments
    srcTokens.push_back(NULL_SRC_TOKEN_ID);
      
    // one state
    int stateId = srcFst.AddState();

    // one arc per source token
    for(vector<int>::const_iterator tokenIter = srcTokens.begin(); 
	tokenIter != srcTokens.end(); 
	tokenIter++) {
      srcFst.AddArc(stateId, LogArc(*tokenIter, *tokenIter, 0, stateId));
    }

    // set start and final states
    srcFst.SetStart(stateId);
    srcFst.SetFinal(stateId, 0);

    // enable composition to the left
    ArcSort(&srcFst, ILabelCompare<LogArc>());
  }

  ////
  // constructs a grammar FST representing current model parameters for all sentence pairs
  ////
  void IbmModel1::CreateGrammarFst() {
    // clear grammar
    if (grammarFst.NumStates() > 0) {
      grammarFst.DeleteArcs(grammarFst.Start());
      grammarFst.DeleteStates();
    }
    
    // create the only state in this fst, and make it initial and final
    int stateId = grammarFst.AddState();
    grammarFst.SetStart(stateId);
    grammarFst.SetFinal(stateId, stateId);

    // create an arc for each parameter t|s
    for(Model1Param::const_iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
      for(map<int,float>::const_iterator tgtIter = (*srcIter).second.begin(); 
	  tgtIter != (*srcIter).second.end(); 
	  tgtIter++) {
	int tgtToken = (*tgtIter).first;
	int srcToken = (*srcIter).first;
	float paramValue = (*tgtIter).second;
	grammarFst.AddArc(stateId, LogArc(tgtToken, srcToken, paramValue, stateId));
      }
    }

    // enable composition to the left
    ArcSort(&grammarFst, ILabelCompare<LogArc>());
  }

  
  ////
  // argmax_{model parameters} likelihood(target sent GIVEN source sent & |target sent|)
  ////
  void LearnParameters() {

    // initialize model parameters
    InitParams();

    // create a "grammar FST" based on current model parameters
    CreateGrammarFst();
    
    // learning iterations
    do {

      // reset iteration's fractional counts
      ClearParams();
      
      // reset iteration's likelihood
      float logLikelihood = 0;
      
      // for each sentence pair
      ifstream tgtCorpus(tgtCorpusFilename.c_str(), ios::in), srcCorpus(srcCorpusFilename.c_str(), ios::in);
      string tgtLine, srcLine;
      while(getline(tgtCorpus, tgtLine) && getline(srcCorpus, srcLine)) {

	// create tgt fst
	VectorFst<LogArc> tgtFst;
	CreateTgtFst(StringUtils::ReadIntTokens(tgtLine), tgtFst);

	// create src fsts
	VectorFst<LogArc> srcFst;
	CreateSrcFst(StringUtils::ReadIntTokens(srcLine), srcFst);
	
	// build the alignment fst
	VectorFst<LogArc> tempFst;
	Compose(tgtFst, grammarFst, &tempFst);
	ArcSort(&temp, ILabelCompare<LogArc>());
	Compose(temp, srcFst, &alignmentFst);
	
	// compute alignments count
	double alignmentsCount = 1;
	for (int stateId = 0; stateId < alignmentFst.NumStates() ;stateId++) {
	  if (alignmentFst.Final(stateId) != LogWeight::Zero()) continue;
	  int sources = 0;
	  for (ArcIterator<VectorFst< LogArc > > arcIter(alignmentFst, stateId);
	       !arcIter.Done();
	       arcIter.Next()) {
	    sources++;
	  }
	  alignmentsCount *= sources;
	  assert(alignmentsCount > 1);
	}
	
	// compute potentials
	vector<LogWeight> alphas, betas;
	ShortestDistance(alignmentFst, &alphas, false);
	ShortestDistance(alignmentFst, &betas, true);
	float fSentLogLikelihood = betas[alignmentFst.Start()].Value();

	// compute fractional counts
	// for each alignment state
	for (int stateId = 0; stateId < alignmentFst.NumStates() ;stateId++) {
	  // for each alignment arc 
	  for (ArcIterator<VectorFst< LogArc > > arcIter(alignmentFst, stateId);
	       !arcIter.Done();
	       arcIter.Next()) {
	    int srcToken = arcIter.Value().olabel, tgtToken = arcIter.Value().ilabel;
	    int fromState = stateId, toState = arcIter.Value().nextstate;
	    
	    // probability of using this parameter given this sentence pair and the previous model
	    LogWeight currentParamLogProb = arcIter.Value().weight;
	    float fPosteriorLogProb = 
	      Times(Times(alphas[fromState], currentParamLogProb), betas[toState]).Value() - fSentLogLikelihood;
	    
	    // accumulate the fractional count for this parameter
	    params[srcToken][tgtToken] = 
	      Plus(LogWeight(params[srcToken][tgtToken]), LogWeight(fNormalizedPosteriorLogProb)).Value();
	  }
	}   
	
	// update the iteration log likelihood with this sentence's likelihod
	logLikelihood += fSentLogLikelihood;
      }
      
      // normalize fractional counts such that \sum_t p(t|s) = 1 \forall s
      NormalizeParams();
      
      // create a new grammar for the next iteration
      CreateGrammarFst();
      
      // update learningInfo
      learningInfo.logLikelihood.push_back(logLikelihood);
      learningInfo.iterationsCount++;
      
      // check for convergence
    } while(!IsModelConverged());
  }

};
