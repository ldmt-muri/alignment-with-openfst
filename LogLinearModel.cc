#include "LogLinearModel.h"

using namespace std;
using namespace fst;
using namespace OptUtils;

// finds out what are the parameters needed by reading hte corpus, and assigning initial weights based on the number of co-occurences
void LogLinearModel::InitParams() {

  // model trivially initialized to all zeros
  params.Clear();
}

// initialize model weights to zeros
LogLinearModel::LogLinearModel(const string& srcIntCorpusFilename, 
			       const string& tgtIntCorpusFilename, 
			       const string& outputFilenamePrefix, 
			       const Regularizer::Regularizer& regularizationType, 
			       const float regularizationConst, 
			       const LearningInfo& learningInfo) {
  // set member variables
  this->srcCorpusFilename = srcIntCorpusFilename;
  this->tgtCorpusFilename = tgtIntCorpusFilename;
  this->outputPrefix = outputFilenamePrefix;
  this->regularizationType = regularizationType;
  this->regularizationConst = regularizationConst;
  this->learningInfo = learningInfo;
  
  // initialize the model parameters
  InitParams();
  stringstream initialModelFilename;
  initialModelFilename << outputPrefix << ".param.init";
  params.PersistParams(initialModelFilename.str());
  
  // populate srcTgtFreq
  ifstream srcCorpus(srcCorpusFilename.c_str(), ios::in); 
  ifstream tgtCorpus(tgtCorpusFilename.c_str(), ios::in); 
  // for each line
  string srcLine, tgtLine;
  while(getline(srcCorpus, srcLine)) {
    getline(tgtCorpus, tgtLine);
    // read the list of integers representing target tokens
    vector< int > srcTokens, tgtTokens;
    StringUtils::ReadIntTokens(srcLine, srcTokens);
    StringUtils::ReadIntTokens(tgtLine, tgtTokens);
    // fill srcTgtFreq with frequency of co-occurence of each srcToken-tgtToken pair
    for(vector<int>::const_iterator srcTokenIter = srcTokens.begin(); srcTokenIter != srcTokens.end(); srcTokenIter++) {
      for(vector<int>::const_iterator tgtTokenIter = tgtTokens.begin(); tgtTokenIter != tgtTokens.end(); tgtTokenIter++) {
	map<int,int>& tgtFreq = srcTgtFreq[*srcTokenIter];
	if(tgtFreq.count(*tgtTokenIter) == 0) { tgtFreq[*tgtTokenIter] = 0; }
	tgtFreq[*tgtTokenIter]++;
      }
    }
  }
  
  // for debugging
  //  cerr << "================srcTgtFreq===================" << endl;
  //  for(map<int, map<int,int> >::const_iterator srcTokenIter = srcTgtFreq.begin();
  //      srcTokenIter != srcTgtFreq.end();
  //      srcTokenIter++) {
  //    for(map<int,int>::const_iterator tgtTokenIter = srcTokenIter->second.begin();
  //	tgtTokenIter != srcTokenIter->second.end();
  //	tgtTokenIter++) {
  //      cerr << "count(" << srcTokenIter->first << "," << tgtTokenIter->first << ")=" << tgtTokenIter->second << endl;
  //    }
  //  }
  //  string dummy;
  //  cin >> dummy;
}

// create a transducer that represents possible translations of the source sentence of a given length
void LogLinearModel::CreateAllTgtFst(const vector<int>& srcTokens, 
				     int tgtSentLen, 
				     typename DiscriminativeLexicon::DiscriminativeLexicon lexicon, 
				     VectorFst<LogTripleArc>& allTgtFst,
				     set<int>& uniqueTgtTokens) {
  // determine the set of possible target tokens allowed to be in a translation
  uniqueTgtTokens.clear();
  switch(lexicon)
    {
    case DiscriminativeLexicon::ALL:
      for(vector<int>::const_iterator srcTokenIter = srcTokens.begin(); srcTokenIter != srcTokens.end(); srcTokenIter++) {
	map<int,int>& tgtFreq = srcTgtFreq[*srcTokenIter];
	for(map<int,int>::const_iterator tgtTokenIter = tgtFreq.begin(); tgtTokenIter != tgtFreq.end(); tgtTokenIter++) {
	  uniqueTgtTokens.insert(tgtTokenIter->first);
	}
      }
      break;
    case DiscriminativeLexicon::COOCC:
      // TODO
      assert(false);
      break;
    }

  // create the fst
  int statesCount = tgtSentLen + 1;
  for(int stateId = 0; stateId < statesCount; stateId++) {
    int temp = allTgtFst.AddState();
    assert(temp == stateId);
    if(stateId == 0) continue;
    for (set<int>::const_iterator uniqueTgtTokenIter = uniqueTgtTokens.begin(); 
	 uniqueTgtTokenIter != uniqueTgtTokens.end(); 
	 uniqueTgtTokenIter++) {
      allTgtFst.AddArc(stateId-1, LogTripleArc(*uniqueTgtTokenIter, *uniqueTgtTokenIter, FstUtils::EncodeTriple(stateId, 0, 0), stateId));
    }
  }
  allTgtFst.SetStart(0);
  allTgtFst.SetFinal(statesCount-1, LogTripleWeight::One());
  ArcSort(&allTgtFst, ILabelCompare<LogTripleArc>());
  
  // for debugging
  //  cerr << "=====================" << endl;
  //  cerr << "allTgtFst is as follows:" << endl;
  //  cerr << "=====================" << endl;
  //  FstUtils::PrintFstSummary(allTgtFst);
  //  string dummy;
  //  cin >> dummy;
}

// create the tgt sent transducer: a linear chain of target words in order, with ProductWeight<LogWeight,LogWeight>
// the first and second values in the weight semiring represent the tgt and src token position, respectively. 
// note: tgtFst is assumed to be empty
void LogLinearModel::CreateTgtFst(const vector<int>& tgtTokens, VectorFst<LogTripleArc>& tgtFst, set<int>& uniqueTgtTokens) {
  // create the fst
  int statesCount = tgtTokens.size() + 1;
  for(int stateId = 0; stateId < tgtTokens.size()+1; stateId++) {
    int temp = tgtFst.AddState();
    assert(temp == stateId);
    if(stateId == 0) continue;
    tgtFst.AddArc(stateId-1, LogTripleArc(tgtTokens[stateId-1], tgtTokens[stateId-1], FstUtils::EncodeTriple(stateId, 0, 0), stateId));
    uniqueTgtTokens.insert(tgtTokens[stateId-1]);
  }
  tgtFst.SetStart(0);
  tgtFst.SetFinal(tgtTokens.size(), LogTripleWeight::One());
  ArcSort(&tgtFst, ILabelCompare<LogTripleArc>());
  
  // for debugging
  //  cerr << "=====================" << endl;
  //  cerr << "tgtFst is as follows:" << endl;
  //  cerr << "=====================" << endl;
  //  FstUtils::PrintFstSummary(tgtFst);
}

// this grammar fst is designed such that when composed with tgtFst and srcFst, it produces the alignmentFst.
// in addition to the initial state, there's one final state per sourceTokenId in this sentence pair.
// all states (including initial) emit arcs to all other states (excluding initial). 
// arcs from state x to state y have iLabel=<TGT-TOKEN-ID> and oLabel=y, where <TGT-TOKEN-ID> is a wildcard for all target words in this sentence.
// all arcs have the weight: LogTripleWeight::One(). 
// note: grammarFst is assumed to be empty
void LogLinearModel::CreatePerSentGrammarFst(const vector<int>& srcTokens, const set<int>& uniqueTgtTokens, VectorFst<LogTripleArc>& grammarFst) {
  // first, create the initial state
  int initialState = grammarFst.AddState();
  assert(initialState == 0);
  grammarFst.SetStart(initialState);

  // then create a final state for each unique src token id
  map<int,int> srcTokenIdToStateId;
  for(vector<int>::const_iterator srcTokenIter = srcTokens.begin(); srcTokenIter != srcTokens.end(); srcTokenIter++) {
    if(srcTokenIdToStateId.count(*srcTokenIter) > 0) { continue; }
    srcTokenIdToStateId[*srcTokenIter] = grammarFst.AddState();
    grammarFst.SetFinal(srcTokenIdToStateId[*srcTokenIter], LogTripleWeight::One());
  }

  // then, from every state ...
  for(StateIterator< VectorFst<LogTripleArc> > siter(grammarFst); !siter.Done(); siter.Next()) {
    int fromState = siter.Value();
    
    // to each of the srcTokenId-bound states ...
    for(map<int,int>::const_iterator boundStateIter = srcTokenIdToStateId.begin(); 
	boundStateIter != srcTokenIdToStateId.end();
	boundStateIter++) {
      int toState = boundStateIter->second;
      int srcTokenId = boundStateIter->first;

      // add arcs for each unique tgtTokenId
      for(set<int>::const_iterator tgtTokenIter = uniqueTgtTokens.begin();
	  tgtTokenIter != uniqueTgtTokens.end();
	  tgtTokenIter++) {
	int tgtTokenId = *tgtTokenIter;

	// add the damn arc!
	grammarFst.AddArc(fromState, LogTripleArc(tgtTokenId, srcTokenId, LogTripleWeight::One(), toState));
      }
    }
  }

  // sort input labels to enable composition
  ArcSort(&grammarFst, ILabelCompare<LogTripleArc>());

  // for debugging
  //  cerr << "==================== " << endl;
  //  cerr << "PER-SENT GRAMMAR FST " << endl;
  //  cerr << "==================== " << endl;
  //  FstUtils::PrintFstSummary(grammarFst);  
}

// this is a single-state acceptor which accepts any sequence of srcTokenIds in this sentence pair
// the weight on the arcs is LogTripleWeight(0,SRC-TOKEN-POS). In case a srcTokenId is repeated more
// in this source sentence, we create multiple arcs for it in order to adequately represent the 
// corresponding position in src sentence. 
// note: the first token in srcTokens must be the NULL source token ID
// note: srcFst is assumed to be empty
void LogLinearModel::CreateSrcFst(const vector<int>& srcTokens, VectorFst<LogTripleArc>& srcFst) {
  // note: the first token in srcTokens must be the NULL source token ID
  assert(srcTokens[0] == NULL_SRC_TOKEN_ID);

  // create the initial/final and only state
  int stateId = srcFst.AddState();
  srcFst.SetStart(stateId);
  srcFst.SetFinal(stateId, LogTripleWeight::One());
  // note: srcFst is assumed to be empty
  assert(stateId == 0);

  // now add the arcs
  for(int srcTokenPos = 0; srcTokenPos < srcTokens.size(); srcTokenPos++) {
    int srcToken = srcTokens[srcTokenPos];
    srcFst.AddArc(stateId, LogTripleArc(srcToken, srcToken, FstUtils::EncodeTriple(0, srcTokenPos, 0), stateId));
  }
  ArcSort(&srcFst, ILabelCompare<LogTripleArc>());
    
  // for debugging
  //  cerr << "=============SRC FST==========" << endl;
  //  FstUtils::PrintFstSummary(srcFst);
}

// alignment fst is a transducer on which each complete path represents a unique alignment for a sentence pair.
// when tgtLineIsGiven is false, this function builds an alignment FST for all tgt sentences of length |tgtLine| which
// may be a translation of the source sentence. Effectively, this FST represents p(alignment, tgtSent | srcSent, L_t).
// lexicon is only used when tgtLineIsGiven=false. Depending on its value, the constructed FST may represent a subset of 
// possible translations (cuz it's usually too expensive to represnet all translations). 
void LogLinearModel::BuildAlignmentFst(const vector<int>& srcTokens, const vector<int>& tgtTokens, VectorFst<LogTripleArc>& alignmentFst, 
				       bool tgtLineIsGiven, typename DiscriminativeLexicon::DiscriminativeLexicon lexicon, 
				       int sentId) {

  stringstream alignmentFstFilename;
  alignmentFstFilename << outputPrefix << ".align";
  if(tgtLineIsGiven) {
    alignmentFstFilename << "GivenTS." << sentId;
  } else {
    alignmentFstFilename << "GivenS." << sentId;
  }

  // in the first pass over the corpus, construct the FSTs
  if(!learningInfo.saveAlignmentFstsOnDisk || 
     learningInfo.saveAlignmentFstsOnDisk && learningInfo.iterationsCount == 0) {
    // tgt transducer
    VectorFst<LogTripleArc> tgtFst;
    // unique target tokens used in tgtFst. this is populated by CreateTgtFst or CreateAllTgtFst and later used by CreatePerSentGrammarFst
    set<int> uniqueTgtTokens;
    // in this model, two kinds of alignment FSTs are needed: one assumes a particular target sentence, 
    // while the other represents many more translations.
    if(tgtLineIsGiven) {
      CreateTgtFst(tgtTokens, tgtFst, uniqueTgtTokens);
    } else {
      CreateAllTgtFst(srcTokens, tgtTokens.size(), lexicon, tgtFst, uniqueTgtTokens);
    }
    
    // per-sentence grammar
    VectorFst<LogTripleArc> grammarFst;
    CreatePerSentGrammarFst(srcTokens, uniqueTgtTokens, grammarFst);
    
    // src transducer
    VectorFst<LogTripleArc> srcFst;
    CreateSrcFst(srcTokens, srcFst);
    
    // compose the three transducers to get the alignmentFst with weights representing tgt/src positions
    VectorFst<LogTripleArc> temp;
    Compose(tgtFst, grammarFst, &temp);
    
    // for debugging
    //  if(tgtLineIsGiven) {
    //    cerr << "=======================tgtFst + perSentGrammar=====================" << endl;
    //    FstUtils::PrintFstSummary(temp);
    //    string dummy;
    //    cin >> dummy;
    //  }
    
    Compose(temp, srcFst, &alignmentFst);
    
    // save the resulting alignmentFst for future use
    if(learningInfo.saveAlignmentFstsOnDisk) {
      alignmentFst.Write(alignmentFstFilename.str());
    }
    
  } else if (learningInfo.saveAlignmentFstsOnDisk) {
    // in subsequent passes, read the previously stored FST
        
    // TODO: debug this block. something is wrong with the read operation. The huge FSTs, when read, don't have any states.
    cerr << "hi" << endl;
    alignmentFst.Read(alignmentFstFilename.str());
    bool verified = Verify(alignmentFst);
    if (verified) {
      cerr << "the model loaded was verified." << endl;
    } else {
      cerr << "the model loaded was not verified." << endl;
    }

    if(!tgtLineIsGiven && sentId == 1) {
      FstUtils::PrintFstSummary(alignmentFst);
    }
    string dummy;
    cin >> dummy;

  }
  
  // compute the probability of each transition on the alignment FST according to the current model parameters
  // set the third value in the LogTripleWeights on the arcs = the computed prob for that arc
  for(StateIterator< VectorFst<LogTripleArc> > siter(alignmentFst); !siter.Done(); siter.Next()) {
    LogTripleArc::StateId stateId = siter.Value();
    for(MutableArcIterator< VectorFst<LogTripleArc> > aiter(&alignmentFst, stateId); !aiter.Done(); aiter.Next()) {
      LogTripleArc arc = aiter.Value();
      int tgtTokenId = arc.ilabel;
      int srcTokenId = arc.olabel;
      float tgtTokenPos, srcTokenPos, dummy;
      FstUtils::DecodeTriple(arc.weight, tgtTokenPos, srcTokenPos, dummy);
      float arcProb = params.ComputeLogProb(srcTokenId, tgtTokenId, srcTokenPos, tgtTokenPos, srcTokens.size(), tgtTokens.size());
      arc.weight = FstUtils::EncodeTriple(tgtTokenPos, srcTokenPos, arcProb);
      aiter.SetValue(arc);
    } 
  }

  // for debugging
  //  cerr << "end of BuildAlignmentFst()" << endl;
  //  if(tgtLineIsGiven) {
  //cerr << "=======================tgtFst + perSentGrammar + srcFst=====================" << endl;
  //FstUtils::PrintFstSummary(alignmentFst);
  //string dummy;
  //cin >> dummy;
  //}

}

void LogLinearModel::AddSentenceContributionToGradient(const VectorFst< LogTripleArc >& descriptorFst, 
						       const VectorFst< LogArc >& totalProbFst, 
						       LogLinearParams& gradient,
						       float logPartitionFunction,
						       int srcTokensCount,
						       int tgtTokensCount,
						       bool subtract) {
  clock_t d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0, temp;
  temp = clock();
  for (int stateId = 0; stateId < descriptorFst.NumStates() ;stateId++) {
    d1 += clock() - temp;
    temp = clock();
    ArcIterator< VectorFst< LogArc > > totalProbArcIter(totalProbFst, stateId);
    for (ArcIterator< VectorFst< LogTripleArc > > descriptorArcIter(descriptorFst, stateId);
	 !descriptorArcIter.Done() && !totalProbArcIter.Done();
	 descriptorArcIter.Next(), totalProbArcIter.Next()) {
      d2 += clock() - temp;
      temp = clock();
      // parse the descriptorArc and totalProbArc
      int tgtToken = descriptorArcIter.Value().ilabel;
      int srcToken = descriptorArcIter.Value().olabel;
      float tgtPos, srcPos, dummy;
      FstUtils::DecodeTriple(descriptorArcIter.Value().weight, tgtPos, srcPos, dummy);
      LogWeight totalProb = totalProbArcIter.Value().weight;
      d3 += clock() - temp;
      temp = clock();
      // find the features activated on this transition, and their values
      map<string, float> activeFeatures;
      gradient.FireFeatures(srcToken, tgtToken, (int)srcPos, (int)tgtPos, srcTokensCount, tgtTokensCount, activeFeatures);
      d4 += clock() - temp;
      temp = clock();
      // for debugging
      //      cerr << endl << "=================features fired===================" << endl;
      //      cerr << "arc: tgtToken=" << tgtToken << " srcToken=" << srcToken << " tgtPos=" << tgtPos << " srcPos=" << srcPos;
      //      cerr << " srcTokensCount=" << srcTokensCount << " tgtTokensCount=" << tgtTokensCount << " totalProb=" << totalProb << endl;
      //      cerr << "Features fired are:" << endl;

      // now, for each feature fired on each arc of aGivenTS
      for(map<string,float>::const_iterator feature = activeFeatures.begin(); feature != activeFeatures.end(); feature++) {
	if(gradient.params.count(feature->first) == 0) {
	  gradient.params[feature->first] = 0;
	}

	// for debugging 
	//	cerr << "gradient[" << feature->first << "=" << feature->second << "] was " << gradient.params[feature->first] << " became ";
	
	// the positive contribution to the derivative of this feature by this arc is
	float contribution = feature->second * FstUtils::nExp(Times(totalProb, -1.0 * logPartitionFunction).Value());
	if(subtract) {
	  gradient.params[feature->first] -= contribution;
	} else {
	  gradient.params[feature->first] += contribution;
	}

	// for debugging
	//	cerr << gradient.params[feature->first] << endl;
      }
      d5 += clock() - temp;
      temp = clock();
    }
    temp = clock();
  }
  //  string dummy2;
  //  cin >> dummy2;
  cerr << "d1=" << d1 << " d2=" << d2 << " d3=" << d3 << " d4=" << d4 << " d5=" << d5 << endl;
}

// for each feature in the model, add the corresponding regularization term to the gradient
void LogLinearModel::AddRegularizerTerm(LogLinearParams& gradient) {

  // compute ||params||_2
  float l2 = 0;
  if(regularizationType == Regularizer::L2) {
    for(map<string, float>::const_iterator featureIter = params.params.begin();
	featureIter != params.params.end();
	featureIter++) {
      l2 += featureIter->second * featureIter->second;
    }
  }

  // for each feature
  for(map<string, float>::const_iterator featureIter = params.params.begin(); 
      featureIter != params.params.end();
      featureIter++) {
    float term;
    switch(regularizationType) {
    case Regularizer::L2:
      term = 2.0 * regularizationConst * featureIter->second / l2;
      gradient.params[featureIter->first] += term;
      assert(gradient.params[featureIter->first] == term);
      break;
    case Regularizer::NONE:
      break;
    default:
      assert(false);
      break;
    }
  }
}

void LogLinearModel::Train() {

  clock_t accUpdateClocks = 0, accShortestDistanceClocks = 0, accBuildingFstClocks = 0, accReadClocks = 0, accRuntimeClocks = 0, accRegularizationClocks = 0, accWriteClocks = 0;

  // passes over the training data
  do {
    
    clock_t timestamp1 = clock();

    float logLikelihood = 0;

    ifstream srcCorpus(srcCorpusFilename.c_str(), ios::in); 
    ifstream tgtCorpus(tgtCorpusFilename.c_str(), ios::in); 

    // IMPORTANT NOTE: this is the gradient of the regualarized log-likelihood, in the real-domain, not in the log-domain.
    // in other words, when the equations on paper say we should gradient[feature] += x, we usually have x in the log-domain -log(x),
    // so effectively we need to add (rather than log-add) e^{- -log(x) } which is equivalent to += x
    // TODO OPTIMIZATION: instead of defining gradient here, define it as a class member.
    LogLinearParams gradient;

    // first, compute the regularizer's term for each feature in the gradient
    clock_t timestamp2 = clock();
    AddRegularizerTerm(gradient);
    accRegularizationClocks += clock() - timestamp2;
    
    // for each line
    string srcLine, tgtLine;
    int sentsCounter = 1;
    clock_t timestamp3 = clock();
    while(getline(srcCorpus, srcLine)) {
      getline(tgtCorpus, tgtLine);
      accReadClocks += clock() - timestamp3;

      // for debugging
      //      cerr << "srcLine: " << srcLine << endl;
      //      cerr << "tgtLine: " << tgtLine << endl << endl;

      // read the list of integers representing target tokens
      vector< int > srcTokens, tgtTokens;
      StringUtils::ReadIntTokens(srcLine, srcTokens);
      StringUtils::ReadIntTokens(tgtLine, tgtTokens);
      // add the null src token as a possible alignment for any target token
      srcTokens.insert(srcTokens.begin(), 1, NULL_SRC_TOKEN_ID);
      
      // build FST(a|t,s) and build FST(a,t|s)
      clock_t timestamp4 = clock();
      VectorFst< LogTripleArc > aGivenTS, aTGivenS;
      BuildAlignmentFst(srcTokens, tgtTokens, aGivenTS, true, learningInfo.neighborhood, sentsCounter);
      BuildAlignmentFst(srcTokens, tgtTokens, aTGivenS, false, learningInfo.neighborhood, sentsCounter);

      // change the LogTripleWeight semiring to LogWeight using LogTripleToLogMapper
      VectorFst< LogArc > aGivenTSProbs, aTGivenSProbs;
      ArcMap(aGivenTS, &aGivenTSProbs, LogTripleToLogMapper());
      ArcMap(aTGivenS, &aTGivenSProbs, LogTripleToLogMapper());
      accBuildingFstClocks += clock() - timestamp4;
      
      // for debugging
      //      cerr << "========================aGivenTSProbs=====================" << endl;
      //      FstUtils::PrintFstSummary(aGivenTSProbs);
      //      string dummy;
      //      cin >> dummy;

      // get the total (i.e. end-to-end) probability of traversing each arc
      clock_t timestamp5 = clock();
      VectorFst< LogArc > aGivenTSTotalProb, aTGivenSTotalProb;
      LogWeight aGivenTSBeta0, aTGivenSBeta0;
      FstUtils::ComputeTotalProb<LogWeight,LogArc>(aGivenTSProbs, aGivenTSTotalProb, aGivenTSBeta0);
      FstUtils::ComputeTotalProb<LogWeight,LogArc>(aTGivenSProbs, aTGivenSTotalProb, aTGivenSBeta0);
      accShortestDistanceClocks += clock() - timestamp5;

      // add this sentence's contribution to the gradient of model parameters.
      // lickily, the contribution factorizes into: the end-to-end arc probabilities and beta[0] of aGivenTS and aTGivenS.
      clock_t timestamp6 = clock();
      AddSentenceContributionToGradient(aGivenTS, aGivenTSTotalProb, gradient, aGivenTSBeta0.Value(), srcTokens.size(), tgtTokens.size(), false);
      AddSentenceContributionToGradient(aTGivenS, aTGivenSTotalProb, gradient, aTGivenSBeta0.Value(), srcTokens.size(), tgtTokens.size(), true);

      // update the iteration log likelihood with this sentence's likelihod
      assert(aTGivenSBeta0.Value() != 0);
      logLikelihood += aGivenTSBeta0.Value() - aTGivenSBeta0.Value();
      
      // if the optimization algorithm is stochastic, update the parameters here.
      cerr << "s";
      if(IsStochastic(learningInfo.optimizationMethod.algorithm) && sentsCounter % learningInfo.optimizationMethod.miniBatchSize == 0) {
	cerr << "u";
	params.UpdateParams(gradient, learningInfo.optimizationMethod);
	gradient.Clear();
      }
      accUpdateClocks += clock() - timestamp6;

      // for debugging only
      // report accumulated times
      cerr << "accumulated runtime = " << (float) accRuntimeClocks / CLOCKS_PER_SEC << " sec." << endl;
      cerr << "accumulated disk write time = " << (float) accWriteClocks / CLOCKS_PER_SEC << " sec." << endl;
      cerr << "accumulated disk read time = " << (float) accReadClocks / CLOCKS_PER_SEC << " sec." << endl;
      cerr << "accumulated fst construction time = " << (float) accBuildingFstClocks / CLOCKS_PER_SEC << " sec." << endl;
      cerr << "accumulated fst shortest-distance time = " << (float) accShortestDistanceClocks / CLOCKS_PER_SEC << " sec." << endl;
      cerr << "accumulated param update time = " << (float) accUpdateClocks / CLOCKS_PER_SEC << " sec." << endl;
      cerr << "accumulated regularization time = " << (float) accRegularizationClocks / CLOCKS_PER_SEC << " sec." << endl;
      
      // logging
      if (++sentsCounter % 50 == 0) {
	cerr << endl << sentsCounter << " sents processed.." << endl;
      }

      timestamp3 = clock();
    }
    
    // if the optimization algorithm isn't stochastic, update the parameters here.
    clock_t timestamp7 = clock();
    if(IsStochastic(learningInfo.optimizationMethod.algorithm)) {
      params.UpdateParams(gradient, learningInfo.optimizationMethod);
    }
    accUpdateClocks += clock() - timestamp7;
    
    // persist parameters
    stringstream filename;
    clock_t timestamp8 = clock();
    filename << outputPrefix << ".param." << learningInfo.iterationsCount;
    params.PersistParams(filename.str());
    accWriteClocks += clock() - timestamp8;
    
    // logging
    cerr << endl << "WOOHOOOO" << endl;
    cerr << "completed iteration # " << learningInfo.iterationsCount << " - total loglikelihood = " << logLikelihood << endl;
    cerr << "total iteration time = " << (float) (clock()-timestamp1) / CLOCKS_PER_SEC << " sec." << endl << endl;
    
    // book keeping
    accRuntimeClocks += clock() - timestamp1;
    
    // update learningInfo
    learningInfo.logLikelihood.push_back(logLikelihood);
    learningInfo.iterationsCount++;

    // close the files
    srcCorpus.close();
    tgtCorpus.close();

    // check for convergence
  } while(!learningInfo.IsModelConverged());

  // persist parameters
  params.PersistParams(outputPrefix + ".param.final");

  // report accumulated times
  cerr << "accumulated runtime = " << (float) accRuntimeClocks / CLOCKS_PER_SEC << " sec." << endl;
  cerr << "accumulated disk write time = " << (float) accWriteClocks / CLOCKS_PER_SEC << " sec." << endl;
  cerr << "accumulated disk read time = " << (float) accReadClocks / CLOCKS_PER_SEC << " sec." << endl;
  cerr << "accumulated fst construction time = " << (float) accBuildingFstClocks / CLOCKS_PER_SEC << " sec." << endl;
  cerr << "accumulated fst shortest-distance time = " << (float) accShortestDistanceClocks / CLOCKS_PER_SEC << " sec." << endl;
  cerr << "accumulated param update time = " << (float) accUpdateClocks / CLOCKS_PER_SEC << " sec." << endl;
  cerr << "accumulated regularization time = " << (float) accRegularizationClocks / CLOCKS_PER_SEC << " sec." << endl;
}
  
// TODO: not implemented
// given the current model, align the corpus
void LogLinearModel::Align() {
  
}
