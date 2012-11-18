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
			       const LearningInfo& learningInfo) : params(*learningInfo.srcVocabDecoder,
  									  *learningInfo.tgtVocabDecoder,
  									  *learningInfo.ibm1ForwardLogProbs,
  									  *learningInfo.ibm1BackwardLogProbs) {
  // set member variables
  this->srcCorpusFilename = srcIntCorpusFilename;
  this->tgtCorpusFilename = tgtIntCorpusFilename;
  this->outputPrefix = outputFilenamePrefix;
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
  map<int,int> tgtTokenIdFreq;
  int sentsCounter = 0;
  while(getline(srcCorpus, srcLine)) {
    getline(tgtCorpus, tgtLine);
    sentsCounter++;
    // read the list of integers representing target tokens
    vector< int > srcTokens, tgtTokens;
    StringUtils::ReadIntTokens(srcLine, srcTokens);
    StringUtils::ReadIntTokens(tgtLine, tgtTokens);
    // update tgtTokenIdFreq (i.e. how many times each target token id has been repeated in the corpus)
    for(vector<int>::const_iterator tgtTokenIter = tgtTokens.begin(); tgtTokenIter != tgtTokens.end(); tgtTokenIter++) {
      if(tgtTokenIdFreq.count(*tgtTokenIter) == 0) { tgtTokenIdFreq[*tgtTokenIter] = 0; }
      tgtTokenIdFreq[*tgtTokenIter]++;
    }
    // fill srcTgtFreq with frequency of co-occurence of each srcToken-tgtToken pair
    for(vector<int>::const_iterator srcTokenIter = srcTokens.begin(); srcTokenIter != srcTokens.end(); srcTokenIter++) {
      map<int,int>& tgtFreq = srcTgtFreq[*srcTokenIter];
      for(vector<int>::const_iterator tgtTokenIter = tgtTokens.begin(); tgtTokenIter != tgtTokens.end(); tgtTokenIter++) {
	if(tgtFreq.count(*tgtTokenIter) == 0) { tgtFreq[*tgtTokenIter] = 0; }
	tgtFreq[*tgtTokenIter]++;
      }
    }
  }
  // initialize corpusSize
  corpusSize = sentsCounter;

  // add the null alignments to the grammar
  map<int,int>& tgtFreq = srcTgtFreq[NULL_SRC_TOKEN_ID];
  for(map<int,int>::const_iterator tgtTokenIter = tgtTokenIdFreq.begin(); tgtTokenIter != tgtTokenIdFreq.end(); tgtTokenIter++) {
    tgtFreq[tgtTokenIter->first] = tgtTokenIter->second;
  }
  srcCorpus.close();
  tgtCorpus.close();

  // grammar is unweighted, so there's no need to change from one iteration to the next (unless we want to prune it)
  CreateGrammarFst();

  // some configurations are not unsupported:
  if(learningInfo.optimizationMethod.algorithm == OptUtils::STOCHASTIC_GRADIENT_DESCENT &&
     learningInfo.optimizationMethod.regularizer == Regularizer::L2) {
    // - SGD with L2
    assert(false);
  } else if(learningInfo.optimizationMethod.algorithm == OptUtils::GRADIENT_DESCENT &&
	    learningInfo.optimizationMethod.regularizer == Regularizer::L1) {
    // - GD with L1
    assert(false);
  }

  // bool vectors indicating which feature types to use
  assert(enabledFeatureTypesSimple.size() == 0 && enabledFeatureTypesFirstOrder.size() == 0);
  for(int i = 0; i < 25; i++) {
    enabledFeatureTypesSimple.push_back(true);
    enabledFeatureTypesFirstOrder.push_back(true);
  }
  // disable nonlocal features for a simpler model
  enabledFeatureTypesSimple[4] = false; // F4
  enabledFeatureTypesSimple[5] = false; // F5
  enabledFeatureTypesSimple[6] = false; // F6  
  enabledFeatureTypesSimple[19] = false; // F19
  enabledFeatureTypesSimple[20] = false; // F20
  enabledFeatureTypesSimple[23] = false; // F23
  enabledFeatureTypesSimple[24] = false; // F24

}

// assumptions: 
// - srcTgtFreq has been populated
// - grammarFst is empty
void LogLinearModel::CreateGrammarFst() {
  assert(grammarFst.NumStates() == 0);
  assert(srcTgtFreq.size() != 0);

  // create the single state of this fst and make it initial and final
  int stateId = grammarFst.AddState();
  assert(stateId == 0);
  grammarFst.SetStart(stateId);
  grammarFst.SetFinal(stateId, LogQuadWeight::One());

  // for each src type
  for(map<int, map<int, int> >::const_iterator srcIter = srcTgtFreq.begin();
      srcIter != srcTgtFreq.end();
      srcIter++) {
    // for each tgt type that cooccurs with that src type
    for(map<int, int>::const_iterator tgtIter = srcIter->second.begin(); 
	tgtIter != srcIter->second.end();
	tgtIter++) {
      // for each cooccuring tgt-src pair in the corpus, add an arc
      grammarFst.AddArc(stateId, LogQuadArc(tgtIter->first, srcIter->first, LogQuadWeight::One(), stateId));
    }
  }
}

// create a transducer that represents possible translations of the source sentence of a given length
void LogLinearModel::CreateAllTgtFst(const vector<int>& srcTokens, 
				     int tgtSentLen, 
				     DiscriminativeLexicon::DiscriminativeLexicon lexicon, 
				     VectorFst<LogQuadArc>& allTgtFst,
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
      allTgtFst.AddArc(stateId-1, LogQuadArc(*uniqueTgtTokenIter, *uniqueTgtTokenIter, FstUtils::EncodeQuad(stateId, 0, 0, 0), stateId));
    }
  }
  allTgtFst.SetStart(0);
  allTgtFst.SetFinal(statesCount-1, LogQuadWeight::One());
  ArcSort(&allTgtFst, ILabelCompare<LogQuadArc>());
  
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
void LogLinearModel::CreateTgtFst(const vector<int>& tgtTokens, VectorFst<LogQuadArc>& tgtFst, set<int>& uniqueTgtTokens) {
  // create the fst
  int statesCount = tgtTokens.size() + 1;
  for(int stateId = 0; stateId < tgtTokens.size()+1; stateId++) {
    int temp = tgtFst.AddState();
    assert(temp == stateId);
    if(stateId == 0) continue;
    tgtFst.AddArc(stateId-1, LogQuadArc(tgtTokens[stateId-1], tgtTokens[stateId-1], FstUtils::EncodeQuad(stateId, 0, 0, 0), stateId));
    uniqueTgtTokens.insert(tgtTokens[stateId-1]);
  }
  tgtFst.SetStart(0);
  tgtFst.SetFinal(tgtTokens.size(), LogQuadWeight::One());
  ArcSort(&tgtFst, ILabelCompare<LogQuadArc>());
  
  // for debugging
  //  cerr << "=====================" << endl;
  //  cerr << "tgtFst is as follows:" << endl;
  //  cerr << "=====================" << endl;
  //  FstUtils::PrintFstSummary(tgtFst);
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
void LogLinearModel::Create1stOrderSrcFst(const vector<int>& srcTokens, VectorFst<LogQuadArc>& srcFst) {

  // enforce assumptions
  assert(srcTokens[0] == NULL_SRC_TOKEN_ID);
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
      srcFst.SetFinal(i, LogQuadWeight::One());
    }

    // each state can go to itself with the null src token
    srcFst.AddArc(i, LogQuadArc(srcTokens[0], srcTokens[0], FstUtils::EncodeQuad(0, i, i, 0), i));

    // each state can go to states representing non-null alignments
    for(int j = 1; j < srcTokens.size(); j++) {
      srcFst.AddArc(i, LogQuadArc(srcTokens[j], srcTokens[j], FstUtils::EncodeQuad(0, j, i, 0), j));
    }
  }
 
  // arc sort to enable composition
  ArcSort(&srcFst, ILabelCompare<LogQuadArc>());

  // for debugging
  //  cerr << "=============SRC FST==========" << endl;
  //  cerr << FstUtils::PrintFstSummary(srcFst);
}  

// this is a single-state acceptor which accepts any sequence of srcTokenIds in this sentence pair
// the weight on the arcs is LogQuadWeight(0,SRC-TOKEN-POS). In case a srcTokenId is repeated more
// in this source sentence, we create multiple arcs for it in order to adequately represent the 
// corresponding position in src sentence. 
// note: the first token in srcTokens must be the NULL source token ID
// note: srcFst is assumed to be empty
void LogLinearModel::CreateSimpleSrcFst(const vector<int>& srcTokens, VectorFst<LogQuadArc>& srcFst) {
  // note: the first token in srcTokens must be the NULL source token ID
  assert(srcTokens[0] == NULL_SRC_TOKEN_ID);

  // create the initial/final and only state
  int stateId = srcFst.AddState();
  srcFst.SetStart(stateId);
  srcFst.SetFinal(stateId, LogQuadWeight::One());
  // note: srcFst is assumed to be empty
  assert(stateId == 0);

  // now add the arcs
  for(int srcTokenPos = 0; srcTokenPos < srcTokens.size(); srcTokenPos++) {
    int srcToken = srcTokens[srcTokenPos];
    srcFst.AddArc(stateId, LogQuadArc(srcToken, srcToken, FstUtils::EncodeQuad(0, srcTokenPos, 0, 0), stateId));
  }
  ArcSort(&srcFst, ILabelCompare<LogQuadArc>());
}

// alignment fst is a transducer on which each complete path represents a unique alignment for a sentence pair.
// when tgtLineIsGiven is false, this function builds an alignment FST for all tgt sentences of length |tgtLine| which
// may be a translation of the source sentence. Effectively, this FST represents p(alignment, tgtSent | srcSent, L_t).
// lexicon is only used when tgtLineIsGiven=false. Depending on its value, the constructed FST may represent a subset of 
// possible translations (cuz it's usually too expensive to represnet all translations). 
void LogLinearModel::BuildAlignmentFst(const vector<int>& srcTokens, const vector<int>& tgtTokens, VectorFst<LogQuadArc>& alignmentFst, 
				       bool tgtLineIsGiven, DiscriminativeLexicon::DiscriminativeLexicon lexicon, 
				       int sentId, Distribution::Distribution distribution, VectorFst<LogQuadArc>& tgtFst) {

  assert(alignmentFst.NumStates() == 0);

  stringstream alignmentFstFilename;
  alignmentFstFilename << outputPrefix << ".align";
  if(tgtLineIsGiven) {
    alignmentFstFilename << "GivenTS." << sentId;
  } else {
    alignmentFstFilename << "GivenS." << sentId;
  }

  // in the first pass over the corpus, construct the FSTs
  if(distribution != Distribution::TRUE ||
     !learningInfo.saveAlignmentFstsOnDisk || 
     (learningInfo.saveAlignmentFstsOnDisk && learningInfo.iterationsCount == 0)) {

    // build the alignment FST according to the true distribution
    // note: we always use the true distribution when the translation is given because,
    //       with the current set of features, the alignment FST is not terribly huge O(tgtLength * srcLength^2).
    //       However, when the translation is not given (i.e. tgtLineIsGiven == false), 
    //       we use whichever distribution given as a parameter to this method.
    if(distribution == Distribution::TRUE || tgtLineIsGiven) {
      
      // tgt transducer
      assert(tgtFst.NumStates() == 0);
      // unique target tokens used in tgtFst. this is populated by CreateTgtFst or CreateAllTgtFst and later used by CreatePerSentGrammarFst
      set<int> uniqueTgtTokens;
      // in this model, two kinds of alignment FSTs are needed: one assumes a particular target sentence, 
      // while the other represents many more translations.
      if(tgtLineIsGiven) {
	CreateTgtFst(tgtTokens, tgtFst, uniqueTgtTokens);
      } else {
	CreateAllTgtFst(srcTokens, tgtTokens.size(), lexicon, tgtFst, uniqueTgtTokens);
      }
      
      // src transducer(s)
      VectorFst<LogQuadArc> firstOrderSrcFst;
      Create1stOrderSrcFst(srcTokens, firstOrderSrcFst);
      
      // compose the three transducers (tgt, grammar, src) to get the alignmentFst with weights representing tgt/src positions
      VectorFst<LogQuadArc> temp;
      Compose(tgtFst, grammarFst, &temp);
      Compose(temp, firstOrderSrcFst, &alignmentFst);

      // for debugging
      //      cerr << "====================== ALIGNMENT FST | TRUE ===========================" << endl;
      //      cerr << FstUtils::PrintFstSummary(alignmentFst) << endl;

    // build the alignment FST according to the distribution LOCAL (which uses the local subset of features in the loglinear model)
    } else if (distribution == Distribution::LOCAL) {

      // tgt transducer
      VectorFst<LogQuadArc> tgtFst;
      // unique target tokens used in tgtFst. this is populated by CreateTgtFst or CreateAllTgtFst and later used by CreatePerSentGrammarFst
      set<int> uniqueTgtTokens;

      // this distribution is only used to generate alignment Fsts, without conditioning on the reference translation.
      assert(!tgtLineIsGiven);
      CreateAllTgtFst(srcTokens, tgtTokens.size(), lexicon, tgtFst, uniqueTgtTokens);
      
      // src transducer(s)
      VectorFst<LogQuadArc> simpleSrcFst;
      CreateSimpleSrcFst(srcTokens, simpleSrcFst);
      
      // compose the three transducers (tgt, grammar, src) to get the alignmentFst with weights representing tgt/src positions
      VectorFst<LogQuadArc> temp, simpleAlignmentFst;
      Compose(tgtFst, grammarFst, &temp);
      Compose(temp, simpleSrcFst, &simpleAlignmentFst);

      // sample from the simple alignment fst, producing an fst of sample alignments (and translations). each state in the sample FST
      // should have exactly one incoming arc and one outgoing arc, except for the initial and final states. it should look like:
      //      --[]--[]--
      //     |--[]--[]--|
      //     |--[]--[]--|
      // []--|--[]--[]--|--[]
      VectorFst<LogQuadArc>& sampleAlignmentFst = alignmentFst;
      FstUtils::SampleFst(simpleAlignmentFst, sampleAlignmentFst, this->learningInfo.samplesCount);

      // for debugging only
      //cerr << FstUtils::PrintFstSummary(sampleAlignmentFst);

      // add context information to the weight of the sampled FST, and calculate the true distribution's weights accordingly
      vector<int> previousAlignment;
      previousAlignment.reserve(sampleAlignmentFst.NumStates());
      previousAlignment[0] = 0;
      for(StateIterator< VectorFst<LogQuadArc> > siter(sampleAlignmentFst); !siter.Done(); siter.Next()) {
	LogQuadArc::StateId stateId = siter.Value();

	// for debugging only
	//cerr << "visiting sampleAlignmentFst's state #" << stateId << endl;

	// invert the stopping log probability of the final state
	LogQuadWeight stoppingWeight = sampleAlignmentFst.Final(stateId);
	if(stoppingWeight != LogQuadWeight::Zero()) {
	  float dummy, stoppingLogProb;
	  FstUtils::DecodeQuad(stoppingWeight, dummy, dummy, dummy, stoppingLogProb);
	  sampleAlignmentFst.SetFinal(stateId, FstUtils::EncodeQuad(0.0, 0.0, 0.0, 0.0 - stoppingLogProb));
	  
	  // for debugging only
	  //cerr << "stopping weight of the sampleAlignmentFst's final state converted into " << FstUtils::PrintQuad(sampleAlignmentFst.Final(stateId)) << endl;
	}	

	for(MutableArcIterator< VectorFst<LogQuadArc> > aiter(&sampleAlignmentFst, stateId); !aiter.Done(); aiter.Next()) {
	  LogQuadArc arc = aiter.Value();
	  int tgtTokenId = arc.ilabel;
	  int srcTokenId = arc.olabel;
	  int toState = arc.nextstate;
	  LogQuadWeight arcWeight = arc.weight;

	  // for debugging only
	  //cerr << "arc->" << toState << " with labels " << tgtTokenId << ":" << srcTokenId << " and";

	  // variables with suffix 'Holder' are not properly set yet on the arc weight.
	  float tgtTokenPos, srcTokenPos, prevSrcTokenPosHolder, arcProbHolder;
	  FstUtils::DecodeQuad(arcWeight, tgtTokenPos, srcTokenPos, prevSrcTokenPosHolder, arcProbHolder);
	  prevSrcTokenPosHolder = previousAlignment[(int)stateId];
	  int prevSrcTokenId = prevSrcTokenPosHolder == INITIAL_SRC_POS? INITIAL_SRC_POS : srcTokens[prevSrcTokenPosHolder];
	  float trueUnnormalizedArcLogProb = params.ComputeLogProb(srcTokenId, prevSrcTokenId, tgtTokenId, 
								   srcTokenPos, prevSrcTokenPosHolder, tgtTokenPos, 
								   srcTokens.size(), tgtTokens.size(), enabledFeatureTypesFirstOrder);

	  // for debugging only
	  //cerr <<  " tgtTokenPos=" << tgtTokenPos << " srcTokenPos=" << srcTokenPos << " prevSrcTokenPos=" << prevSrcTokenPosHolder << " trueUnnormalizedArcLogProb=" << trueUnnormalizedArcLogProb << " proposalUnnormalizedArcLogProb=" << arcProbHolder;

	  // arc weight = importance weight = unnormalizedProb(arc|true distribution) / unnormalizedProb(arc|proposal distribution)
	  // note: the stopping weight on the single final state of this sampledFst = normalization constant of the proposal distribution.
	  //       this is important so that each path through the sampledFst will
	  //       have the weight = unnormalizedProb(path|true dist) / normalizedProb(path|proposal dist)
	  arcProbHolder = trueUnnormalizedArcLogProb - arcProbHolder;
	  arc.weight = FstUtils::EncodeQuad(tgtTokenPos, srcTokenPos, prevSrcTokenPosHolder, arcProbHolder);
	  aiter.SetValue(arc);

	  // for debugging only
	  //cerr << " importanceWeight=" << arcProbHolder << endl;

	  // remember the alignment decision that led to 'arc.nextstate'
	  previousAlignment[toState] = srcTokenPos;
	}
      }

      // for debugging only
      //cerr << "=================SAMPLED ALIGNMENT FST============" << endl;
      //cerr << FstUtils::PrintFstSummary(alignmentFst);
      //cerr << endl << "finished fixing the sampleAlignmentFst weights" << endl;
      //cerr << "alignmentFst is ready" << endl << endl;
    } else if(distribution == Distribution::CUSTOM) {
      assert(learningInfo.customDistribution != 0);

      // draw samples from the custom distribution
      vector< vector<int> > translations, alignments;
      vector< double > logProbs;
      vector< int > emptyIntVector;
      for(int i = 0; i < learningInfo.samplesCount; i++) {
	double logProb = -1;
	translations.push_back(emptyIntVector);
	alignments.push_back(emptyIntVector);
	learningInfo.customDistribution->SampleAT(srcTokens, tgtTokens.size(), translations[i], alignments[i], logProb);
	logProbs.push_back(logProb);
      }

      // build an alignment FST using those samples
      CreateSampleAlignmentFst(srcTokens, translations, alignments, logProbs, alignmentFst);

      // for debugging only
      //            cerr << "=========translations/alignments sampled from the custom distribution========" << endl;
      //            cerr << FstUtils::PrintFstSummary(alignmentFst) << endl;
    }

    // save the resulting alignmentFst for future use
    if(learningInfo.saveAlignmentFstsOnDisk) {
      alignmentFst.Write(alignmentFstFilename.str());
    }
    
  } else if (learningInfo.saveAlignmentFstsOnDisk) {
    // in subsequent passes, read the previously stored FST
        
    // TODO: debug this block. something is wrong with the read operation. The huge FSTs, when read, don't have any states.
    //cerr << "hi" << endl;
    alignmentFst.Read(alignmentFstFilename.str());
    bool verified = Verify(alignmentFst);
    if (verified) {
      //cerr << "the model loaded was verified." << endl;
    } else {
      //cerr << "the model loaded was not verified." << endl;
    }

    if(!tgtLineIsGiven && sentId == 1) {
      cerr << FstUtils::PrintFstSummary(alignmentFst);
    }
    string dummy;
    cin >> dummy;

  } else {
    // unexpected error!
    assert(false);
  }
  
  // if another distribution were used to generate the alignmentFst, then we have already set the weights appropriately
  if(distribution == Distribution::TRUE) {
    // compute the probability of each transition on the alignment FST according to the current model parameters
    // set the fourth value in the LogQuadWeights on the arcs = the computed prob for that arc
    for(StateIterator< VectorFst<LogQuadArc> > siter(alignmentFst); !siter.Done(); siter.Next()) {
      LogQuadArc::StateId stateId = siter.Value();
      for(MutableArcIterator< VectorFst<LogQuadArc> > aiter(&alignmentFst, stateId); !aiter.Done(); aiter.Next()) {
	LogQuadArc arc = aiter.Value();
	int tgtTokenId = arc.ilabel;
	int srcTokenId = arc.olabel;
	float tgtTokenPos, srcTokenPos, prevSrcTokenPos, dummy;
	FstUtils::DecodeQuad(arc.weight, tgtTokenPos, srcTokenPos, prevSrcTokenPos, dummy);
	int prevSrcTokenId = prevSrcTokenPos == INITIAL_SRC_POS? INITIAL_SRC_POS : srcTokens[prevSrcTokenPos];
	float arcProb = params.ComputeLogProb(srcTokenId, prevSrcTokenId,  tgtTokenId, srcTokenPos, prevSrcTokenPos, tgtTokenPos, 
					      srcTokens.size(), tgtTokens.size(), enabledFeatureTypesFirstOrder);
	arc.weight = FstUtils::EncodeQuad(tgtTokenPos, srcTokenPos, prevSrcTokenPos, arcProb);
	aiter.SetValue(arc);
      }
    }
  }
}

// assumptions:
// - translations, alignemnts and logProbs have the same length, i.e. the number of samples generated from the proposal distribution.
// - translations[i], alignments[i] and logProbs[i] refer to the tgt token sequence, the alignemnt sequence and the normalized logprob of sample #i.
// - furthermore, translations[i] and alignments[i] have the same length, i.e. the number of target tokens in the translation.
// - the first element in srcTokens is the NULL token.
// each sample will have its own final state. the stopping weight = 1/proposal_prob(sample) 
// i.e. the last LogWeight in the LogQuadWeight on the arc is set to -logProbs[sampleId]
// each alignment arc carries the following information: srcToken, tgtToken, tgtPosition, srcPosition, previousSrcPosition, arcWeight
// all this information can be inferred from the sample's alignment/translations vector, except for the arcWeight.
// we set the arcWeight to the unnormalized prior probability of the arc, according to the true distribution
// this way, the weight of any complete path in this alignment FST is equal to the importance weight = unnormalized_true_p(sample) / proposal_p(sample)
void LogLinearModel::CreateSampleAlignmentFst(const vector<int>& srcTokens,
					      const vector< vector<int> >& translations, 
					      const vector< vector<int> >& alignments, 
					      const vector< double >& logProbs,
					      VectorFst< LogQuadArc >& alignmentFst) {
  assert(alignmentFst.NumStates() == 0);
  assert(translations.size() == alignments.size());
  assert(translations.size() == logProbs.size());
  assert(srcTokens.size() > 0 && srcTokens[0] == NULL_SRC_TOKEN_ID);
  
  // create a start state
  int startState = alignmentFst.AddState();
  alignmentFst.SetStart(startState);
  
  // for each sample
  for(int i = 0; i < translations.size(); i++) {
    assert(translations[i].size() == alignments[i].size());
    
    // start each sample's path with the start state
    int previousState = startState;
    int previousSrcPosition = INITIAL_SRC_POS;

    // for each target position
    for(int tgtPos = 0; tgtPos < translations[i].size(); tgtPos++) {
      
      // create the next state in this sample's path, and make it final if it's the last one.
      int nextState = alignmentFst.AddState();
      if(tgtPos == translations[i].size() - 1) {
	alignmentFst.SetFinal(nextState, FstUtils::EncodeQuad(0,0,0,-logProbs[i]));
      }

      // prepare the information needed on this transition
      int srcPos = alignments[i][tgtPos];
      // previousSrcPosition = previousSrcPosition
      // tgtPos = tgtPos
      int srcToken = srcTokens[srcPos];
      int tgtToken = translations[i][tgtPos];
      int prevSrcToken = previousSrcPosition == INITIAL_SRC_POS? INITIAL_SRC_POS : srcTokens[previousSrcPosition];
      float unnormalizedTrueDistPriorProb = 
	params.ComputeLogProb(srcToken, prevSrcToken, tgtToken, srcPos, previousSrcPosition, tgtPos, 
			      srcTokens.size(), translations[i].size(), enabledFeatureTypesFirstOrder);
      
      // now add the arc
      alignmentFst.AddArc(previousState, 
			  LogQuadArc(tgtToken, srcToken, 
				     FstUtils::EncodeQuad(tgtPos, srcPos, previousSrcPosition, unnormalizedTrueDistPriorProb),
				     nextState));
      
      // update previousSrcPosition and previousState
      previousSrcPosition = (srcPos == NULL_SRC_TOKEN_POS)?
	previousSrcPosition:
	srcPos;
      previousState = nextState;
    }
  }
}

void LogLinearModel::AddSentenceContributionToGradient(const VectorFst< LogQuadArc> &descriptorFst, 
						       const VectorFst< LogArc > &totalProbFst, 
						       LogLinearParams& gradient,
						       const vector<int> &srcTokens,
						       int tgtTokensCount,
						       bool subtract) {
  clock_t d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0, temp;
  temp = clock();
  // make sure the totalProbFst is a shadow fst of descriptorFst
  assert(descriptorFst.NumStates() == totalProbFst.NumStates());

  // traverse arcs
  for (int stateId = 0; stateId < descriptorFst.NumStates() ;stateId++) {
    d1 += clock() - temp;
    temp = clock();
    ArcIterator< VectorFst< LogArc > > totalProbArcIter(totalProbFst, stateId);
    for (ArcIterator< VectorFst< LogQuadArc > > descriptorArcIter(descriptorFst, stateId);
	 !descriptorArcIter.Done() && !totalProbArcIter.Done();
	 descriptorArcIter.Next(), totalProbArcIter.Next()) {
      d2 += clock() - temp;
      temp = clock();

      // make sure the totalProbFst is a shadow fst of descriptorFst
      assert(descriptorArcIter.Value().ilabel == totalProbArcIter.Value().ilabel);
      assert(descriptorArcIter.Value().olabel == totalProbArcIter.Value().olabel);
      assert(!descriptorArcIter.Done());
      assert(!totalProbArcIter.Done());
      assert(descriptorArcIter.Value().nextstate == totalProbArcIter.Value().nextstate);

      // parse the descriptorArc and totalProbArc
      int tgtToken = descriptorArcIter.Value().ilabel;
      int srcToken = descriptorArcIter.Value().olabel;
      float tgtPos, srcPos, prevSrcPos, dummy;
      FstUtils::DecodeQuad(descriptorArcIter.Value().weight, tgtPos, srcPos, prevSrcPos, dummy);
      LogWeight totalProb = totalProbArcIter.Value().weight;

      // make sure the weight on this arc in the totalProbFst is a valid probability
      assert(totalProb.Value() >= FstUtils::LOG_PROBS_MUST_BE_GREATER_THAN_ME);

      d3 += clock() - temp;
      temp = clock();

      // find the features activated on this transition, and their values
      map<string, float> activeFeatures;
      int prevSrcToken = INITIAL_SRC_POS ? INITIAL_SRC_POS : srcTokens[(int)prevSrcPos];
      // epsilon arc
      if(srcToken == 0 && tgtToken == 0) {
	continue;
      }
      assert(srcToken != 0);
      assert(tgtToken != 0 && tgtPos != 0);
      gradient.FireFeatures(srcToken, prevSrcToken, tgtToken, (int)srcPos, (int)prevSrcPos, (int)tgtPos, 
			    srcTokens.size(), tgtTokensCount, enabledFeatureTypesFirstOrder, activeFeatures);
      d4 += clock() - temp;
      temp = clock();
      // for debugging
      //      cerr << endl << "=================features fired===================" << endl;
      //      cerr << "arc: tgtToken=" << tgtToken << " srcToken=" << srcToken << " tgtPos=" << tgtPos << " srcPos=" << srcPos;
      //      cerr << " srcTokensCount=" << srcTokensCount << " tgtTokensCount=" << tgtTokensCount << " totalLogProb=" << totalProb << endl;
      //      cerr << "Features fired are:" << endl;

      // now, for each feature fired on each arc of aGivenTS
      for(map<string,float>::const_iterator feature = activeFeatures.begin(); feature != activeFeatures.end(); feature++) {
	if(gradient.params.count(feature->first) == 0) {
	  gradient.params[feature->first] = 0;
	}

	// for debugging 
	//	cerr << "val[" << feature->first << "]=" << feature->second << ". gradient[" << feature->first << "] was " << gradient.params[feature->first] << " became ";
	
	// the positive contribution to the derivative of this feature by this arc is
	float contribution = feature->second * FstUtils::nExp(totalProb.Value());
	if(subtract) {
	  gradient.params[feature->first] -= contribution;
	} else {
	  gradient.params[feature->first] += contribution;
	}

	// for debugging
	//cerr << gradient.params[feature->first] << endl;
      }
      d5 += clock() - temp;
      temp = clock();
    }
    temp = clock();
  }
  //  string dummy2;
  //  cin >> dummy2;
  //  cerr << "d1=" << d1 << " d2=" << d2 << " d3=" << d3 << " d4=" << d4 << " d5=" << d5 << endl;
}

// for each feature in the model, add the corresponding regularization term to the gradient
void LogLinearModel::AddRegularizerTerm(LogLinearParams& gradient) {

  // compute ||params||_2
  float l2 = 0;
  if(learningInfo.optimizationMethod.regularizer == Regularizer::L2) {
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
    switch(learningInfo.optimizationMethod.regularizer) {
    case Regularizer::L2:
      term = 2.0 * learningInfo.optimizationMethod.regularizationStrength * featureIter->second / l2;
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

  clock_t accUpdateClocks = 0, accShortestDistanceClocks = 0, accBuildingFstClocks = 0, accReadClocks = 0, accRuntimeClocks = 0, accRegularizationClocks = 0, accWriteClocks = 0, accGenericClocks = 0;

  // passes over the training data
  int iterationCounter = 0;
  bool outputAlignments = false, done = false;
  do {

    //for debugging
    //cerr << "=============" << endl;
    //cerr << "iteration # " << iterationCounter++ << endl;
    //cerr << "=============" << endl << endl;

    clock_t timestamp1 = clock();

    float logLikelihood = 0;

    ifstream srcCorpus(srcCorpusFilename.c_str(), ios::in); 
    ifstream tgtCorpus(tgtCorpusFilename.c_str(), ios::in); 

    // IMPORTANT NOTE: this is the gradient of the regualarized log-likelihood, in the real-domain, not in the log-domain.
    // in other words, when the equations on paper say we should gradient[feature] += x, we effectively need to add 
    // (rather than log-add) e^{- -log(x) } (because we only have x in the log-domain -log(x)) which is equivalent to += x.
    // TODO OPTIMIZATION: instead of defining gradient here, define it as a class member.
    // TODO (refactoring): LogLinearParams requires initialization thru this constructor, but we don't need them for the gradient
    LogLinearParams gradient(params.srcTypes, params.tgtTypes, params.ibmModel1ForwardScores, params.ibmModel1BackwardScores);

    // this is needed only for the cumulative L1 regularizer. it reprsents the total actual l1 penalty each feature has received
    // so far (i.e. it's updated every batchsize sentences) in this iteration
    // TODO (refactoring): LogLinearParams requires initialization thru this constructor, but we don't need them for this instance
    LogLinearParams appliedL1Penalty(params.srcTypes, params.tgtTypes, params.ibmModel1ForwardScores, params.ibmModel1BackwardScores);
    
    // this is needed only for the cumulative L1 regularizer. it represents the total l1 penalty any feature should have receive so far.
    // it's updated every batchsize sentences.
    double correctL1Penalty = 0;

    // first, compute the regularizer's term for each feature in the gradient
    clock_t timestamp2 = clock();
    if(learningInfo.optimizationMethod.algorithm == OptUtils::GRADIENT_DESCENT) {
      AddRegularizerTerm(gradient);
    }
    accRegularizationClocks += clock() - timestamp2;
    
    // for each line
    string srcLine, tgtLine;
    int sentsCounter = 1;
    clock_t timestamp3 = clock();
    while(getline(srcCorpus, srcLine)) {
      getline(tgtCorpus, tgtLine);
      accReadClocks += clock() - timestamp3;

      //for debugging
      //cerr << "srcLine: " << srcLine << endl;
      //cerr << "tgtLine: " << tgtLine << endl << endl;

      // read the list of integers representing target tokens
      vector< int > srcTokens, tgtTokens;
      StringUtils::ReadIntTokens(srcLine, srcTokens);
      StringUtils::ReadIntTokens(tgtLine, tgtTokens);
      // add the null src token as a possible alignment for any target token
      srcTokens.insert(srcTokens.begin(), 1, NULL_SRC_TOKEN_ID);
      
      // build FST(a|t,s) and build FST(a,t|s)
      clock_t timestamp4 = clock();
      VectorFst< LogQuadArc > aGivenTS, aTGivenS, tgtFst, dummy;
      //cerr << "building aGivenTS" << endl;
      BuildAlignmentFst(srcTokens, tgtTokens, aGivenTS, true, learningInfo.neighborhood, sentsCounter, Distribution::TRUE, tgtFst);
      //cerr << "building aTGivenS" << endl;
      BuildAlignmentFst(srcTokens, tgtTokens, aTGivenS, false, learningInfo.neighborhood, sentsCounter, learningInfo.distATGivenS, dummy);
      // union aGivenTS into aTGivenS so that we have good samples as well as bad samples
      // TODO: currently, we don't control how much weight goes to (a,t) pairs coming from aGivenTS vs. aTGivenS when we do this
      //       union. It would be better if we control it in a smart way.  
      if(learningInfo.distATGivenS != Distribution::TRUE && learningInfo.unionAllCompatibleAlignments) {
      	Union(&aTGivenS, aGivenTS);
      }

      // for debugging
      //cerr << "both aGivenTS and aTGivenS was built" << endl;

      // change the LogQuadWeight semiring to LogWeight using LogQuadToLogMapper
      VectorFst< LogArc > aGivenTSProbs, aTGivenSProbs;
      //cerr << "building aGivenTSProbs" << endl;
      ArcMap(aGivenTS, &aGivenTSProbs, LogQuadToLogMapper());
      //cerr << "building aTGivenSProbs" << endl;
      ArcMap(aTGivenS, &aTGivenSProbs, LogQuadToLogMapper());
      //cerr << "aTGivenS's start state = " << aTGivenS.Start() << " while aTGivenSProbs's start state = " << aTGivenSProbs.Start() << endl;

      // output alignments (after the model converges)
      VectorFst< StdArc > bestAlignment;
      if(outputAlignments) {
	// tropical has the path property
	VectorFst< StdArc > aGivenTSProbsWithPathProperty;
	ArcMap(aGivenTSProbs, &aGivenTSProbsWithPathProperty, LogToTropicalMapper());
	cerr << "===============best alignment==============================" << endl;
	cerr << "sent " << sentsCounter << endl;
	ShortestPath(aGivenTSProbsWithPathProperty, &bestAlignment);
	cerr << FstUtils::PrintFstSummary(bestAlignment);
      }

      // prune paths whose probabilities are less than 1/e^3 of the best path
      //      cerr << "aTGivenSProbs size: " << endl;
      //      cerr << "before pruning: " << aTGivenSProbs.NumStates() << " stataes" << endl;
      //      VectorFst<StdArc> prunableFst;
      //      ArcMap(aTGivenSProbs, &prunableFst, LogToTropicalMapper());
      //      Prune(&prunableFst, 1);
      //      ArcMap(prunableFst, &aTGivenSProbs, TropicalToLogMapper());
      //      cerr << "after pruning: " << aTGivenSProbs.NumStates() << " stataes" << endl;
      //      cerr << endl;

      accBuildingFstClocks += clock() - timestamp4;
      
      // for debugging
      //      cerr << "========================aGivenTSProbs=====================" << endl;
      //      FstUtils::PrintFstSummary(aGivenTSProbs);
      //      string dummy;
      //      cin >> dummy;

      // get the posterior probability of traversing each arc given src and tgt(s).
      clock_t timestamp5 = clock();
      VectorFst< LogArc > aGivenTSTotalProb, aTGivenSTotalProb;
      LogWeight aGivenTSBeta0, aTGivenSBeta0;
      //cerr << "building aGivenTSTotalProb" << endl;
      FstUtils::ComputeTotalProb<LogWeight,LogArc>(aGivenTSProbs, aGivenTSTotalProb, aGivenTSBeta0);
      //cerr << "building aTGivenSTotalProb" << endl;
      FstUtils::ComputeTotalProb<LogWeight,LogArc>(aTGivenSProbs, aTGivenSTotalProb, aTGivenSBeta0);
      accShortestDistanceClocks += clock() - timestamp5;

      // add this sentence's contribution to the gradient of model parameters.
      // luckily, the contribution factorizes into: the end-to-end arc probabilities and beta[0] of aGivenTS and aTGivenS.
      clock_t timestamp6 = clock();
      //      cerr << "===========================" << endl << "add sent #" << sentsCounter << " contribution to E(features) according to p(a|T,S)" << endl;
      //      cerr << "===========================" << endl;
      AddSentenceContributionToGradient(aGivenTS, aGivenTSTotalProb, gradient, srcTokens, tgtTokens.size(), false);

      clock_t timestamp6d5 = clock();
      //      cerr << "===========================" << endl << "add sent contribution to E(features) according to p(a,T|S)" << endl;
      AddSentenceContributionToGradient(aTGivenS, aTGivenSTotalProb, gradient, srcTokens, tgtTokens.size(), true);
      accGenericClocks += clock() - timestamp6d5;

      // update the iteration log likelihood with this sentence's likelihod
      // the update is equal to \sum_{a compatible with reference t} p(a,t|s) = ShortestDistance(FstCompose(ref t, aTGivenSProbs / beta0 )) 
      assert(aTGivenSBeta0.Value() != 0);
      VectorFst< LogArc > tgtFstLogArc, constrainedATGivenSProbs;
      ArcMap(tgtFst, &tgtFstLogArc, LogQuadToLogMapper());
      Compose(tgtFstLogArc, aTGivenSProbs, &constrainedATGivenSProbs);
      FstUtils::MakeOneFinalState(constrainedATGivenSProbs);
      int finalState = FstUtils::FindFinalState(constrainedATGivenSProbs);
      assert(finalState != -1);
      constrainedATGivenSProbs.SetFinal(finalState, Divide(LogWeight::One(), aTGivenSBeta0));
      vector< LogWeight > betas;
      ShortestDistance(constrainedATGivenSProbs, &betas, true);
      LogWeight constrainedATGivenSBeta0 = betas[constrainedATGivenSProbs.Start()];
      //      cerr << "constrainedATGivenSBeta0 = " << constrainedATGivenSBeta0.Value() << endl;
      assert(constrainedATGivenSBeta0.Value() >= FstUtils::LOG_PROBS_MUST_BE_GREATER_THAN_ME);
      logLikelihood += constrainedATGivenSBeta0.Value();
      
      // for debugging only
      //      cerr << "====================constrainedATGivenSProbs==============" << endl;
      //      cerr << FstUtils::PrintFstSummary(constrainedATGivenSProbs) << endl;
      //      cerr << "=============sentence's likelihood===========" << endl;
      cerr << "-log p(ref translation|s) = " << constrainedATGivenSBeta0.Value() << endl;
      //      cerr << "iteration's loglikelihood now = " << logLikelihood << endl << endl;

      // if using cumulative L1, update correctL1Penalty
      if(learningInfo.optimizationMethod.regularizer == Regularizer::L1) {
	correctL1Penalty += learningInfo.optimizationMethod.learningRate * learningInfo.optimizationMethod.regularizationStrength / this->corpusSize;
      }

      // if the optimization algorithm is stochastic, update the parameters here.
      cerr << "s";
      if(IsStochastic(learningInfo.optimizationMethod.algorithm) && sentsCounter % learningInfo.optimizationMethod.miniBatchSize == 0) {
	cerr << "u";
	params.UpdateParams(gradient, learningInfo.optimizationMethod);
	if(learningInfo.optimizationMethod.regularizer == Regularizer::L1) {
	  params.ApplyCumulativeL1Penalty(gradient, appliedL1Penalty, correctL1Penalty);
	}
	gradient.Clear();
      }
      accUpdateClocks += clock() - timestamp6;

      // for debugging only
      // report accumulated times
      //      cerr << "accumulated runtime = " << (float) accRuntimeClocks / CLOCKS_PER_SEC << " sec." << endl;
      //      cerr << "accumulated disk write time = " << (float) accWriteClocks / CLOCKS_PER_SEC << " sec." << endl;
      //      cerr << "accumulated disk read time = " << (float) accReadClocks / CLOCKS_PER_SEC << " sec." << endl;
      //      cerr << "accumulated fst construction time = " << (float) accBuildingFstClocks / CLOCKS_PER_SEC << " sec." << endl;
      //      cerr << "accumulated fst shortest-distance time = " << (float) accShortestDistanceClocks / CLOCKS_PER_SEC << " sec." << endl;
      //      cerr << "accumulated param update time = " << (float) accUpdateClocks / CLOCKS_PER_SEC << " sec." << endl;
      //      cerr << "accumulated regularization time = " << (float) accRegularizationClocks / CLOCKS_PER_SEC << " sec." << endl;
      //      cerr << "=========finished processing sentence " << sentsCounter << "=============" << endl;
      
      // logging
      if (sentsCounter % 1 == 1000) {
      	cerr << endl << sentsCounter << " sents processed.." << endl;
      }
      sentsCounter++;

      timestamp3 = clock();
    }

    // if cumulative L1 regularization is being used, now is the time to update all weights with the difference between
    // the correctL1Penalty and the appliedL1Penalty 
    if(learningInfo.optimizationMethod.regularizer == Regularizer::L1) {
      params.ApplyCumulativeL1Penalty(params, appliedL1Penalty, correctL1Penalty);
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

    // if we output alignments during this iteration, then we're done
    done = false;
    if(outputAlignments) { 
      done = true;
    } else {
      // otherwise, check for model convergence to decide whether or not next iteration will output alignments
      outputAlignments = learningInfo.IsModelConverged();
      if(outputAlignments) {
	cerr << "===============================================" << endl;
	cerr << "==WE ARE DONE TRAINING. NOW OUTPUT ALIGNMENTS==" << endl;
	cerr << "===============================================" << endl;
      }
    }
   
  } while(!done);

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
  cerr << "accumulated generic time = " << (float) accGenericClocks / CLOCKS_PER_SEC << " sec." << endl;
}

// given the current model, align a test sent
// assumptions: 
// - the null token has *NOT* been inserted yet
string LogLinearModel::AlignSent(vector<int> srcTokens, vector<int> tgtTokens) {
  
  static int sentCounter = 0;
  
  // insert the null token
  assert(srcTokens.size() > 0 && srcTokens[0] != NULL_SRC_TOKEN_ID);
  srcTokens.insert(srcTokens.begin(), 1, NULL_SRC_TOKEN_ID);
  
  // build aGivenTS
  VectorFst< LogQuadArc > aGivenTS, dummy;
  BuildAlignmentFst(srcTokens, tgtTokens, aGivenTS, true, DiscriminativeLexicon::COOCC, sentCounter, Distribution::TRUE, dummy);
  VectorFst< LogArc > aGivenTSProbs;
  ArcMap(aGivenTS, &aGivenTSProbs, LogQuadToLogPositionMapper());
  // tropical has the path property
  VectorFst< StdArc > aGivenTSProbsWithPathProperty, bestAlignment;
  ArcMap(aGivenTSProbs, &aGivenTSProbsWithPathProperty, LogToTropicalMapper());
  ShortestPath(aGivenTSProbsWithPathProperty, &bestAlignment);
  return FstUtils::PrintAlignment(bestAlignment);
}
