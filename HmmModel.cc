#include "HmmModel.h"

using namespace std;
using namespace fst;
using namespace MultinomialParams;

// initialize model 1 scores
HmmModel::HmmModel(const string& srcIntCorpusFilename, 
		   const string& tgtIntCorpusFilename, 
		   const string& outputFilenamePrefix, 
		   const LearningInfo& learningInfo) {

  // Note: seed with time(0) if you don't care about reproducbility
  srand(425);

  // set member variables
  this->outputPrefix = outputFilenamePrefix;
  this->learningInfo = learningInfo;

  // encode training data
  vocabEncoder.Read(srcIntCorpusFilename, srcSents);
  vocabEncoder.Read(tgtIntCorpusFilename, tgtSents);
  assert(srcSents.size() > 0 && srcSents.size() == tgtSents.size());

  // initialize the model parameters
  cerr << "init hmm params" << endl;
  InitParams();
  if(learningInfo.mpiWord->rank() == 0) {
    stringstream initialModelFilename;
    initialModelFilename << outputPrefix << ".param.init";
    PersistParams(initialModelFilename.str());
  }

  // create the initial grammar FST
  cerr << "create grammar fst" << endl;
  CreateGrammarFst();
}

void HmmModel::Train() {

  // create tgt fsts
  cerr << "create tgt fsts" << endl;
  vector< VectorFst <LogQuadArc> > tgtFsts;
  CreateTgtFsts(tgtFsts);

  // training iterations
  cerr << "train!" << endl;
  LearnParameters(tgtFsts);

  // persist parameters
  cerr << "persist" << endl;
  PersistParams(outputPrefix + ".param.final");
}

// src fsts are 1st order markov models
void HmmModel::CreateSrcFsts(vector< VectorFst< LogQuadArc > >& srcFsts) {
  for(unsigned sentId = 0; sentId < srcSents.size(); sentId++) {
    vector< int > &intTokens = srcSents[sentId];
    assert(intTokens[0] == NULL_SRC_TOKEN_ID);

    // create the fst
    VectorFst< LogQuadArc > srcFst;
    Create1stOrderSrcFst(intTokens, srcFst);
    srcFsts.push_back(srcFst);
  }
}

// assumptions:
// - tgtFst is empty
void HmmModel::CreateTgtFst(const vector<int> tgtTokens, VectorFst< LogQuadArc > &tgtFst) {
  assert(tgtFst.NumStates() == 0);
  int statesCount = tgtTokens.size() + 1;
  for(int stateId = 0; stateId < tgtTokens.size()+1; stateId++) {
    int temp = tgtFst.AddState();
    assert(temp == stateId);
    if(stateId == 0) continue;
    int tgtPos = stateId;
    tgtFst.AddArc(stateId-1, 
		  LogQuadArc(tgtTokens[stateId-1], 
			       tgtTokens[stateId-1], 
			       FstUtils::EncodeQuad(tgtPos, 0, 0, 0), 
			       stateId));
  }
  tgtFst.SetStart(0);
  tgtFst.SetFinal(tgtTokens.size(), LogQuadWeight::One());
  ArcSort(&tgtFst, ILabelCompare<LogQuadArc>());
}

void HmmModel::CreateTgtFsts(vector< VectorFst< LogQuadArc > >& targetFsts) {
  // for each line
  for(unsigned sentId = 0; sentId < srcSents.size(); sentId++) {
    vector< int > &intTokens = tgtSents[sentId];
    
    // create the fst
    VectorFst< LogQuadArc > tgtFst;
    CreateTgtFst(intTokens, tgtFst);
    targetFsts.push_back(tgtFst);
  }
}

void HmmModel::NormalizeFractionalCounts() {
  MultinomialParams::NormalizeParams(aFractionalCounts);
  MultinomialParams::NormalizeParams(tFractionalCounts);
}

void HmmModel::PrintParams() {
  MultinomialParams::PrintParams(aParams);
  MultinomialParams::PrintParams(tFractionalCounts);
}

void HmmModel::PersistParams(const string& outputFilename) {
  ofstream paramsFile(outputFilename.c_str());
  cerr << "rank #" << learningInfo.mpiWorld->rank() << ": writing model params at " << outputFilename << endl;
  paramsFile << "=============== translation parameters p(tgtWord|srcWord) ============" << endl;
  MultinomialParams::PersistParams(paramsFile, tFractionalCounts);
  paramsFile << endl << "=============== alignment parameters p(a_i|a_{i-1}) ==================" << endl;
  MultinomialParams::PersistParams(paramsFile, aParams);
  paramsFile.close();
}

// finds out what are the parameters needed by reading hte corpus, and assigning initial weights based on the number of co-occurences
void HmmModel::InitParams() {
  // for each parallel sentence
  for(int sentId = 0; sentId < srcSents.size(); sentId++) {

    // read the list of integers representing target tokens
    vector< int > &tgtTokens = tgtSents[sentId], &srcTokens = srcSents[sentId];
    
    // we want to allow target words to align to NULL (which has srcTokenId = 1).
    srcTokens.push_back(NULL_SRC_TOKEN_ID); 
    
    // for each srcToken
    for(int i=0; i<srcTokens.size(); i++) {

      // INITIALIZE TRANSLATION PARAMETERS
      int srcToken = srcTokens[i];
      // get the corresponding map of tgtTokens (and the corresponding probabilities)
      map<int, double> tParamsGivenS_i = tFractionalCounts[srcToken];
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
    
  NormalizeFractionalCounts();
  DeepCopy(aFractionalCounts, aParams);
}

// make a deep copy of parameters
void HmmModel::DeepCopy(const ConditionalMultinomialParam& original, 
			ConditionalMultinomialParam& duplicate) {
  // zero duplicate
  MultinomialParams::ClearParams(duplicate);

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
  LogQuadArc::StateId dummy = grammarFst.AddState();
  assert(dummy == 0);
  grammarFst.SetStart(0);
  grammarFst.SetFinal(0, LogQuadWeight::One());
  int fromState = 0, toState = 0;
  for(ConditionalMultinomialParam::const_iterator srcIter = tFractionalCounts.begin(); srcIter != tFractionalCounts.end(); srcIter++) {
    for(MultinomialParam::const_iterator tgtIter = (*srcIter).second.begin(); tgtIter != (*srcIter).second.end(); tgtIter++) {
      int tgtToken = (*tgtIter).first;
      int srcToken = (*srcIter).first;
      float paramValue = (*tgtIter).second;
      grammarFst.AddArc(fromState, 
			LogQuadArc(tgtToken, 
				     srcToken, 
				     FstUtils::EncodeQuad(0, 0, 0, paramValue), 
				     toState));
    }
  }
  ArcSort(&grammarFst, ILabelCompare<LogQuadArc>());
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
void HmmModel::Create1stOrderSrcFst(const vector<int>& srcTokens, VectorFst<LogQuadArc>& srcFst) {
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
      srcFst.SetFinal(i, LogQuadWeight::One());
    }

    // we don't allow prevAlignment to be null alignment in our markov model. if a null alignment happens after alignment = 5, we use 5 as prevAlignment, not the null alignment. if null alignment happens before any non-null alignment, we use a special src position INITIAL_SRC_POS to indicate the prevAlignment
    int prevAlignment = i == 0? INITIAL_SRC_POS : i;

    // each state can go to itself with the null src token
    srcFst.AddArc(i, LogQuadArc(srcTokens[0], srcTokens[0], FstUtils::EncodeQuad(0, i, prevAlignment, aParams[prevAlignment][i]), i));

    // each state can go to states representing non-null alignments
    for(int j = 1; j < srcTokens.size(); j++) {
      srcFst.AddArc(i, LogQuadArc(srcTokens[j], srcTokens[j], FstUtils::EncodeQuad(0, j, prevAlignment, aParams[prevAlignment][j]), j));
    }
  }
 
  // arc sort to enable composition
  ArcSort(&srcFst, ILabelCompare<LogQuadArc>());

  // for debugging
  //  cerr << "=============SRC FST==========" << endl;
  //  cerr << FstUtils::PrintFstSummary(srcFst);
}

void HmmModel::ClearFractionalCounts() {
  MultinomialParams::ClearParams(tFractionalCounts);
  MultinomialParams::ClearParams(aFractionalCounts);
}

void HmmModel::BuildAlignmentFst(const VectorFst< LogQuadArc > &tgtFst, 
				 const VectorFst< LogQuadArc > &srcFst, 
				 VectorFst< LogQuadArc > &alignmentFst) {
  VectorFst< LogQuadArc > temp;
  Compose(tgtFst, grammarFst, &temp);
  Compose(temp, srcFst, &alignmentFst);  
}

void HmmModel::LearnParameters(vector< VectorFst< LogQuadArc > >& tgtFsts) {
  clock_t compositionClocks = 0, forwardBackwardClocks = 0, updatingFractionalCountsClocks = 0, normalizationClocks = 0;
  clock_t t00 = clock();
  do {
    clock_t t05 = clock();

    // create src fsts (these encode the aParams as weights on their arcs)
    cerr << "create src fsts" << endl;
    vector< VectorFst <LogQuadArc> > srcFsts;
    CreateSrcFsts(srcFsts);

    clock_t t10 = clock();
    float logLikelihood = 0, validationLogLikelihood = 0;
    //    cerr << "iteration's loglikelihood = " << logLikelihood << endl;
    
    // this vector will be used to accumulate fractional counts of parameter usages
    ClearFractionalCounts();
    
    // iterate over sentences
    int sentsCounter = 0;
    for( vector< VectorFst< LogQuadArc > >::const_iterator tgtIter = tgtFsts.begin(), srcIter = srcFsts.begin(); 
	 tgtIter != tgtFsts.end() && srcIter != srcFsts.end(); 
	 tgtIter++, srcIter++) {

      // every core works on its sentences
      if(sentsCounter % learningInfo.mpiWorld->size() != learningInfo.mpiWorld->rank()) {
	continue;
      }

      // build the alignment fst
      clock_t t20 = clock();
      const VectorFst< LogQuadArc > &tgtFst = *tgtIter, &srcFst = *srcIter;
      VectorFst< LogQuadArc > alignmentFst;
      BuildAlignmentFst(tgtFst, srcFst, alignmentFst);
      compositionClocks += clock() - t20;
      
      // run forward/backward for this sentence
      clock_t t30 = clock();
      vector<LogQuadWeight> alphas, betas;
      ShortestDistance(alignmentFst, &alphas, false);
      ShortestDistance(alignmentFst, &betas, true);
      float fSentLogLikelihood, dummy;
      FstUtils::DecodeQuad(betas[alignmentFst.Start()], 
			     dummy, dummy, dummy, fSentLogLikelihood);
      forwardBackwardClocks += clock() - t30;
      
      // compute and accumulate fractional counts for model parameters
      clock_t t40 = clock();
      bool excludeFractionalCountsInThisSent = 
	learningInfo.useEarlyStopping && 
	sentsCounter % learningInfo.trainToDevDataSize == 0;
      for (int stateId = 0; !excludeFractionalCountsInThisSent && stateId < alignmentFst.NumStates() ;stateId++) {
	for (ArcIterator<VectorFst< LogQuadArc > > arcIter(alignmentFst, stateId);
	     !arcIter.Done();
	     arcIter.Next()) {

	  // decode arc information
	  int srcToken = arcIter.Value().olabel, tgtToken = arcIter.Value().ilabel;
	  int fromState = stateId, toState = arcIter.Value().nextstate;
	  float fCurrentTgtPos, fCurrentSrcPos, fPrevSrcPos, arcLogProb;
	  FstUtils::DecodeQuad(arcIter.Value().weight, fCurrentTgtPos, fCurrentSrcPos, fPrevSrcPos, arcLogProb);
	  int currentSrcPos = (int) fCurrentSrcPos, prevSrcPos = (int) fPrevSrcPos;

	  // probability of using this parameter given this sentence pair and the previous model
	  float alpha, beta, dummy;
	  FstUtils::DecodeQuad(alphas[fromState], dummy, dummy, dummy, alpha);
	  FstUtils::DecodeQuad(betas[toState], dummy, dummy, dummy, beta);
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
	cerr << sentsCounter << " sents processed. so far, iterationLoglikelihood on this core = " << logLikelihood <<  endl;
      }
    }
    
    // all processes send their fractional counts to the master and the master accumulates them
    mpi::reduce<MultinomialParams::ConditionalMultinomialParam>(*learningInfo.mpiWorld, tFractionalCounts, tFractionalCounts, MultinomialParams::AccumulateConditionalMultinomials, 0);
    mpi::reduce<MultinomialParams::ConditionalMultinomialParam>(*learningInfo.mpiWorld, aFractionalCounts, aFractionalCounts, MultinomialParams::AccumulateConditionalMultinomials, 0);
    mpi::all_reduce<float>(*learningInfo.mpiWorld, logLikelihood, logLikelihood, std::plus<float>(), 0);

    // master only: normalize fractional counts such that \sum_t p(t|s) = 1 \forall s
    if(learningInfo.mpiWorld->rank() == 0) {
      clock_t t50 = clock();
      NormalizeFractionalCounts();
      DeepCopy(aFractionalCounts, aParams);
      normalizationClocks += clock() - t50;
    }
    
    // update a few things on slaves
    mpi::broadcast<MultinomialParams::ConditionalMultinomialParam>(*learningInfo.mpiWorld, tFractionalCounts, 0);    
    mpi::broadcast<MultinomialParams::ConditionalMultinomialParam>(*learningInfo.mpiWorld, aFractionalCounts, 0);    
    mpi::broadcast<MultinomialParams::ConditionalMultinomialParam>(*learningInfo.mpiWorld, aParams, 0);    
    
    // persist parameters, if need be
    if(learningInfo.persistParamsAfterEachIteration && learningInfo.mpiWorld->rank() == 0) {
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

// assumptions:
// - both aParams and tFractionalCounts are properly normalized logProbs
// sample both an alignment and a translation, given src sentence and tgt length
void HmmModel::SampleATGivenS(const vector<int>& srcTokens, int tgtLength, vector<int>& tgtTokens, vector<int>& alignments, double& hmmLogProb) {

  // intialize
  int prevAlignment = INITIAL_SRC_POS;
  hmmLogProb = 0;

  // for each target position,
  for(int i = 0; i < tgtLength; i++) {
    // for debugging only
    //cerr << "at tgtPos=" << i << ", prevAlignment=" << prevAlignment << ", ";

    // sample a src position (i.e. an alignment)
    int currentAlignment;
    do {
      currentAlignment = SampleFromMultinomial(aParams[prevAlignment]);
    } while(currentAlignment >= srcTokens.size());
    alignments.push_back(currentAlignment);
    // for debugging only
    // cerr << "sample srcPos=" << currentAlignment << " which happens to be (" << srcTokens[currentAlignment] << "), ";
    
    // sample a translation
    int currentTranslation = SampleFromMultinomial(tFractionalCounts[srcTokens[currentAlignment]]);
    tgtTokens.push_back(currentTranslation);
    // for debugging only
    //cerr << "and sample translation=" << currentTranslation << endl;

    // update the sample probability according to the model
    hmmLogProb += aParams[prevAlignment][currentAlignment];
    hmmLogProb += tFractionalCounts[srcTokens[currentAlignment]][currentTranslation];

    // update prevAlignment
    if(currentAlignment != NULL_SRC_TOKEN_POS) {
      prevAlignment = currentAlignment;
    }
  }

	//cerr << "hmmLogProb=" << hmmLogProb << endl;
  assert(tgtTokens.size() == tgtLength && alignments.size() == tgtLength);
  assert(hmmLogProb >= 0);
}

// sample an alignment given a source sentence and a its translation.
void HmmModel::SampleAGivenST(const std::vector<int> &srcTokens,
		    const std::vector<int> &tgtTokens,
		    std::vector<int> &alignments,
		    double &logProb) {
  cerr << "method not implemented" << endl;
  assert(false);
}

// given the current model, align a test sentence
// assumptions: 
// - the null token has *NOT* been inserted yet
string HmmModel::AlignSent(vector<int> srcTokens, vector<int> tgtTokens) {
  
  static int sentCounter = 0;
  
  // insert the null token
  assert(srcTokens.size() > 0);
  assert(srcTokens[0] != NULL_SRC_TOKEN_ID);
  srcTokens.insert(srcTokens.begin(), 1, NULL_SRC_TOKEN_ID);
  
  // build aGivenTS
  VectorFst<LogQuadArc> tgtFst, srcFst, alignmentFst;
  CreateTgtFst(tgtTokens, tgtFst);  
  Create1stOrderSrcFst(srcTokens, srcFst);
  BuildAlignmentFst(tgtFst, srcFst, alignmentFst);
  VectorFst< LogArc > alignmentFstProbs;
  ArcMap(alignmentFst, &alignmentFstProbs, LogQuadToLogPositionMapper());
  // tropical has the path property
  VectorFst< StdArc > alignmentFstProbsWithPathProperty, bestAlignment;
  ArcMap(alignmentFstProbs, &alignmentFstProbsWithPathProperty, LogToTropicalMapper());
  ShortestPath(alignmentFstProbsWithPathProperty, &bestAlignment);
  return FstUtils::PrintAlignment(bestAlignment);
}

void HmmModel::AlignTestSet(const string &srcTestSetFilename, const string &tgtTestSetFilename, const string &outputAlignmentsFilename) {

  vector< vector<int> > srcTestSents, tgtTestSents;
  vocabEncoder.Read(srcTestSetFilename, srcTestSents);
  vocabEncoder.Read(tgtTestSetFilename, tgtTestSents);
  assert(srcTestSents.size() == tgtTestSents.size());
  
  ofstream outputAlignments(outputAlignmentsFilename.c_str(), ios::out);

  // for each parallel line
  for(unsigned sentId = 0; sentId < srcTestSents.size(); sentId++) {
    string alignmentsLine;
    vector< int > &srcTokens = srcTestSents[sentId], &tgtTokens = tgtTestSents[sentId];
    cout << "sent #" << sentId << " |srcTokens| = " << srcTokens.size() << endl;
    alignmentsLine = AlignSent(srcTokens, tgtTokens);
    outputAlignments << alignmentsLine;
  }
  outputAlignments.close();
}

void HmmModel::Align() {
  Align(outputPrefix + ".train.align");
}

void HmmModel::Align(const string &alignmentsFilename) {
  ofstream outputAlignments(alignmentsFilename.c_str(), ios::out);
  for(unsigned sentId = 0; sentId < srcSents.size(); sentId++) {
    vector<int> &srcSent = srcSents[sentId], &tgtSent = tgtSents[sentId];
    string alignmentsLine = AlignSent(srcSent, tgtSent);
    outputAlignments << alignmentsLine;
  }
  outputAlignments.close();
}
