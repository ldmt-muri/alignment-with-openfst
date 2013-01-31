#include "LatentCrfModel.h"

using namespace std;
using namespace fst;
using namespace OptAlgorithm;

// singlenton instance definition and trivial initialization
LatentCrfModel* LatentCrfModel::instance = 0;

// singleton
LatentCrfModel& LatentCrfModel::GetInstance(const string &textFilename, const string &outputPrefix, LearningInfo &learningInfo) {
  if(!LatentCrfModel::instance) {
    LatentCrfModel::instance = new LatentCrfModel(textFilename, outputPrefix, learningInfo);
  }
  return *LatentCrfModel::instance;
}

LatentCrfModel& LatentCrfModel::GetInstance() {
  if(!instance) {
    assert(false);
  }
  return *instance;
}

LatentCrfModel::~LatentCrfModel() {
  delete &lambda->srcTypes;
  delete lambda;
}

// initialize model weights to zeros
LatentCrfModel::LatentCrfModel(const string &textFilename, const string &outputPrefix, LearningInfo &learningInfo) : 
  vocabEncoder(textFilename) {

  if(learningInfo.mpiWorld->rank() == 0) {
    vocabEncoder.PersistVocab(outputPrefix + string(".vocab"));
  }
  VocabDecoder *vocabDecoder = new VocabDecoder(outputPrefix + string(".vocab"));
  lambda = new LogLinearParams(*vocabDecoder);

  // set member variables
  this->textFilename = textFilename;
  this->outputPrefix = outputPrefix;
  this->learningInfo = learningInfo;
  this->lambda->SetLearningInfo(learningInfo);

  // set constants
  this->START_OF_SENTENCE_Y_VALUE = 2;
  this->END_OF_SENTENCE_Y_VALUE = 3;

  // POS tag yDomain
  unsigned latentClasses = 4;
  this->yDomain.insert(START_OF_SENTENCE_Y_VALUE); // the conceptual yValue of word at position -1 in a sentence
  for(unsigned i = 0; i < latentClasses; i++) {
    this->yDomain.insert(START_OF_SENTENCE_Y_VALUE + i + 1);
  }

  /*
  this->yDomain.insert(4); // verb
  this->yDomain.insert(5); // adjective
  this->yDomain.insert(6); // adverb
  this->yDomain.insert(7); // pronoun
  this->yDomain.insert(8); // determiner/article
  this->yDomain.insert(9); // preposition/postposition
  this->yDomain.insert(10); // numerals
  this->yDomain.insert(11); // conjunctions
  this->yDomain.insert(12); // particles
  this->yDomain.insert(13); // punctuation marks
  this->yDomain.insert(14); // others (e.g. abbreviations, foreign words ...etc)
  this->yDomain.insert(15); // noun

  // vowel/consonant tag yDomain
  this->yDomain.insert(START_OF_SENTENCE_Y_VALUE); // the conceptual yValue of a letter at position -1 in a word
  this->yDomain.insert(END_OF_SENTENCE_Y_VALUE); // the conceptual yValue of a letter at the position after the last word
  this->yDomain.insert(4); // class4
  this->yDomain.insert(5); // class5
  //  this->yDomain.insert(6); // class6
  //  this->yDomain.insert(7); // class7
  */

  // zero is reserved for FST epsilon
  assert(this->yDomain.count(0) == 0);

  // words xDomain
  for(map<int,string>::const_iterator vocabIter = vocabEncoder.intToToken.begin();
      vocabIter != vocabEncoder.intToToken.end();
      vocabIter++) {
    if(vocabIter->second == "_unk_") {
      continue;
    }
    this->xDomain.insert(vocabIter->first);
  }
  // zero is reserved for FST epsilon
  assert(this->xDomain.count(0) == 0);

  // read and encode data
  data.clear();
  vocabEncoder.Read(textFilename, data);

  // bool vectors indicating which feature types to use
  assert(enabledFeatureTypes.size() == 0);
  // features 1-50 are reserved for wordalignment
  for(int i = 0; i <= 50; i++) {
    enabledFeatureTypes.push_back(false);
  }
  // features 51-100 are reserved for latentCrf model
  for(int i = 51; i < 100; i++) {
    enabledFeatureTypes.push_back(false);
  }
  // only enable hmm-like features -- for better comparison with HMM
  enabledFeatureTypes[51] = true;
  //  enabledFeatureTypes[54] = true;

  // initialize the theta params to unnormalized uniform
  nLogTheta.clear();
  for(set<int>::const_iterator yDomainIter = yDomain.begin(); yDomainIter != yDomain.end(); yDomainIter++) {
    for(set<int>::const_iterator zDomainIter = xDomain.begin(); zDomainIter != xDomain.end(); zDomainIter++) {
      //      nLogTheta[*yDomainIter][*zDomainIter] = 1;
      nLogTheta[*yDomainIter][*zDomainIter] = abs(gaussianSampler.Draw());
    }
  }
  // REMOVE ME  4 = consonant, 5 = vowel (dyer's tiny letters)
  /*  nLogTheta[4][3] = 10;
  nLogTheta[4][5] = 10;
  nLogTheta[4][7] = 10;
  nLogTheta[4][8] = 10;
  nLogTheta[4][10] = 10;
  nLogTheta[4][11] = 10;
  nLogTheta[4][12] = 10;
  nLogTheta[4][13] = 10;
  nLogTheta[4][15] = 10;
  nLogTheta[4][16] = 10;
  nLogTheta[4][17] = 10;
  nLogTheta[5][4] = 10;
  nLogTheta[5][6] = 10;
  nLogTheta[5][9] = 10;
  nLogTheta[5][14] = 10;
  // good initialization for wammar's tiny letters with four classes and four unique letters
  nLogTheta[4][vocabEncoder.Encode("a")] = 0.01;
  nLogTheta[5][vocabEncoder.Encode("b")] = 0.01;
  nLogTheta[6][vocabEncoder.Encode("c")] = 0.01;
  nLogTheta[7][vocabEncoder.Encode("d")] = 0.01;
  */

  // then normalize
  MultinomialParams::NormalizeParams(nLogTheta);
  //  MultinomialParams::PrintParams(nLogTheta);
  if(learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
    MultinomialParams::PrintParams(nLogTheta, vocabEncoder);
  }

  // lambdas are initialized to all zeros
  assert(lambda->GetParamsCount() == 0);

  // hand-crafted weights for constrained features
  REWARD_FOR_CONSTRAINED_FEATURES = 10.0;
  PENALTY_FOR_CONSTRAINED_FEATURES = -10.0;
}

// compute the partition function Z_\lambda(x)
// assumptions:
// - fst and betas are populated using BuildLambdaFst()
double LatentCrfModel::ComputeNLogZ_lambda(const VectorFst<LogArc> &fst, const vector<fst::LogWeight> &betas) {
  return betas[fst.Start()].Value();
}

// compute the partition function Z_\lambda(x)
double LatentCrfModel::ComputeNLogZ_lambda(const vector<int> &x) {
  VectorFst<LogArc> fst;
  vector<fst::LogWeight> alphas;
  vector<fst::LogWeight> betas;
  BuildLambdaFst(x, fst, alphas, betas);
  return ComputeNLogZ_lambda(fst, betas);
}

// build an FST to compute Z(x) = \sum_y \prod_i \exp \lambda h(y_i, y_{i-1}, x, i)
void LatentCrfModel::BuildLambdaFst(const vector<int> &x, VectorFst<LogArc> &fst, vector<fst::LogWeight> &alphas, vector<fst::LogWeight> &betas) {
  clock_t timestamp = clock();

  // arcs represent a particular choice of y_i at time step i
  // arc weights are -\lambda h(y_i, y_{i-1}, x, i)
  assert(fst.NumStates() == 0);
  int startState = fst.AddState();
  fst.SetStart(startState);

  // map values of y_{i-1} and y_i to fst states
  map<int, int> yIM1ToState, yIToState;
  assert(yIM1ToState.size() == 0);
  assert(yIToState.size() == 0);
  yIM1ToState[START_OF_SENTENCE_Y_VALUE] = startState;

  // for each timestep
  for(int i = 0; i < x.size(); i++){

    // timestep i hasn't reached any states yet
    yIToState.clear();
    // from each state reached in the previous timestep
    for(map<int, int>::const_iterator prevStateIter = yIM1ToState.begin();
	prevStateIter != yIM1ToState.end();
	prevStateIter++) {

      int fromState = prevStateIter->second;
      int yIM1 = prevStateIter->first;
      // to each possible value of y_i
      for(set<int>::const_iterator yDomainIter = yDomain.begin();
	  yDomainIter != yDomain.end();
	  yDomainIter++) {

	int yI = *yDomainIter;
	
	// skip special classes
	if(yI == START_OF_SENTENCE_Y_VALUE || yI == END_OF_SENTENCE_Y_VALUE) {
	  continue;
	}

	// compute h(y_i, y_{i-1}, x, i)
	map<string, double> h;
	lambda->FireFeatures(yI, yIM1, x, i, enabledFeatureTypes, h);
	// compute the weight of this transition:
	// \lambda h(y_i, y_{i-1}, x, i), and multiply by -1 to be consistent with the -log probability representation
	double nLambdaH = -1.0 * lambda->DotProduct(h);
	// determine whether to add a new state or reuse an existing state which also represent label y_i and timestep i
	int toState;
	if(yIToState.count(yI) == 0) {
	  toState = fst.AddState();
	  yIToState[yI] = toState;
	  // is it a final state?
	  if(i == x.size() - 1) {
	    fst.SetFinal(toState, LogWeight::One());
	  }
	} else {
	  toState = yIToState[yI];
	}
	// now add the arc
	fst.AddArc(fromState, fst::LogArc(yIM1, yI, nLambdaH, toState));
      } 
   }
    // now, that all states reached in step i have already been created, yIM1ToState has become irrelevant
    yIM1ToState = yIToState;
  }

  // now compute potentials
  assert(alphas.size() == 0);
  ShortestDistance(fst, &alphas, false);
  assert(betas.size() == 0);
  ShortestDistance(fst, &betas, true);

  if(learningInfo.debugLevel == DebugLevel::SENTENCE) {
    cerr << " BuildLambdaFst() for this sentence took " << (float) (clock() - timestamp) / CLOCKS_PER_SEC << " sec. " << endl;
  }
}

// assumptions: 
// - fst is populated using BuildLambdaFst()
// - FXk is cleared
void LatentCrfModel::ComputeF(const vector<int> &x,
			      const VectorFst<LogArc> &fst,
			      const vector<fst::LogWeight> &alphas, const vector<fst::LogWeight> &betas,
			      FastSparseVector<LogVal<double> > &FXk) {
  clock_t timestamp = clock();
  
  assert(FXk.size() == 0);
  assert(fst.NumStates() > 0);
  
  // schedule for visiting states such that we know the timestep for each arc
  set<int> iStates, iP1States;
  iStates.insert(fst.Start());

  // for each timestep
  for(int i = 0; i < x.size(); i++) {
    int xI = x[i];
    
    //    cerr << "i = " << i << " out of " << x.size() << endl;

    // from each state at timestep i
    for(set<int>::const_iterator iStatesIter = iStates.begin(); 
	iStatesIter != iStates.end(); 
	iStatesIter++) {
      int fromState = *iStatesIter;

      //      cerr << "  from state# " << fromState << endl;

      // for each arc leaving this state
      for(ArcIterator< VectorFst<LogArc> > aiter(fst, fromState); !aiter.Done(); aiter.Next()) {
	LogArc arc = aiter.Value();
	int yIM1 = arc.ilabel;
	int yI = arc.olabel;
	double arcWeight = arc.weight.Value();
	int toState = arc.nextstate;

	//	cerr << "    to state# " << toState << " yIM1=" << yIM1 << " yI=" << yI << " weight=" << arcWeight << endl;

	// compute marginal weight of passing on this arc
	double nLogMarginal = alphas[fromState].Value() + betas[toState].Value() + arcWeight;

	// for each feature that fires on this arc
	FastSparseVector<double> h;
	lambda->FireFeatures(yI, yIM1, x, i, enabledFeatureTypes, h);
	for(FastSparseVector<double>::iterator h_k = h.begin(); h_k != h.end(); ++h_k) {

	  //	  cerr << "      featureId=" << h_k->first << " value=" << h_k->second << endl;

	  // add the arc's h_k feature value weighted by the marginal weight of passing through this arc
	  if(FXk.find(h_k->first) == FXk.end()) {
	    FXk[h_k->first] = LogVal<double>(0.0);
	  }
	  //cerr << FXk[h_k->first];
	  FXk[h_k->first] += LogVal<double>(-1.0 * nLogMarginal, init_lnx()) * LogVal<double>(h_k->second);
	  //cerr << " => " << FXk[h_k->first] << endl;
	  /*
	  if(isinf(FXk[h_k->first])) {
	    cerr << "ERROR: FXk[" << h_k->first << "] = " << FXk[h_k->first] << ", nLogMarginal = " << nLogMarginal;
	    cerr << ", MultinomialParams::nExp(nLogMarginal) = " << MultinomialParams::nExp(nLogMarginal) << endl;
	  }
	  */
	}

	// prepare the schedule for visiting states in the next timestep
	iP1States.insert(toState);
      } 
    }

    // prepare for next timestep
    iStates = iP1States;
    iP1States.clear();
  }  

  if(learningInfo.debugLevel == DebugLevel::SENTENCE) {
    cerr << "ComputeF() for this sentence took " << (float) (clock() - timestamp) / CLOCKS_PER_SEC << " sec." << endl;
  }
}			   

// assumptions: 
// - fst is populated using BuildThetaLambdaFst()
// - DXZk is cleared
void LatentCrfModel::ComputeD(const vector<int> &x, const vector<int> &z, 
			      const VectorFst<LogArc> &fst,
			      const vector<fst::LogWeight> &alphas, const vector<fst::LogWeight> &betas,
			      FastSparseVector<double> &DXZk) {
  //  cerr << "ComputeD(){";
  clock_t timestamp = clock();

  // enforce assumptions
  assert(DXZk.size() == 0);

  // schedule for visiting states such that we know the timestep for each arc
  set<int> iStates, iP1States;
  iStates.insert(fst.Start());

  // for each timestep
  for(int i = 0; i < x.size(); i++) {
    int xI = x[i];
    int zI = z[i];
    
    // from each state at timestep i
    for(set<int>::const_iterator iStatesIter = iStates.begin(); 
	iStatesIter != iStates.end(); 
	iStatesIter++) {
      int fromState = *iStatesIter;

      // for each arc leaving this state
      for(ArcIterator< VectorFst<LogArc> > aiter(fst, fromState); !aiter.Done(); aiter.Next()) {
	LogArc arc = aiter.Value();
	int yIM1 = arc.ilabel;
	int yI = arc.olabel;
	double arcWeight = arc.weight.Value();
	int toState = arc.nextstate;

	// compute marginal weight of passing on this arc
	double nLogMarginal = alphas[fromState].Value() + betas[toState].Value() + arcWeight;

	// for each feature that fires on this arc
	FastSparseVector<double> h;
	lambda->FireFeatures(yI, yIM1, x, i, enabledFeatureTypes, h);
	for(FastSparseVector<double>::iterator h_k = h.begin(); h_k != h.end(); ++h_k) {

	  // add the arc's h_k feature value weighted by the marginal weight of passing through this arc
	  if(DXZk.find(h_k->first) == DXZk.end()) {
	    DXZk[h_k->first] = 0;
	  }
	  DXZk[h_k->first] += MultinomialParams::nExp(nLogMarginal) * h_k->second;
	}

	// prepare the schedule for visiting states in the next timestep
	iP1States.insert(toState);
      } 
    }

    // prepare for next timestep
    iStates = iP1States;
    iP1States.clear();
  }  

  if(learningInfo.debugLevel == DebugLevel::SENTENCE) {
    cerr << "ComputeD() for this sentence took " << (float) (clock() - timestamp) / CLOCKS_PER_SEC << " sec." << endl;
  }

  //  cerr << "}\n";
}

// assumptions:
// - fst, betas are populated using BuildThetaLambdaFst()
double LatentCrfModel::ComputeNLogC(const VectorFst<LogArc> &fst,
				 const vector<fst::LogWeight> &betas) {
  double nLogC = betas[fst.Start()].Value();
  return nLogC;
}

// compute B(x,z) which can be indexed as: BXZ[y^*][z^*] to give B(x, z, z^*, y^*)
// assumptions: 
// - BXZ is cleared
// - fst, alphas, and betas are populated using BuildThetaLambdaFst
void LatentCrfModel::ComputeB(const vector<int> &x, const vector<int> &z, 
			   const VectorFst<LogArc> &fst, 
			   const vector<fst::LogWeight> &alphas, const vector<fst::LogWeight> &betas, 
			   map< int, map< int, double > > &BXZ) {
  // \sum_y [ \prod_i \theta_{z_i\mid y_i} e^{\lambda h(y_i, y_{i-1}, x, i)} ] \sum_i \delta_{y_i=y^*,z_i=z^*}
  //  cerr << "ComputeB(){"<<endl;
  assert(BXZ.size() == 0);

  // debug
  if(learningInfo.debugLevel == DebugLevel::REDICULOUS) {
    cerr << "thetas are: " << endl;
    MultinomialParams::PrintParams(nLogTheta);
    cerr << "thetas (with string observables): " << endl;
    MultinomialParams::PrintParams(nLogTheta, vocabEncoder);
    cerr << "lambdas are: " << endl;
    lambda->PrintParams();
    cerr << "ComputeB() is called with the following params: " << endl;
    cerr << "x = ";
    for(unsigned i = 0; i < x.size(); i++) {
      cerr << x[i] << " ";
    }
    cerr << endl << "z = ";
    for(unsigned i = 0; i < z.size(); i++) {
      cerr << z[i] << " ";
    }
    cerr << endl << "thetaLambdaFst = " << endl;
    cerr << FstUtils::PrintFstSummary(fst) << endl;
    cerr << "alphas = ";
    for(unsigned i = 0; i < alphas.size(); i++) {
      cerr << i << ":" << alphas[i].Value() << "(" << MultinomialParams::nExp(alphas[i].Value()) << ") ";
    }
    cerr << endl << "betas = ";
    for(unsigned i = 0; i < betas.size(); i++) {
      cerr << i << ":" << betas[i].Value() << "(" << MultinomialParams::nExp(betas[i].Value()) << ") ";
    }
    cerr << endl << endl;
  }

  // schedule for visiting states such that we know the timestep for each arc
  set<int> iStates, iP1States;
  iStates.insert(fst.Start());

  // for each timestep
  for(int i = 0; i < x.size(); i++) {
    int xI = x[i];
    int zI = z[i];
    
    // from each state at timestep i
    for(set<int>::const_iterator iStatesIter = iStates.begin(); 
	iStatesIter != iStates.end(); 
	iStatesIter++) {
      int fromState = *iStatesIter;

      // for each arc leaving this state
      for(ArcIterator< VectorFst<LogArc> > aiter(fst, fromState); !aiter.Done(); aiter.Next()) {
	LogArc arc = aiter.Value();
	int yIM1 = arc.ilabel;
	int yI = arc.olabel;
	double arcWeight = arc.weight.Value();
	int toState = arc.nextstate;

	// compute marginal weight of passing on this arc
	double nLogMarginal = alphas[fromState].Value() + betas[toState].Value() + arcWeight;

	// update the corresponding B value
	if(BXZ.count(yI) == 0 || BXZ[yI].count(zI) == 0) {
	  BXZ[yI][zI] = 0;
	}
	//	cerr << BXZ[yI][zI];
	BXZ[yI][zI] += MultinomialParams::nExp(nLogMarginal);
	//	cerr << " => " << BXZ[yI][zI] << endl;

	// prepare the schedule for visiting states in the next timestep
	iP1States.insert(toState);
      } 
    }

    // prepare for next timestep
    iStates = iP1States;
    iP1States.clear();
  }
  
  //  cerr << "}\n";
}

/*
// build an FST with log-pair weights. the first component pathsums to:
// -log \sum_y [ \prod_i \theta_{z_i\mid y_i} e^{\lambda h(y_i, y_{i-1}, x, i)} ]
// while the other component pathsums to:
// -log \sum_y [ \prod_i e^{\lambda h(y_i, y_{i-1}, x, i)} ]
void LatentCrfModel::BuildThetaLambdaAndLambdaFst(const vector<int> &x, const vector<int> &z, 
						  VectorFst<LogPairArc> &fst, 
						  vector<LogPairWeight> &alphas, vector<fst::LogPairWeight> &betas) {
  clock_t timestamp = clock();

  // arcs represent a particular choice of y_i at time step i
  // arc weights are -log \theta_{z_i|y_i} - \lambda h(y_i, y_{i-1}, x, i)
  assert(fst.NumStates() == 0);
  int startState = fst.AddState();
  fst.SetStart(startState);
  
  // map values of y_{i-1} and y_i to fst states
  map<int, int> yIM1ToState, yIToState;
  assert(yIM1ToState.size() == 0);
  assert(yIToState.size() == 0);

  yIM1ToState[START_OF_SENTENCE_Y_VALUE] = startState;

  // for each timestep
  for(int i = 0; i < x.size(); i++){

    // timestep i hasn't reached any states yet
    yIToState.clear();
    // from each state reached in the previous timestep
    for(map<int, int>::const_iterator prevStateIter = yIM1ToState.begin();
	prevStateIter != yIM1ToState.end();
	prevStateIter++) {

      int fromState = prevStateIter->second;
      int yIM1 = prevStateIter->first;
      // to each possible value of y_i
      for(set<int>::const_iterator yDomainIter = yDomain.begin();
	  yDomainIter != yDomain.end();
	  yDomainIter++) {

	int yI = *yDomainIter;

	// skip special classes
	if(yI == START_OF_SENTENCE_Y_VALUE || yI == END_OF_SENTENCE_Y_VALUE) {
	  continue;
	}

	// compute h(y_i, y_{i-1}, x, i)
	map<string, double> h;
	lambda->FireFeatures(yI, yIM1, x, i, enabledFeatureTypes, h);

	// prepare -log \theta_{z_i|y_i}
	int zI = z[i];
	double nLogTheta_zI_yI = this->nLogTheta[yI][zI];

	// compute the weight of this transition: \lambda h(y_i, y_{i-1}, x, i), and multiply by -1 to be consistent with the -log probability representatio
	double nLambdaH = -1.0 * lambda->DotProduct(h);
	double weight1 = nLambdaH + nLogTheta_zI_yI;
	double weight2 = nLambdaH;

	// determine whether to add a new state or reuse an existing state which also represent label y_i and timestep i
	int toState;	
	if(yIToState.count(yI) == 0) {
	  toState = fst.AddState();
	  yIToState[yI] = toState;
	  // is it a final state?
	  if(i == x.size() - 1) {
	    fst.SetFinal(toState, LogWeight::One());
	  }
	} else {
	  toState = yIToState[yI];
	}
	// now add the arc
	fst.AddArc(fromState, fst::LogArc(yIM1, yI, EncodePair(weight1, weight2), toState));	
      }
    }
    // now, that all states reached in step i have already been created, yIM1ToState has become irrelevant
    yIM1ToState = yIToState;
  }

  // compute forward/backward state potentials
  assert(alphas.size() == 0);
  assert(betas.size() == 0);
  ShortestDistance(fst, &alphas, false);
  ShortestDistance(fst, &betas, true);

  if(learningInfo.debugLevel == DebugLevel::SENTENCE) {
    cerr << " BuildThetaLambdaFst() for this sentence took " << (float) (clock() - timestamp) / CLOCKS_PER_SEC << " sec. " << endl;
  }
}
*/

// build an FST which path sums to 
// -log \sum_y [ \prod_i \theta_{z_i\mid y_i} e^{\lambda h(y_i, y_{i-1}, x, i)} ]
void LatentCrfModel::BuildThetaLambdaFst(const vector<int> &x, const vector<int> &z, VectorFst<LogArc> &fst, vector<fst::LogWeight> &alphas, vector<fst::LogWeight> &betas) {
  clock_t timestamp = clock();

  // arcs represent a particular choice of y_i at time step i
  // arc weights are -log \theta_{z_i|y_i} - \lambda h(y_i, y_{i-1}, x, i)
  assert(fst.NumStates() == 0);
  int startState = fst.AddState();
  fst.SetStart(startState);
  
  // map values of y_{i-1} and y_i to fst states
  map<int, int> yIM1ToState, yIToState;
  assert(yIM1ToState.size() == 0);
  assert(yIToState.size() == 0);

  yIM1ToState[START_OF_SENTENCE_Y_VALUE] = startState;

  // for each timestep
  for(int i = 0; i < x.size(); i++){

    // timestep i hasn't reached any states yet
    yIToState.clear();
    // from each state reached in the previous timestep
    for(map<int, int>::const_iterator prevStateIter = yIM1ToState.begin();
	prevStateIter != yIM1ToState.end();
	prevStateIter++) {

      int fromState = prevStateIter->second;
      int yIM1 = prevStateIter->first;
      // to each possible value of y_i
      for(set<int>::const_iterator yDomainIter = yDomain.begin();
	  yDomainIter != yDomain.end();
	  yDomainIter++) {

	int yI = *yDomainIter;

	// skip special classes
	if(yI == START_OF_SENTENCE_Y_VALUE || yI == END_OF_SENTENCE_Y_VALUE) {
	  continue;
	}

	// compute h(y_i, y_{i-1}, x, i)
	map<string, double> h;
	lambda->FireFeatures(yI, yIM1, x, i, enabledFeatureTypes, h);

	// prepare -log \theta_{z_i|y_i}
	int zI = z[i];
	double nLogTheta_zI_yI = this->nLogTheta[yI][zI];

	// compute the weight of this transition: \lambda h(y_i, y_{i-1}, x, i), and multiply by -1 to be consistent with the -log probability representatio
	double nLambdaH = -1.0 * lambda->DotProduct(h);
	double weight = nLambdaH + nLogTheta_zI_yI;

	// determine whether to add a new state or reuse an existing state which also represent label y_i and timestep i
	int toState;	
	if(yIToState.count(yI) == 0) {
	  toState = fst.AddState();
	  yIToState[yI] = toState;
	  // is it a final state?
	  if(i == x.size() - 1) {
	    fst.SetFinal(toState, LogWeight::One());
	  }
	} else {
	  toState = yIToState[yI];
	}
	// now add the arc
	fst.AddArc(fromState, fst::LogArc(yIM1, yI, weight, toState));	
      }
    }
    // now, that all states reached in step i have already been created, yIM1ToState has become irrelevant
    yIM1ToState = yIToState;
  }

  // compute forward/backward state potentials
  assert(alphas.size() == 0);
  assert(betas.size() == 0);
  ShortestDistance(fst, &alphas, false);
  ShortestDistance(fst, &betas, true);

  if(learningInfo.debugLevel == DebugLevel::SENTENCE) {
    cerr << " BuildThetaLambdaFst() for this sentence took " << (float) (clock() - timestamp) / CLOCKS_PER_SEC << " sec. " << endl;
  }
}

// compute p(y, z | x) = \frac{\prod_i \theta_{z_i|y_i} \exp \lambda h(y_i, y_{i-1}, x, i)}{Z_\lambda(x)}
double LatentCrfModel::ComputeNLogPrYZGivenX(vector<int>& x, vector<int>& y, vector<int>& z) {
  assert(x.size() == y.size());
  assert(x.size() == z.size());

  // initialize prob = 1.0
  double result = 0;

  // divide by Z_\lambda(x)
  result -= ComputeNLogZ_lambda(x);

  for(int i = 0; i < x.size(); i++) {

    // multiply \theta_{z_i|y_i} (which is already stored using in its -log value)
    result += nLogTheta[y[i]][z[i]];

    // multiply \exp \lambda h(y_i, y_{i-1}, x, i)
    //  compute h(y_i, y_{i-1}, x, i)
    map<string, double> h;
    lambda->FireFeatures(y[i], y[i-1], x, i, enabledFeatureTypes, h);
    //  compute \lambda h(y_i, y_{i-1}, x, i) , multiply by -1 to be consistent with the -log probability representation
    double nlambdaH = -1 * lambda->DotProduct(h);
    result += nlambdaH;
  }

  return result;
}

// copute p(y | x, z) = \frac  {\prod_i \theta_{z_i|y_i} \exp \lambda h(y_i, y_{i-1}, x, i)} 
//                             -------------------------------------------
//                             {\sum_y' \prod_i \theta_{z_i|y'_i} \exp \lambda h(y'_i, y'_{i-1}, x, i)}
double LatentCrfModel::ComputeNLogPrYGivenXZ(vector<int> &x, vector<int> &y, vector<int> &z) {
  assert(x.size() == y.size());
  assert(x.size() == z.size());

  double result = 0;

  // multiply the numerator
  for(int i = 0; i < x.size(); i++) {

    // multiply \theta_{z_i|y_i} (which is already stored in its -log value)
    result += nLogTheta[y[i]][z[i]];

    // multiply \exp \lambda h(y_i, y_{i-1}, x, i)
    //  compute h(y_i, y_{i-1}, x, i)
    map<string, double> h;
    lambda->FireFeatures(y[i], y[i-1], x, i, enabledFeatureTypes, h);
    //  compute \lambda h(y_i, y_{i-1}, x, i)
    double lambdaH = -1 * lambda->DotProduct(h);
    //  now multiply \exp \lambda h(y_i, y_{i-1}, x, i)
    result += lambdaH;
  }

  // compute the denominator using an FST
  //  denominator = \sum_y' \prod_i \theta_{z_i|y'_i} \exp \lambda h(y'_i, y'_{i-1}, x, i)
  //  arcs represent a particular choice of y_i at time step i
  //  arc weights are \lambda h(y_i, y_{i-1}, x, i) 
  VectorFst<LogArc> fst;
  assert(fst.NumStates() == 0);
  int startState = fst.AddState();
  fst.SetStart(startState);
  
  //  map values of y_{i-1} and y_i to fst states
  map<int, int> yIM1ToState, yIToState;
  assert(yIM1ToState.size() == 0);
  assert(yIToState.size() == 0);
  yIM1ToState[START_OF_SENTENCE_Y_VALUE] = startState;

  //  for each timestep
  for(int i = 0; i < x.size(); i++){

    // timestep i hasn't reached any states yet
    yIToState.clear();
    // from each state reached in the previous timestep
    for(map<int, int>::const_iterator prevStateIter = yIM1ToState.begin();
	prevStateIter != yIM1ToState.end();
	prevStateIter++) {

      int fromState = prevStateIter->second;
      int yIM1 = prevStateIter->first;
      // to each possible value of y_i
      for(set<int>::const_iterator yDomainIter = this->yDomain.begin();
	  yDomainIter != yDomain.end();
	  yDomainIter++) {

	int yI = *yDomainIter;

	// skip special classes
	if(yI == START_OF_SENTENCE_Y_VALUE || yI == END_OF_SENTENCE_Y_VALUE) {
	  continue;
	}

	// compute h(y_i, y_{i-1}, x, i)
	map<string, double> h;
	lambda->FireFeatures(yI, yIM1, x, i, enabledFeatureTypes, h);
	// \lambda h(...,i)
	double lambdaH = -1.0 * lambda->DotProduct(h);
	// compute the weight of this transition: -log p_\theta(z_i|y_i) -log \exp \lambda h(y_i, y_{i-1}, x, i)
	// note: parameters theta[y_{i-1}][y_i] is already in the -log representation
	double weight = lambdaH + nLogTheta[yI][z[i]];
	// determine whether to add a new state or reuse an existing state which also represent label y_i and timestep i
	int toState;	
	if(yIToState.count(yI) == 0) {
	  toState = fst.AddState();
	  yIToState[yI] = toState;
	  // is it a final state?
	  if(i == x.size() - 1) {
	    fst.SetFinal(toState, LogWeight::One());
	  }
	} else {
	  toState = yIToState[yI];
	}
	// now add the arc
	fst.AddArc(fromState, fst::LogArc(yIM1, yI, weight, toState));	
      }
    }
    // now, that all states reached in step i have already been created, yIM1ToState has become irrelevant
    yIM1ToState = yIToState;
  }

  //  now compute the path sum, i.e. -\log [ \sum_y' \prod_i \theta_{z_i|y'_i} \exp \lambda h(y'_i, y'_{i-1}, x, i) ]
  vector<fst::LogWeight> distancesToFinal;
  ShortestDistance(fst, &distancesToFinal, true);

  //  finally, divide by the denominator
  double denominator = distancesToFinal[startState].Value();
  result -= denominator;

  // return p(y | x, z)
  return result;
}

void LatentCrfModel::Train() {
  switch(learningInfo.optimizationMethod.algorithm) {
  case BLOCK_COORD_DESCENT:
    BlockCoordinateDescent();
    break;
    /*  case EXPECTATION_MAXIMIZATION:
    ExpectationMaximization();
    break;*/
  default:
    assert(false);
    break;
  }
}

// can be optimized if need be (hint: the Evaluate callback function computes lambda derivatives which we don't need here)
double LatentCrfModel::ComputeCorpusNloglikelihood() {
  int index = -1;
  double gradient[lambda->GetParamsCount()];
  return EvaluateNLogLikelihoodDerivativeWRTLambda(&index, lambda->GetParamWeightsArray(), gradient, lambda->GetParamsCount(), 0);
}

// to interface with the simulated annealing library at http://www.taygeta.com/annealing/simanneal.html
float LatentCrfModel::EvaluateNLogLikelihood(float *lambdasArray) {
  // singleton
  LatentCrfModel &model = LatentCrfModel::GetInstance();
  // unconstrained lambda parameters count
  unsigned lambdasCount = model.lambda->GetParamsCount() - model.countOfConstrainedLambdaParameters;
  // which sentences to work on?
  static int fromSentId = 0;
  int sentsCount = model.data.size();
  if(fromSentId >= sentsCount) {
    fromSentId = 0;
  }
  double *dblLambdasArray = model.lambda->GetParamWeightsArray();
  for(unsigned i = 0; i < lambdasCount; i++) {
    dblLambdasArray[i] = (double)lambdasArray[i];
  }
  // next time, work on different sentences
  fromSentId += model.learningInfo.optimizationMethod.subOptMethod->miniBatchSize;
  // call the other function ;-)
  void *ptrFromSentId = &fromSentId;
  double dummy[lambdasCount];
  float objective = (float)EvaluateNLogLikelihoodDerivativeWRTLambda(ptrFromSentId, dblLambdasArray, dummy, lambdasCount, 1.0);
  cerr << "objective = " << objective << endl;
  return objective;
}

FastSparseVector<double> LatentCrfModel::AccumulateDerivatives(const FastSparseVector<double> &v1, const FastSparseVector<double> &v2) {
  FastSparseVector<double> vTotal(v1);
  for(FastSparseVector<double>::const_iterator v2Iter = v2.begin(); v2Iter != v2.end(); ++v2Iter) {
    vTotal[v2Iter->first] += v2Iter->second;
  }
  return vTotal;
}

// a call back function that computes the gradient and the nloglikelihood function for the lbfgs minimizer
double LatentCrfModel::EvaluateNLogLikelihoodDerivativeWRTLambda(void *ptrFromSentId,
								 const double *lambdasArray,
								 double *gradient,
								 const int lambdasCount,
								 const double step) {
  //  cerr << "EvaluateNLogLikelihoodDerivativeWRTLambda(){" << endl;
  clock_t timestamp = clock();

  LatentCrfModel &model = LatentCrfModel::GetInstance();

  // important note: the parameters array manipulated by liblbfgs is the same one used in lambda. so, the new weights are already in effect

  // debug
  if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
    cerr << "lbfgs suggests the following lambda parameter weights for process #" << model.learningInfo.mpiWorld->rank() << endl;
    model.lambda->PrintParams();
  }

  // for each sentence in this mini batch, aggregate the nloglikelihood and its derivatives across sentences
  double nlogLikelihood = 0;
  FastSparseVector<double> derivativeWRTLambdaSparseVector;
  int index = *((int*)ptrFromSentId), from, to;
  if(index == -1) {
    from = 0;
    to = model.data.size();
  } else {
    from = index;
    to = min((int)model.data.size(), from + model.learningInfo.optimizationMethod.subOptMethod->miniBatchSize);
  }
  for(int sentId = from; sentId < to; sentId++) {
    // sentId is assigned to the process with rank = sentId % world.size()
    if(sentId % model.learningInfo.mpiWorld->size() != model.learningInfo.mpiWorld->rank()) {
      continue;
    }
    clock_t timestamp2 = clock();
    /*
    // TODO: work in progress... optimization as explained in issue #55 https://github.com/ldmt-muri/alignment-with-openfst/issues/55
    // - we want to replace both ComputeD() and ComputeF() with ComputeDAndF(). 
    // - we also need to modify computeNLogC() and computeNLogZ()
    assert(false);
    // build one FST that encodes both ThetaLambdaFst and LambdaFst using LogPair weights
    VectorFst<LogPairArc> thetaLambdaAndLambdaFst;
    vector<LogPairWeight> thetaLambdaAndLambdaAlphas, thetaLambdaAndLambdaBetas;
    model.BuildThetaLambdaAndLambdaFst(model.data[sentId], model.data[sentId], thetaLambdaAndLambdaFst, 
				       thetaLambdaAndLambdaAlphas, thetaLambdaAndLambdaBetas);
    */
    // build the FSTs
    VectorFst<LogArc> thetaLambdaFst, lambdaFst;
    vector<fst::LogWeight> thetaLambdaAlphas, lambdaAlphas, thetaLambdaBetas, lambdaBetas;
    model.BuildThetaLambdaFst(model.data[sentId], model.data[sentId], thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas);
    model.BuildLambdaFst(model.data[sentId], lambdaFst, lambdaAlphas, lambdaBetas);
    // compute the D map for this sentence
    FastSparseVector<double> DSparseVector;
    model.ComputeD(model.data[sentId], model.data[sentId], thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas, DSparseVector);
    // compute the C value for this sentence
    double nLogC = model.ComputeNLogC(thetaLambdaFst, thetaLambdaBetas);
    if(isnan(nLogC) || isinf(nLogC)) {
      if(model.learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
	cerr << "ERROR: nLogC = " << nLogC << ". my mistake. will halt!" << endl;
	cerr << "thetaLambdaFst summary:" << endl;
	cerr << FstUtils::PrintFstSummary(thetaLambdaFst);
      }
      assert(false);
    } 
    // update the loglikelihood
    nlogLikelihood += nLogC;
    // add D/C to the gradient
    for(FastSparseVector<double>::iterator dIter = DSparseVector.begin(); dIter != DSparseVector.end(); ++dIter) {
      double d = dIter->second;
      double nLogd = MultinomialParams::nLog(d);
      double dOverC = MultinomialParams::nExp(nLogd - nLogC);
      if(isnan(dOverC) || isinf(dOverC)) {
	if(model.learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
	  cerr << "ERROR: dOverC = " << dOverC << ", nLogd = " << nLogd << ", d = " << d << ". my mistake. will halt!" << endl;
	}
        assert(false);
      }
      derivativeWRTLambdaSparseVector[dIter->first] -= dOverC;
    }
    // compute the F map fro this sentence
    FastSparseVector<LogVal<double> > FSparseVector;
    model.ComputeF(model.data[sentId], lambdaFst, lambdaAlphas, lambdaBetas, FSparseVector);
    // compute the Z value for this sentence
    double nLogZ = model.ComputeNLogZ_lambda(lambdaFst, lambdaBetas);
    // update the log likelihood
    if(isnan(nLogZ) || isinf(nLogZ)) {
      if(model.learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
	cerr << "ERROR: nLogZ = " << nLogZ << ". my mistake. will halt!" << endl;
      }
      assert(false);
    } 
    nlogLikelihood -= nLogZ;
    //      cerr << "nloglikelihood -= " << nLogZ << ", |x| = " << data[sentId].size() << endl;
    // subtract F/Z from the gradient
    for(FastSparseVector<LogVal<double> >::iterator fIter = FSparseVector.begin(); fIter != FSparseVector.end(); ++fIter) {
      double nLogf = fIter->second.s_? fIter->second.v_ : -fIter->second.v_; // multiply the inner logF representation by -1.
      double fOverZ = MultinomialParams::nExp(nLogf - nLogZ);
      if(isnan(fOverZ) || isinf(fOverZ)) {
	if(model.learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
	  cerr << "ERROR: fOverZ = " << nLogZ << ", nLogf = " << nLogf << ". my mistake. will halt!" << endl;
	}
	assert(false);
      }
      derivativeWRTLambdaSparseVector[fIter->first] += fOverZ;
      if(isnan(derivativeWRTLambdaSparseVector[fIter->first]) || isinf(derivativeWRTLambdaSparseVector[fIter->first])) {
	cerr << "ERROR: fOverZ = " << nLogZ << ", nLogf = " << nLogf << ". my mistake. will halt!" << endl;
	assert(false);
      }
    }
    if(model.learningInfo.debugLevel >= DebugLevel::SENTENCE) {
      cerr << ".";
      cerr << "EvaluateNLogLikelihoodDerivativeWRTLambda() for this sentence took " << (float) (clock() - timestamp2) / CLOCKS_PER_SEC << " sec." << endl;
    }
  }

  // debug
  assert(lambdasCount == model.lambda->GetParamsCount() - model.countOfConstrainedLambdaParameters);

  // write the gradient in the array 'gradient' (which is pre-allocated by the lbfgs library)
  // init gradient to zero
  for(int displacedIndex = 0; displacedIndex < lambdasCount; displacedIndex++) {
    gradient[displacedIndex] = 0;
  }
  // for each active feature in this mini batch
  for(FastSparseVector<double>::iterator derivativeIter = derivativeWRTLambdaSparseVector.begin(); 
      derivativeIter != derivativeWRTLambdaSparseVector.end(); 
      ++derivativeIter) {
    // skip constrained features
    if(derivativeIter->first < model.countOfConstrainedLambdaParameters) {
      continue;
    }
    // set active unconstrained feature's gradient
    gradient[derivativeIter->first - model.countOfConstrainedLambdaParameters] = derivativeIter->second;
  }

  // accumulate nloglikelihood from all processes
  mpi::all_reduce<double>(*model.learningInfo.mpiWorld, nlogLikelihood, nlogLikelihood, std::plus<double>());

  // accumulate the gradient vectors from all processes
  try {
    vector<double> gradientVector(gradient, gradient + lambdasCount);
    mpi::all_reduce<vector<double> >(*model.learningInfo.mpiWorld, gradientVector, gradientVector, LatentCrfModel::AggregateVectors);
    assert(gradientVector.size() == lambdasCount);
    for(int i = 0; i < gradientVector.size(); i++) {
      gradient[i] = gradientVector[i];
    }
  } catch (boost::exception &e) {
    cerr << "all_reduce diagnostic info: " << boost::diagnostic_information(e);
    exit(1);
  }

  // move-away penalty is applied for all features. however, features that didn't fire in
  // this minibatch have a penalty of zero (and penalty derivative of zero). 
  // so we only need to update the derivative and likelihood with the penalty 
  // applied to features in derivativeWRTLambda (i.e. those that fired in this minibatch)
  double totalMoveAwayPenalty = 0;
  if(model.learningInfo.optimizationMethod.subOptMethod->moveAwayPenalty != 0.0) {
    for(unsigned displacedIndex = 0; displacedIndex < lambdasCount; displacedIndex++) {
      if(gradient[displacedIndex] == 0) {
	continue;
      }
      // get the difference
      double newMinusOld = model.lambda->GetParamNewMinusOldWeight(displacedIndex + model.countOfConstrainedLambdaParameters);
      // update the derivative for this feature
      gradient[displacedIndex] += 2.0 * model.learningInfo.optimizationMethod.subOptMethod->moveAwayPenalty * newMinusOld;
      // update the likelihood
      nlogLikelihood += model.learningInfo.optimizationMethod.subOptMethod->moveAwayPenalty * newMinusOld * newMinusOld;
      // update totalMoveAwayPenalty
      totalMoveAwayPenalty += model.learningInfo.optimizationMethod.subOptMethod->moveAwayPenalty * newMinusOld * newMinusOld;
    }
  }
  // debug
  if(model.learningInfo.debugLevel >= DebugLevel::MINI_BATCH) {
    cerr << "-eval(nloglikelihood=" << nlogLikelihood << ",totalMoveAwayPenalty=" << totalMoveAwayPenalty << ")\n";
  }
  //  cerr << "nloglikelihood derivative wrt lambdas: " << endl;
  //  LogLinearParams::PrintParams(derivativeWRTLambda);

  // return the to-be-minimized objective function
  //  cerr << "Evaluate returning " << nlogLikelihood;
  //  cerr << ". step is " << step;
  //  cerr << ". covering data range (" << from << "," << to << ")" << endl;
  //  cerr << "===================================" << endl;
  //  cerr << "gradient: ";
  //  for(map<string, double>::const_iterator gradientIter = derivativeWRTLambda.begin(); 
  //      gradientIter != derivativeWRTLambda.end(); gradientIter++) {
  //    cerr << gradientIter->first << ":" << gradientIter->second << " ";
  //  }
  //  cerr << endl;
  if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
    cerr << endl << " EvaluateNLogLikelihoodDerivativeWRTLambda() for this minibatch took " << (float) (clock() - timestamp) / CLOCKS_PER_SEC << " sec. " << endl;
    cerr << "listing the current values in the gradient array (no problem enountered though): " << endl;
    for(int displacedIndex = 0; displacedIndex < model.lambda->GetParamsCount() - model.countOfConstrainedLambdaParameters; displacedIndex++) {
      cerr << gradient[displacedIndex] << " ";
    } 
    cerr << endl;
  }

  // make sure the gradient and lambdas don't contain a nan
  /*
  for(int displacedIndex = 0; displacedIndex < model.lambda->GetParamsCount() - model.countOfConstrainedLambdaParameters; displacedIndex++) {
    if(isnan(gradient[displacedIndex]) || isinf(gradient[displacedIndex])) {
      if(model.learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
	cerr << "ERROR: gradient[" << displacedIndex << "] = " << gradient[displacedIndex] << ". my mistake. will halt!" << endl;
      }
      assert(false);
    }
    if(isnan(lambdasArray[displacedIndex]) || isinf(lambdasArray[displacedIndex])) {
      if(model.learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
	cerr << "ERROR: lambdasArray[" << displacedIndex << "] = " << lambdasArray[displacedIndex] << ". my mistake. will halt!" << endl;
      }
      //      model.lambda->UpdateParam(displacedIndex + model.countOfConstrainedLambdaParameters, 0.0);
      assert(false);
    }
  }
  */
  //  cerr << "}" << endl;
  return nlogLikelihood;  
}

int LatentCrfModel::LbfgsProgressReport(void *ptrFromSentId,
					const lbfgsfloatval_t *x, 
					const lbfgsfloatval_t *g,
					const lbfgsfloatval_t fx,
					const lbfgsfloatval_t xnorm,
					const lbfgsfloatval_t gnorm,
					const lbfgsfloatval_t step,
					int n,
					int k,
					int ls) {
  LatentCrfModel &model = LatentCrfModel::GetInstance();

  // for debugging only
  /*
  double *lambdasArray = model.lambda->GetParamWeightsArray() + model.countOfConstrainedLambdaParameters;
  unsigned lambdasArrayLength = model.lambda->GetParamsCount() - model.countOfConstrainedLambdaParameters;
  for(int displacedIndex = 0; displacedIndex < lambdasArrayLength; displacedIndex++) {
    if(isnan(lambdasArray[displacedIndex]) || isinf(lambdasArray[displacedIndex])) {
      if(model.learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
	cerr << "ERROR: lambdasArray[" << displacedIndex << "] = " << lambdasArray[displacedIndex] << ". my mistake (in LatentCrfModel::LbfgsProgressReport). will halt!" << endl;
      }
      //      model.lambda->UpdateParam(displacedIndex + model.countOfConstrainedLambdaParameters, 0.0);
      assert(false);
    }
  }
  */

  int index = *((int*)ptrFromSentId), from, to;
  if(index == -1) {
    from = 0;
    to = model.data.size();
  } else {
    from = index;
    to = min((int)model.data.size(), from + model.learningInfo.optimizationMethod.subOptMethod->miniBatchSize);
  }
  
  // show progress
  if(model.learningInfo.debugLevel >= DebugLevel::MINI_BATCH) {
    cerr << endl << "-report sents:" << from << "-" << to;
    cerr << "\tlbfgs Iteration " << k;
    cerr << ":\tobjective = " << fx;
  }
  if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
    cerr << ",\txnorm = " << xnorm;
    cerr << ",\tgnorm = " << gnorm;
    cerr << ",\tstep = " << step;
  }
  if(model.learningInfo.debugLevel >= DebugLevel::MINI_BATCH) {
    cerr << endl;
  }

  // update the old lambdas
  if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
    cerr << "updating the old lambda params (necessary for applying the moveAwayPenalty) ..." << endl;
  }
  model.lambda->UpdateOldParamWeights();
  if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
    cerr << "done" << endl;
  }

  /*
  // for debug only: make sure we didn't write nans to the lambdasArray
  for(int displacedIndex = 0; displacedIndex < lambdasArrayLength; displacedIndex++) {
    if(isnan(lambdasArray[displacedIndex]) || isinf(lambdasArray[displacedIndex])) {
      if(model.learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
	cerr << "ERROR: lambdasArray[" << displacedIndex << "] = " << lambdasArray[displacedIndex] << ". my mistake (in LatentCrfModel::LbfgsProgressReport). will halt!" << endl;
      }
      //      model.lambda->UpdateParam(displacedIndex + model.countOfConstrainedLambdaParameters, 0.0);
      assert(false);
    }
    } */ 

  return 0;
}

// make sure all features which may fire on this training data have a corresponding parameter in lambda (member)
void LatentCrfModel::WarmUp() {
  if(learningInfo.debugLevel >= DebugLevel::CORPUS) {
    cerr << "warming up..." << endl;
  }

  // make sure no features are fired yet. 
  assert(lambda->GetParamsCount() == 0);

  // first, add constrained features here and set their weights by hand. those weights will not be optimized.
  if(learningInfo.debugLevel >= DebugLevel::CORPUS) {
    cerr << "adding constrained lambda features..." << endl;
  }
  std::map<string, double> activeFeatures;
  int yI, xI;
  int yIM1_dummy, index; // we don't really care
  vector<int> x;
  string xIString;
  vector<bool> constrainedFeatureTypes(lambda->COUNT_OF_FEATURE_TYPES, false);
  for(int i = 0; i < learningInfo.constraints.size(); i++) {
    switch(learningInfo.constraints[i].type) {
      // constrains the latent variable corresponding to certain types
    case ConstraintType::yIExclusive_xIString:
      // we only want to constrain one specific feature type
      std::fill(constrainedFeatureTypes.begin(), constrainedFeatureTypes.end(), false);
      constrainedFeatureTypes[54] = true;
      // parse the constraint
      xIString.clear();
      learningInfo.constraints[i].GetFieldsOfConstraintType_yIExclusive_xIString(yI, xIString);
      xI = vocabEncoder.Encode(xIString);
      // fire positively constrained features
      x.clear();
      x.push_back(xI);
      yIM1_dummy = yI; // we don't really care
      index = 0; // we don't really care
      activeFeatures.clear();
      lambda->FireFeatures(yI, yIM1_dummy, x, index, constrainedFeatureTypes, activeFeatures);
      // set appropriate weights to favor those parameters
      for(map<string, double>::const_iterator featureIter = activeFeatures.begin(); featureIter != activeFeatures.end(); featureIter++) {
	lambda->UpdateParam(featureIter->first, REWARD_FOR_CONSTRAINED_FEATURES);
      }
      // negatively constrained features (i.e. since xI is constrained to get the label yI, any other label should be penalized)
      for(set<int>::const_iterator yDomainIter = yDomain.begin(); yDomainIter != yDomain.end(); yDomainIter++) {
	if(*yDomainIter == yI) {
	  continue;
	}
	// fire the negatively constrained features
	activeFeatures.clear();
	lambda->FireFeatures(*yDomainIter, yIM1_dummy, x, index, constrainedFeatureTypes, activeFeatures);
	// set appropriate weights to penalize those parameters
	for(map<string, double>::const_iterator featureIter = activeFeatures.begin(); featureIter != activeFeatures.end(); featureIter++) {
	  lambda->UpdateParam(featureIter->first, PENALTY_FOR_CONSTRAINED_FEATURES);
	}   
      }
      break;
    case ConstraintType::yI_xIString:
      // we only want to constrain one specific feature type
      std::fill(constrainedFeatureTypes.begin(), constrainedFeatureTypes.end(), false);
      constrainedFeatureTypes[54] = true;
      // parse the constraint
      xIString.clear();
      learningInfo.constraints[i].GetFieldsOfConstraintType_yI_xIString(yI, xIString);
      xI = vocabEncoder.Encode(xIString);
      // fire positively constrained features
      x.clear();
      x.push_back(xI);
      yIM1_dummy = yI; // we don't really care
      index = 0; // we don't really care
      activeFeatures.clear();
      lambda->FireFeatures(yI, yIM1_dummy, x, index, constrainedFeatureTypes, activeFeatures);
      // set appropriate weights to favor those parameters
      for(map<string, double>::const_iterator featureIter = activeFeatures.begin(); featureIter != activeFeatures.end(); featureIter++) {
	lambda->UpdateParam(featureIter->first, REWARD_FOR_CONSTRAINED_FEATURES);
      }
      break;
    default:
      assert(false);
      break;
    }
  }
  // take note of the number of constrained lambda parameters. use this to limit optimization to non-constrained params
  countOfConstrainedLambdaParameters = lambda->GetParamsCount();
  if(learningInfo.debugLevel >= DebugLevel::CORPUS) {
    cerr << "done adding constrainted lambda features. Count:" << lambda->GetParamsCount() << endl;
  }

  // then, add all remaining features warranted by the training set.
  UniformSampler uniform;
  //  cerr << "lambda.GetParamsCount() = " << lambda.GetParamsCount() << endl;
  for(int sentId = 0; sentId < data.size(); sentId++) {
    //        cerr << "now processing sent# " << sentId << endl;
    // build the FST
    VectorFst<LogArc> lambdaFst;
    vector<fst::LogWeight> lambdaAlphas, lambdaBetas;
    BuildLambdaFst(data[sentId], lambdaFst, lambdaAlphas, lambdaBetas);
    // compute the F map from this sentence (implicitly adds the fired features to lambda parameters)
    FastSparseVector<LogVal<double> > F;
    ComputeF(data[sentId], lambdaFst, lambdaAlphas, lambdaBetas, F);
  }
  if(learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
    cerr << "lambdas initialized to: " << endl;
    lambda->PrintParams();
    cerr << "warmup done. Lambda params count:" << lambda->GetParamsCount() << endl;
  }
}

void LatentCrfModel::UpdateThetaMleForSent(const unsigned sentId, 
					   MultinomialParams::ConditionalMultinomialParam &mle, 
					   map<int, double> &mleMarginals) {
  if(learningInfo.debugLevel >= DebugLevel::SENTENCE) {
    cerr << "sentId = " << sentId << endl;
  }
  assert(sentId < data.size());
  // build the FST
  VectorFst<LogArc> thetaLambdaFst;
  vector<fst::LogWeight> alphas, betas;
  BuildThetaLambdaFst(data[sentId], data[sentId], thetaLambdaFst, alphas, betas);
  // compute the B matrix for this sentence
  map< int, map< int, double > > B;
  B.clear();
  ComputeB(this->data[sentId], this->data[sentId], thetaLambdaFst, alphas, betas, B);
  // compute the C value for this sentence
  double nLogC = ComputeNLogC(thetaLambdaFst, betas);
  //cerr << "nloglikelihood += " << nLogC << endl;
  // update mle for each z^*|y^* fired
  for(map< int, map<int, double> >::const_iterator yIter = B.begin(); yIter != B.end(); yIter++) {
    int y_ = yIter->first;
    for(map<int, double>::const_iterator zIter = yIter->second.begin(); zIter != yIter->second.end(); zIter++) {
      int z_ = zIter->first;
      double b = zIter->second;
      double nLogb = MultinomialParams::nLog(b);
      double bOverC = MultinomialParams::nExp(nLogb - nLogC);
      if(learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
	cerr << "b(" << z_ << "|" << y_ << ")/c = " << bOverC << endl;
      }
      mle[y_][z_] += bOverC;
      mleMarginals[y_] += bOverC;
    }
  }
}

void LatentCrfModel::NormalizeThetaMle(MultinomialParams::ConditionalMultinomialParam &mle, 
				       map<int, double> &mleMarginals) {
  // fix theta mle estimates
  for(map<int,  map<int, double> >::const_iterator yIter = mle.begin(); yIter != mle.end(); yIter++) {
    int y_ = yIter->first;
    double unnormalizedMarginalProbz_giveny_ = 0.0;
    // verify that \sum_z* mle[y*][z*] = mleMarginals[y*]
    for(map<int, double>::const_iterator zIter = yIter->second.begin(); zIter != yIter->second.end(); zIter++) {
      int z_ = zIter->first;
      double unnormalizedProbz_giveny_ = zIter->second;
      unnormalizedMarginalProbz_giveny_ += unnormalizedProbz_giveny_;
    }
    if(abs((mleMarginals[y_] - unnormalizedMarginalProbz_giveny_) / mleMarginals[y_]) > 0.01) {
      cerr << "ERROR: abs( (mleMarginals[y_] - unnormalizedMarginalProbz_giveny_) / mleMarginals[y_] ) = ";
      cerr << abs((mleMarginals[y_] - unnormalizedMarginalProbz_giveny_) / mleMarginals[y_]); 
      cerr << "mleMarginals[y_] = " << mleMarginals[y_] << " unnormalizedMarginalProbz_giveny_ = " << unnormalizedMarginalProbz_giveny_;
	cerr << " --error ignored, but try to figure out what's wrong!" << endl;
    }
    // normalize the mle estimates to sum to one for each context
    for(map<int, double>::const_iterator zIter = yIter->second.begin(); zIter != yIter->second.end(); zIter++) {
      int z_ = zIter->first;
      double normalizedProbz_giveny_ = zIter->second / mleMarginals[y_];
      mle[y_][z_] = normalizedProbz_giveny_;
      // take the nlog
      mle[y_][z_] = MultinomialParams::nLog(mle[y_][z_]);
    }
  }
}

// keeps looking for a slave that's not busy and return its rank
unsigned LatentCrfModel::FindASlackingSlave(vector<bool> &busySlaves, 
					    vector<boost::mpi::request> &slavesMleRequests, 
					    vector<boost::mpi::request> &slavesMleMarginalRequests,
					    vector<MultinomialParams::ConditionalMultinomialParam> &slavesMleResults,
					    vector< map<int, double> > &slavesMleMarginalResults,
					    MultinomialParams::ConditionalMultinomialParam &mle,
					    map<int, double> &mleMarginals) {
  while(true) {
    for(unsigned i = 1; i < busySlaves.size(); i++) {
      if(!busySlaves[i]) {
	busySlaves[i] = true;
	return i;
      }
    }
    boost::this_thread::sleep( boost::posix_time::seconds(1) );
    //    cerr << "none of hte slaves are slacking. lets see if any of them finished their work!" << endl;
    // may one of them finished its work by now!
    CollectSlavesWork(busySlaves, slavesMleRequests, slavesMleMarginalRequests, slavesMleResults, slavesMleMarginalResults, mle, mleMarginals);
  }
}

// collect results from slaves
void LatentCrfModel::CollectSlavesWork(vector<bool> &busySlaves, 
				       vector<boost::mpi::request> &slavesMleRequests, 
				       vector<boost::mpi::request> &slavesMleMarginalRequests, 
				       vector<MultinomialParams::ConditionalMultinomialParam> &slavesMleResults,
				       vector< map<int, double> > &slavesMleMarginalResults,
				       MultinomialParams::ConditionalMultinomialParam &mle,
				       map<int, double> &mleMarginals) {
  for(unsigned i = 1; i < busySlaves.size(); i++) {
    if(busySlaves[i]) {
      boost::optional<boost::mpi::status> mleStatus = slavesMleRequests[i].test();
      boost::optional<boost::mpi::status> mleMarginalStatus = slavesMleMarginalRequests[i].test();
      if(mleStatus && mleMarginalStatus) {
	cerr << "master recieved an mle communication from slave#" << i << " with error code " << mleStatus->error()<< endl;
	cerr << "master recieved an mleMarginal communication from slave#" << i << " with error code " << mleMarginalStatus->error()<< endl;
	busySlaves[i] = false;
	// now, that our slave has done the hard work, the master gets to accumulate those numbers on its own numbers
	for(MultinomialParams::ConditionalMultinomialParam::const_iterator yIter = slavesMleResults[i].begin();
	    yIter != slavesMleResults[i].end();
	    yIter++) {
	  // accumulate mle
	  for(std::map<int, double>::const_iterator zIter = yIter->second.begin(); 
	      zIter != yIter->second.end();
	      zIter++) {
	    mle[yIter->first][zIter->first] += zIter->second;
	  }
	  // accumulate mle marginals
	  mleMarginals[yIter->first] += slavesMleMarginalResults[i][yIter->first];
	}
      }
    }
  }
}

lbfgs_parameter_t LatentCrfModel::SetLbfgsConfig() {
  // lbfgs configurations
  lbfgs_parameter_t lbfgsParams;
  lbfgs_parameter_init(&lbfgsParams);
  assert(learningInfo.optimizationMethod.subOptMethod != 0);
  lbfgsParams.max_iterations = learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxIterations;
  lbfgsParams.m = learningInfo.optimizationMethod.subOptMethod->lbfgsParams.memoryBuffer;
  lbfgsParams.xtol = learningInfo.optimizationMethod.subOptMethod->lbfgsParams.precision;
  lbfgsParams.max_linesearch = learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxEvalsPerIteration;
  if(learningInfo.optimizationMethod.subOptMethod->lbfgsParams.l1) {
    lbfgsParams.orthantwise_c = learningInfo.optimizationMethod.subOptMethod->regularizationStrength;
    // this is the only linesearch algorithm that seems to work with orthantwise lbfgs
    lbfgsParams.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
  }
  return lbfgsParams;
}

void LatentCrfModel::BlockCoordinateDescent() {  
  
  // add all features in this data set to lambda.params
  if(learningInfo.mpiWorld->rank() == 0) {
    WarmUp();
  }
  
  // broadcast the initial theta and lambda parameters to the slaves
  /*
  int dummy;
  for(unsigned i = 1; i < learningInfo.mpiWorld->size(); i++) {
    // theta params
    learningInfo.mpiWorld->send(i, LatentCrfModel::MPI_TAG_UPDATE_SLAVE_THETA, nLogTheta);
    cerr << "master syncronously sent slave#" << i << " the message 'UPDATE_SLAVE_THETA'" << endl;
    //learningInfo.mpiWorld->recv(i, LatentCrfModel::MPI_TAG_ACK_THETA_UPDATED);
    //cerr << "ACK RECEIVED\nmaster synchronously received an acknowledgement from slave#" << i << endl;
    // lambda param ids
    learningInfo.mpiWorld->send(i, LatentCrfModel::MPI_TAG_UPDATE_SLAVE_LAMBDA_IDS, lambda->paramIds);
    cerr << "master synchronously sent slave#" << i << " the message 'UPDATE_SLAVE_LAMBDA_IDS'" << endl;
    cerr << "master's lambda->paramIds.size() = " << lambda->paramIds.size() << endl;
    cerr << "master's lambda->paramWeights.size() = " << lambda->paramWeights.size() << endl;
    cerr << "master's lambda->paramIndexes.size() = " << lambda->paramIndexes.size() << endl;
    learningInfo.mpiWorld->recv(i, LatentCrfModel::MPI_TAG_ACK_LAMBDA_INDEXES_UPDATED);
    cerr << "master synchronously received an acknowledgement from slave#" << i << endl;
    // lambda params
    learningInfo.mpiWorld->send(i, LatentCrfModel::MPI_TAG_UPDATE_SLAVE_LAMBDA, lambda->paramWeights);
    cerr << "master synchronously sent slave#" << i << " the message 'UPDATE_SLAVE_LAMBDA'" << endl;
    learningInfo.mpiWorld->recv(i, LatentCrfModel::MPI_TAG_ACK_LAMBDA_UPDATED);
    cerr << "master synchronously received an acknowledgement from slave#" << i << endl;
  }
  */
  mpi::broadcast<MultinomialParams::ConditionalMultinomialParam>(*learningInfo.mpiWorld, nLogTheta, 0);
  lambda->Broadcast(*learningInfo.mpiWorld, 0);

  // TRAINING ITERATIONS
  bool converged = false;
  do {

    // UPDATE THETAS by normalizing soft counts (i.e. the closed form MLE solution)
    // data structure to hold theta MLE estimates
    MultinomialParams::ConditionalMultinomialParam mle;
    map<int, double> mleMarginals;
    // debug info
    if(learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
      double temp = ComputeCorpusNloglikelihood();
      cerr << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << endl;
      cerr << "nloglikelihood before optimizing thetas = " << temp << endl;
      cerr << "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv" << endl;
    }
    if(learningInfo.debugLevel >= DebugLevel::CORPUS) {
      cerr << "updating thetas..." << endl;
    }

    // update the mle for each sentence
    for(unsigned sentId = 0; sentId < data.size(); sentId++) {
      // sentId is assigned to the process # (sentId % world.size())
      if(sentId % learningInfo.mpiWorld->size() != learningInfo.mpiWorld->rank()) {
	continue;
      }
      UpdateThetaMleForSent(sentId, mle, mleMarginals);
    }
    /*
    // wait until all slaves finish their work and collect it
    if(learningInfo.mpiWorld->size() > 1) {
      mpi::wait_all(slavesMleRequests.data() + 1, slavesMleRequests.data() + slavesMleRequests.size());
      mpi::wait_all(slavesMleMarginalRequests.data() + 1, slavesMleMarginalRequests.data() + slavesMleMarginalRequests.size());
      CollectSlavesWork(busySlaves, slavesMleRequests, slavesMleMarginalRequests, slavesMleResults, slavesMleMarginalResults, mle, mleMarginals);
    }
    */
    // accumulate mle counts from slaves
    mpi::reduce<MultinomialParams::ConditionalMultinomialParam>(*learningInfo.mpiWorld, mle, mle, MultinomialParams::AccumulateConditionalMultinomials, 0);
    mpi::reduce<MultinomialParams::MultinomialParam>(*learningInfo.mpiWorld, mleMarginals, mleMarginals, MultinomialParams::AccumulateMultinomials, 0);

    // normalize mle and update nLogTheta on master
    if(learningInfo.mpiWorld->rank() == 0) {
      NormalizeThetaMle(mle, mleMarginals);
      nLogTheta = mle;
    }

    // update nLogTheta on slaves
    mpi::broadcast<MultinomialParams::ConditionalMultinomialParam>(*learningInfo.mpiWorld, nLogTheta, 0);

    // debug info
    if(learningInfo.persistParamsAfterEachIteration && learningInfo.mpiWorld->rank() == 0) {
      stringstream thetaParamsFilename;
      thetaParamsFilename << outputPrefix << "." << learningInfo.iterationsCount;
      thetaParamsFilename << ".theta";
      if(learningInfo.debugLevel >= DebugLevel::CORPUS) {
	cerr << "persisting theta parameters after iteration " << learningInfo.iterationsCount << " at " << thetaParamsFilename.str() << endl;
      }
      MultinomialParams::PersistParams(thetaParamsFilename.str(), nLogTheta, vocabEncoder);
    }
    if(learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
      cerr << "theta params: " << endl;
      MultinomialParams::PrintParams(nLogTheta, vocabEncoder);
    }

    // update the lambdas with mini-batch lbfgs
    double* lambdasArray;
    int lambdasArrayLength;
    double nlogLikelihood = 0;
    if(learningInfo.optimizationMethod.subOptMethod->miniBatchSize <= 0) {
      learningInfo.optimizationMethod.subOptMethod->miniBatchSize = data.size();
    }
    for(int sentId = 0; sentId < data.size(); sentId += learningInfo.optimizationMethod.subOptMethod->miniBatchSize) {

      // populate lambdasArray and lambasArrayLength
      // don't optimize all parameters. only optimize non-constrained ones
      lambdasArray = lambda->GetParamWeightsArray() + countOfConstrainedLambdaParameters;
      lambdasArrayLength = lambda->GetParamsCount() - countOfConstrainedLambdaParameters;
      
      // set lbfgs configurations
      lbfgs_parameter_t lbfgsParams = SetLbfgsConfig();

      // call the lbfgs minimizer for this mini-batch
      double optimizedMiniBatchNLogLikelihood = 0;
      if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH) {
	int to = min(sentId+learningInfo.optimizationMethod.subOptMethod->miniBatchSize, (int)data.size());
	cerr << "calling lbfgs on sents " << sentId << "-" << to << endl;
      }
      if(learningInfo.optimizationMethod.subOptMethod->algorithm == LBFGS) {

	int lbfgsStatus = lbfgs(lambdasArrayLength, lambdasArray, &optimizedMiniBatchNLogLikelihood, 
				EvaluateNLogLikelihoodDerivativeWRTLambda, LbfgsProgressReport, &sentId, &lbfgsParams);
	// debug
	if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH) {
	  cerr << "lbfgsStatusCode = " << LbfgsUtils::LbfgsStatusIntToString(lbfgsStatus) << " = " << lbfgsStatus << endl;
	}
	if(learningInfo.retryLbfgsOnRoundingErrors && lbfgsStatus == LBFGSERR_ROUNDING_ERROR) {
	  if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH) {
	    cerr << "rounding error (" << lbfgsStatus << "). my gradient might be buggy." << endl << "retry..." << endl;
	  }
	  lbfgsStatus = lbfgs(lambdasArrayLength, lambdasArray, &optimizedMiniBatchNLogLikelihood,
			      EvaluateNLogLikelihoodDerivativeWRTLambda, LbfgsProgressReport, &sentId, &lbfgsParams);
	  if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH) {
	    cerr << "lbfgsStatusCode = " << LbfgsUtils::LbfgsStatusIntToString(lbfgsStatus) << " = " << lbfgsStatus << endl;
	  }
	}
      } else if(learningInfo.optimizationMethod.subOptMethod->algorithm == SIMULATED_ANNEALING) {
	simulatedAnnealer.set_up(EvaluateNLogLikelihood, lambdasArrayLength);
	// initialize the parameters array
	float simulatedAnnealingArray[lambdasArrayLength];
	for(int i = 0; i < lambdasArrayLength; i++) {
	  simulatedAnnealingArray[i] = lambdasArray[i];
	}
	simulatedAnnealer.initial(simulatedAnnealingArray);
	// optimize
	simulatedAnnealer.anneal(10);
	// get the optimum parameters
	simulatedAnnealer.current(simulatedAnnealingArray);
	for(int i = 0; i < lambdasArrayLength; i++) {
	  lambdasArray[i] = simulatedAnnealingArray[i];
	}
      } else {
	assert(false);
      }
      
      // debug info
      if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH) {
	cerr << "optimized nloglikelihood is " << optimizedMiniBatchNLogLikelihood << endl;
      }
      
      // update iteration's nloglikelihood
      if(isnan(optimizedMiniBatchNLogLikelihood) || isinf(optimizedMiniBatchNLogLikelihood)) {
	if(learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
	  cerr << "ERROR: optimizedMiniBatchNLogLikelihood = " << optimizedMiniBatchNLogLikelihood << ". didn't add this batch's likelihood to the total likelihood. will halt!" << endl;
	}
	assert(false);
      } else {
	nlogLikelihood += optimizedMiniBatchNLogLikelihood;
      }
    }

    // debug info
    if(learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
      lambda->PrintParams();
    }

    // persist updated lambda params
    stringstream lambdaParamsFilename;
    if(learningInfo.persistParamsAfterEachIteration && learningInfo.mpiWorld->rank() == 0) {
      lambdaParamsFilename << outputPrefix << "." << learningInfo.iterationsCount << ".lambda";
      if(learningInfo.debugLevel >= DebugLevel::CORPUS) {
	cerr << "persisting lambda parameters after iteration " << learningInfo.iterationsCount << " at " << lambdaParamsFilename.str() << endl;
      }
      lambda->PersistParams(lambdaParamsFilename.str());
    }

    // debug info
    if(learningInfo.debugLevel >= DebugLevel::CORPUS) {
      cerr << "finished coordinate descent iteration #" << learningInfo.iterationsCount << " nloglikelihood=" << nlogLikelihood << endl;
    }
    
    // update learningInfo
    learningInfo.logLikelihood.push_back(nlogLikelihood);
    learningInfo.iterationsCount++;

    // check convergence
    if(learningInfo.mpiWorld.rank() == 0) {
      converged = learningInfo.IsModelConverged();
    }
    
    // broadcast the convergence decision
    mpi::broadcast<bool>(*learningInfo.mpiWorld, converged, 0);
  
  } while(!converged);

  // debug
  if(learningInfo.mpiWorld->rank() == 0) {
    lambda->PersistParams(outputPrefix + string(".final.lambda"));
    MultinomialParams::PersistParams(outputPrefix + string(".final.theta"), nLogTheta, vocabEncoder);
  }
}

void LatentCrfModel::Label(vector<string> &tokens, vector<int> &labels) {
  assert(labels.size() == 0);
  assert(tokens.size() > 0);
  vector<int> tokensInt;
  for(int i = 0; i < tokens.size(); i++) {
    tokensInt.push_back(vocabEncoder.Encode(tokens[i]));
  }
  Label(tokensInt, labels);
}

void LatentCrfModel::Label(vector<int> &tokens, vector<int> &labels) {
  assert(labels.size() == 0); 
  assert(tokens.size() > 0);
  VectorFst<LogArc> fst;
  vector<fst::LogWeight> alphas, betas;
  BuildThetaLambdaFst(tokens, tokens, fst, alphas, betas);
  VectorFst<StdArc> fst2, shortestPath;
  fst::ArcMap(fst, &fst2, LogToTropicalMapper());
  fst::ShortestPath(fst2, &shortestPath);
  std::vector<int> dummy;
  FstUtils::LinearFstToVector(shortestPath, dummy, labels);
  assert(labels.size() == tokens.size());
}

void LatentCrfModel::Label(vector<vector<int> > &tokens, vector<vector<int> > &labels) {
  assert(labels.size() == 0);
  labels.resize(tokens.size());
  for(int i = 0; i < tokens.size(); i++) {
    Label(tokens[i], labels[i]);
  }
}

void LatentCrfModel::Label(vector<vector<string> > &tokens, vector<vector<int> > &labels) {
  assert(labels.size() == 0);
  labels.resize(tokens.size());
  for(int i = 0 ; i <tokens.size(); i++) {
    Label(tokens[i], labels[i]);
  }
}

void LatentCrfModel::Label(string &inputFilename, string &outputFilename) {
  std::vector<std::vector<std::string> > tokens;
  StringUtils::ReadTokens(inputFilename, tokens);
  vector<vector<int> > labels;
  Label(tokens, labels);
  StringUtils::WriteTokens(outputFilename, labels);
}

void LatentCrfModel::Analyze(string &inputFilename, string &outputFilename) {
  // label
  std::vector<std::vector<std::string> > tokens;
  StringUtils::ReadTokens(inputFilename, tokens);
  vector<vector<int> > labels;
  Label(tokens, labels);
  // analyze
  map<int, map<string, int> > labelToTypesAndCounts;
  map<string, map<int, int> > typeToLabelsAndCounts;
  for(int sentId = 0; sentId < tokens.size(); sentId++) {
    for(int i = 0; i < tokens[sentId].size(); i++) {
      labelToTypesAndCounts[labels[sentId][i]][tokens[sentId][i]]++;
      typeToLabelsAndCounts[tokens[sentId][i]][labels[sentId][i]]++;
    }
  }
  // write the number of tokens of each labels
  std::ofstream outputFile(outputFilename.c_str(), std::ios::out);
  outputFile << "# LABEL HISTOGRAM #" << endl;
  for(map<int, map<string, int> >::const_iterator labelIter = labelToTypesAndCounts.begin(); labelIter != labelToTypesAndCounts.end(); labelIter++) {
    outputFile << "label:" << labelIter->first;
    int totalCount = 0;
    for(map<string, int>::const_iterator typeIter = labelIter->second.begin(); typeIter != labelIter->second.end(); typeIter++) {
      totalCount += typeIter->second;
    }
    outputFile << " tokenCount:" << totalCount << endl;
  }
  // write the types of each label
  outputFile << endl << "# LABEL -> TYPES:COUNTS #" << endl;
  for(map<int, map<string, int> >::const_iterator labelIter = labelToTypesAndCounts.begin(); labelIter != labelToTypesAndCounts.end(); labelIter++) {
    outputFile << "label:" << labelIter->first << endl << "\ttypes: " << endl;
    for(map<string, int>::const_iterator typeIter = labelIter->second.begin(); typeIter != labelIter->second.end(); typeIter++) {
      outputFile << "\t\t" << typeIter->first << ":" << typeIter->second << endl;
    }
  }
  // write the labels of each type
  outputFile << endl << "# TYPE -> LABELS:COUNT #" << endl;
  for(map<string, map<int, int> >::const_iterator typeIter = typeToLabelsAndCounts.begin(); typeIter != typeToLabelsAndCounts.end(); typeIter++) {
    outputFile << "type:" << typeIter->first << "\tlabels: ";
    for(map<int, int>::const_iterator labelIter = typeIter->second.begin(); labelIter != typeIter->second.end(); labelIter++) {
      outputFile << labelIter->first << ":" << labelIter->second << " ";
    }
    outputFile << endl;
  }
  outputFile.close();
}

double LatentCrfModel::ComputeVariationOfInformation(string &aLabelsFilename, string &bLabelsFilename) {
  vector<string> clusteringA, clusteringB;
  vector<vector<string> > clusteringAByLine, clusteringBByLine;
  StringUtils::ReadTokens(aLabelsFilename, clusteringAByLine);
  StringUtils::ReadTokens(bLabelsFilename, clusteringBByLine);
  assert(clusteringAByLine.size() == clusteringBByLine.size());
  for(int i = 0; i < clusteringAByLine.size(); i++) {
    assert(clusteringAByLine[i].size() == clusteringBByLine[i].size());
    for(int j = 0; j < clusteringAByLine[i].size(); j++) {
      clusteringA.push_back(clusteringAByLine[i][j]);
      clusteringB.push_back(clusteringBByLine[i][j]);			    
    }
  }
  return ClustersComparer::ComputeVariationOfInformation(clusteringA, clusteringB);
}
