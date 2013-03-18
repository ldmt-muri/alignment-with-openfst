#include "LatentCrfModel.h"

using namespace std;
using namespace OptAlgorithm;

// singlenton instance definition and trivial initialization
LatentCrfModel* LatentCrfModel::instance = 0;

// singleton
LatentCrfModel& LatentCrfModel::GetInstance(const string &textFilename, 
					    const string &outputPrefix, 
					    LearningInfo &learningInfo, 
					    unsigned NUMBER_OF_LABELS, 
					    unsigned FIRST_LABEL_ID) {
  if(!LatentCrfModel::instance) {
    LatentCrfModel::instance = new LatentCrfModel(textFilename, outputPrefix, learningInfo, NUMBER_OF_LABELS, FIRST_LABEL_ID);
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
LatentCrfModel::LatentCrfModel(const string &textFilename, 
			       const string &outputPrefix, 
			       LearningInfo &learningInfo, 
			       unsigned NUMBER_OF_LABELS, 
			       unsigned FIRST_LABEL_ID) : 
  vocabEncoder(textFilename),
  gaussianSampler(0.0, 10.0),
  UnsupervisedSequenceTaggingModel(textFilename) {

  countOfConstrainedLambdaParameters = 0;

  AddEnglishClosedVocab();
  
  if(learningInfo.mpiWorld->rank() == 0) {
    vocabEncoder.PersistVocab(outputPrefix + string(".vocab"));
  }
  bool syncAllProcesses;
  mpi::broadcast<bool>(*learningInfo.mpiWorld, syncAllProcesses, 0);
  VocabDecoder *vocabDecoder = new VocabDecoder(outputPrefix + string(".vocab"));
  lambda = new LogLinearParams(*vocabDecoder);
  
  // set member variables
  this->textFilename = textFilename;
  this->outputPrefix = outputPrefix;
  this->learningInfo = learningInfo;
  this->lambda->SetLearningInfo(learningInfo);
  
  // set constants
  this->START_OF_SENTENCE_Y_VALUE = FIRST_LABEL_ID - 1;
  this->FIRST_ALLOWED_LABEL_VALUE = FIRST_LABEL_ID;
  assert(START_OF_SENTENCE_Y_VALUE > 0);

  // POS tag yDomain
  unsigned latentClasses = NUMBER_OF_LABELS;
  assert(latentClasses > 1);
  this->yDomain.insert(START_OF_SENTENCE_Y_VALUE); // the conceptual yValue of word at position -1 in a sentence
  for(unsigned i = 0; i < latentClasses; i++) {
    this->yDomain.insert(START_OF_SENTENCE_Y_VALUE + i + 1);
  }
  
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
  enabledFeatureTypes[51] = true;   // y_i:y_{i-1}
  //  enabledFeatureTypes[52] = true; // y_i:x_{i-2}
  enabledFeatureTypes[53] = true; // y_i:x_{i-1}
  enabledFeatureTypes[54] = true;   // y_i:x_i
  enabledFeatureTypes[55] = true; // y_i:x_{i+1}
  //enabledFeatureTypes[56] = true; // y_i:x_{i+2}
  enabledFeatureTypes[57] = true; // y_i:i
  //  enabledFeatureTypes[58] = true;
  //  enabledFeatureTypes[59] = true;
  //  enabledFeatureTypes[60] = true;
  //  enabledFeatureTypes[61] = true;
  //  enabledFeatureTypes[62] = true;
  //  enabledFeatureTypes[63] = true;
  //  enabledFeatureTypes[64] = true;
  //  enabledFeatureTypes[65] = true;
  enabledFeatureTypes[66] = true; // y_i:(|x|-i)
  enabledFeatureTypes[67] = true; // capital and i != 0
  //enabledFeatureTypes[68] = true;
  enabledFeatureTypes[69] = true; // coarse hash functions
  //enabledFeatureTypes[70] = true;
  //enabledFeatureTypes[71] = true; // y_i:x_{i-1} where x_{i-1} is closed vocab
  //enabledFeatureTypes[72] = true;
  //enabledFeatureTypes[73] = true; // y_i:x_{i+1} where x_{i+1} is closed vocab
  //enabledFeatureTypes[74] = true;
  //enabledFeatureTypes[75] = true; // y_i

  // initialize (and normalize) the log theta params to gaussians
  InitTheta();

  // make sure all slaves have the same theta values
  BroadcastTheta();

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
  
  // hand-crafted weights for constrained features
  REWARD_FOR_CONSTRAINED_FEATURES = 10.0;
  PENALTY_FOR_CONSTRAINED_FEATURES = -10.0;

  // initialize the lambda parameters
  // add all features in this data set to lambda.params
  WarmUp();
}

void LatentCrfModel::AddEnglishClosedVocab() {
  string closedVocab[] = {"a", "an", "the", 
			  "some", "one", "many", "few", "much",
			  "from", "to", "at", "by", "in", "on", "for", "as",
			  ".", ",", ";", "!", "?",
			  "is", "are", "be", "am", "was", "were",  
			  "has", "have", "had",
			  "i", "you", "he", "she", "they", "we", "it",
			  "myself", "himself", "themselves", "herself", "yourself",
			  "this", "that", "which",
			  "and", "or", "but", "not",
			  "what", "how", "why", "when",
			  "can", "could", "will", "would", "shall", "should", "must"};
  vector<string> closedVocabVector(closedVocab, closedVocab + sizeof(closedVocab) / sizeof(closedVocab[0]) );
  for(unsigned i = 0; i < closedVocabVector.size(); i++) {
    vocabEncoder.AddToClosedVocab(closedVocabVector[i]);
    // add the capital initial version as well
    if(closedVocabVector[i][0] >= 'a' && closedVocabVector[i][0] <= 'z') {
      closedVocabVector[i][0] += ('A' - 'a');
       vocabEncoder.AddToClosedVocab(closedVocabVector[i]);
    }
  }
}

void LatentCrfModel::InitTheta() {
  if(learningInfo.mpiWorld->rank() == 0 && learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
    cerr << "master" << learningInfo.mpiWorld->rank() << ": initializing thetas...";
  }

  // first initialize nlogthetas to unnormalized gaussians
  if(learningInfo.zIDependsOnYIM1) {
    nLogThetaGivenTwoLabels.params.clear();
    for(set<int>::const_iterator yDomainIter = yDomain.begin(); yDomainIter != yDomain.end(); yDomainIter++) {
      for(set<int>::const_iterator yDomainIter2 = yDomain.begin(); yDomainIter2 != yDomain.end(); yDomainIter2++) {
	for(set<int>::const_iterator zDomainIter = xDomain.begin(); zDomainIter != xDomain.end(); zDomainIter++) {
	  nLogThetaGivenTwoLabels.params[std::pair<int, int>(*yDomainIter, *yDomainIter2)][*zDomainIter] = fabs(gaussianSampler.Draw());
	}
      }
    }
  } else {
    nLogThetaGivenOneLabel.params.clear();
    for(set<int>::const_iterator yDomainIter = yDomain.begin(); yDomainIter != yDomain.end(); yDomainIter++) {
      for(set<int>::const_iterator zDomainIter = xDomain.begin(); zDomainIter != xDomain.end(); zDomainIter++) {
	nLogThetaGivenOneLabel.params[*yDomainIter][*zDomainIter] = abs(gaussianSampler.Draw());
      }
    }
  }

  // then normalize them
  if(learningInfo.zIDependsOnYIM1) {
    MultinomialParams::NormalizeParams(nLogThetaGivenTwoLabels);
  } else {
    MultinomialParams::NormalizeParams(nLogThetaGivenOneLabel);
  }
  if(learningInfo.mpiWorld->rank() == 0) {
    cerr << "done" << endl;
  }
}

// compute the partition function Z_\lambda(x)
// assumptions:
// - fst and betas are populated using BuildLambdaFst()
double LatentCrfModel::ComputeNLogZ_lambda(const fst::VectorFst<FstUtils::LogArc> &fst, const vector<FstUtils::LogWeight> &betas) {
  return betas[fst.Start()].Value();
}

// compute the partition function Z_\lambda(x)
double LatentCrfModel::ComputeNLogZ_lambda(unsigned sentId) {
  const vector<int> &x = data[sentId];
  fst::VectorFst<FstUtils::LogArc> fst;
  vector<FstUtils::LogWeight> alphas;
  vector<FstUtils::LogWeight> betas;
  BuildLambdaFst(sentId, fst, alphas, betas);
  return ComputeNLogZ_lambda(fst, betas);
}

// builds an FST to compute Z(x) = \sum_y \prod_i \exp \lambda h(y_i, y_{i-1}, x, i), but doesn't not compute the potentials
void LatentCrfModel::BuildLambdaFst(unsigned sentId, fst::VectorFst<FstUtils::LogArc> &fst) {
  const vector<int> &x = data[sentId];
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
	// debug info (expensive)
	if(learningInfo.debugLevel >= DebugLevel::SENTENCE) {
	  int masterLambdaCount = learningInfo.mpiWorld->rank() == 0? lambda->GetParamsCount() : -1;
	  int masterYI = learningInfo.mpiWorld->rank() == 0? yI : -1;
	  int masterYIM1 = learningInfo.mpiWorld->rank() == 0? yIM1 : -1;
	  int masterI = learningInfo.mpiWorld->rank() == 0? i : -1;
	  map<string, double> masterH;
	  if(learningInfo.mpiWorld->rank() == 0) {
	    masterH = h;
	  }
	  mpi::broadcast<int>(*learningInfo.mpiWorld, masterLambdaCount, 0);
	  mpi::broadcast<int>(*learningInfo.mpiWorld, masterYI, 0);
	  mpi::broadcast<int>(*learningInfo.mpiWorld, masterYIM1, 0);
	  mpi::broadcast<int>(*learningInfo.mpiWorld, masterI, 0);
	  mpi::broadcast< map<string, double> > (*learningInfo.mpiWorld, masterH, 0);
	  if(learningInfo.mpiWorld->rank() != 0) {
	    cerr << "rank #0: after calling lambda->FireFeatures(" << masterYI << "," << masterYIM1 << ",x," << masterI << "), |lambda| = " << masterLambdaCount << endl;
	    cerr << "rank #" << learningInfo.mpiWorld->rank() << ": after calling lambda->FireFeatures(" << yI << "," << yIM1 << ",x," << i << "), |lambda| = " << lambda->GetParamsCount() << endl;
	    if(masterLambdaCount != lambda->GetParamsCount()) {
	      cerr << "rank #0: features fired on this arc are: ";
	      for(map<string, double>::const_iterator hIter = masterH.begin(); hIter != masterH.end(); hIter++) {
		cerr << hIter->first << "->" << hIter->second << " ";
	      }
	      cerr << endl;
	      cerr << "rank #" << learningInfo.mpiWorld->rank() << ": features fired of this arc are ";
	      for(map<string, double>::const_iterator hIter = h.begin(); hIter != h.end(); hIter++) {
		cerr << hIter->first << "->" << hIter->second << " ";
	      }
	      cerr << endl;
	    }
	  }
	}
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
	    fst.SetFinal(toState, FstUtils::LogWeight::One());
	  }
	} else {
	  toState = yIToState[yI];
	}
	// now add the arc
	fst.AddArc(fromState, FstUtils::LogArc(yIM1, yI, nLambdaH, toState));
      } 
   }
    // now, that all states reached in step i have already been created, yIM1ToState has become irrelevant
    yIM1ToState = yIToState;
  }  
}

// builds an FST to compute Z(x) = \sum_y \prod_i \exp \lambda h(y_i, y_{i-1}, x, i), and computes the potentials
void LatentCrfModel::BuildLambdaFst(unsigned sentId, fst::VectorFst<FstUtils::LogArc> &fst, vector<FstUtils::LogWeight> &alphas, vector<FstUtils::LogWeight> &betas) {
  clock_t timestamp = clock();

  const vector<int> &x = data[sentId];

  // first, build the fst
  BuildLambdaFst(sentId, fst);

  // then, compute potentials
  assert(alphas.size() == 0);
  ShortestDistance(fst, &alphas, false);
  assert(betas.size() == 0);
  ShortestDistance(fst, &betas, true);

  // debug info
  if(learningInfo.debugLevel == DebugLevel::SENTENCE) {
    cerr << " BuildLambdaFst() for this sentence took " << (float) (clock() - timestamp) / CLOCKS_PER_SEC << " sec. " << endl;
  }
}

// assumptions: 
// - fst is populated using BuildLambdaFst()
// - FXk is cleared
void LatentCrfModel::ComputeF(unsigned sentId,
			      const fst::VectorFst<FstUtils::LogArc> &fst,
			      const vector<FstUtils::LogWeight> &alphas, const vector<FstUtils::LogWeight> &betas,
			      FastSparseVector<LogVal<double> > &FXk) {
  clock_t timestamp = clock();
  
  const vector<int> &x = data[sentId];

  assert(FXk.size() == 0);
  assert(fst.NumStates() > 0);
  
  // a schedule for visiting states such that we know the timestep for each arc
  set<int> iStates, iP1States;
  iStates.insert(fst.Start());

  // for each timestep
  for(int i = 0; i < x.size(); i++) {
    int xI = x[i];
    
    // from each state at timestep i
    for(set<int>::const_iterator iStatesIter = iStates.begin(); 
	iStatesIter != iStates.end(); 
	iStatesIter++) {
      int fromState = *iStatesIter;

      // for each arc leaving this state
      for(fst::ArcIterator< fst::VectorFst<FstUtils::LogArc> > aiter(fst, fromState); !aiter.Done(); aiter.Next()) {
	FstUtils::LogArc arc = aiter.Value();
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
	  if(FXk.find(h_k->first) == FXk.end()) {
	    FXk[h_k->first] = LogVal<double>(0.0);
	  }
	  FXk[h_k->first] += LogVal<double>(-1.0 * nLogMarginal, init_lnx()) * LogVal<double>(h_k->second);
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

void LatentCrfModel::FireFeatures(unsigned sentId,
				  const fst::VectorFst<FstUtils::LogArc> &fst,
				  FastSparseVector<double> &h) {
  clock_t timestamp = clock();
  
  const vector<int> &x = data[sentId];

  assert(fst.NumStates() > 0);
  
  // a schedule for visiting states such that we know the timestep for each arc
  set<int> iStates, iP1States;
  iStates.insert(fst.Start());

  // for each timestep
  for(int i = 0; i < x.size(); i++) {
    int xI = x[i];
    
    // from each state at timestep i
    for(set<int>::const_iterator iStatesIter = iStates.begin(); 
	iStatesIter != iStates.end(); 
	iStatesIter++) {
      int fromState = *iStatesIter;

      // for each arc leaving this state
      for(fst::ArcIterator< fst::VectorFst<FstUtils::LogArc> > aiter(fst, fromState); !aiter.Done(); aiter.Next()) {
	FstUtils::LogArc arc = aiter.Value();
	int yIM1 = arc.ilabel;
	int yI = arc.olabel;
	double arcWeight = arc.weight.Value();
	int toState = arc.nextstate;

	// for each feature that fires on this arc
	lambda->FireFeatures(yI, yIM1, x, i, enabledFeatureTypes, h);

	// prepare the schedule for visiting states in the next timestep
	iP1States.insert(toState);
      } 
    }

    // prepare for next timestep
    iStates = iP1States;
    iP1States.clear();
  }  

  if(learningInfo.debugLevel == DebugLevel::SENTENCE) {
    cerr << "FireFeatures() for this sentence took " << (float) (clock() - timestamp) / CLOCKS_PER_SEC << " sec." << endl;
  }
}			   

// assumptions: 
// - fst is populated using BuildThetaLambdaFst()
// - DXZk is cleared
void LatentCrfModel::ComputeD(unsigned sentId, const vector<int> &z, 
			      const fst::VectorFst<FstUtils::LogArc> &fst,
			      const vector<FstUtils::LogWeight> &alphas, const vector<FstUtils::LogWeight> &betas,
			      FastSparseVector<LogVal<double> > &DXZk) {
  clock_t timestamp = clock();

  const vector<int> &x = data[sentId];
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
      for(fst::ArcIterator< fst::VectorFst<FstUtils::LogArc> > aiter(fst, fromState); !aiter.Done(); aiter.Next()) {
	FstUtils::LogArc arc = aiter.Value();
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
	  DXZk[h_k->first] += LogVal<double>(-nLogMarginal, init_lnx()) * LogVal<double>(h_k->second);
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
}

// assumptions:
// - fst, betas are populated using BuildThetaLambdaFst()
double LatentCrfModel::ComputeNLogC(const fst::VectorFst<FstUtils::LogArc> &fst,
				    const vector<FstUtils::LogWeight> &betas) {
  double nLogC = betas[fst.Start()].Value();
  return nLogC;
}

// compute B(x,z) which can be indexed as: BXZ[y^*][z^*] to give B(x, z, z^*, y^*)
// assumptions: 
// - BXZ is cleared
// - fst, alphas, and betas are populated using BuildThetaLambdaFst
void LatentCrfModel::ComputeB(unsigned sentId, const vector<int> &z, 
			      const fst::VectorFst<FstUtils::LogArc> &fst, 
			      const vector<FstUtils::LogWeight> &alphas, const vector<FstUtils::LogWeight> &betas, 
			      map< int, map< int, LogVal<double> > > &BXZ) {
  // \sum_y [ \prod_i \theta_{z_i\mid y_i} e^{\lambda h(y_i, y_{i-1}, x, i)} ] \sum_i \delta_{y_i=y^*,z_i=z^*}
  assert(BXZ.size() == 0);

  const vector<int> &x = data[sentId];

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
      for(fst::ArcIterator< fst::VectorFst<FstUtils::LogArc> > aiter(fst, fromState); !aiter.Done(); aiter.Next()) {
	FstUtils::LogArc arc = aiter.Value();
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
	BXZ[yI][zI] += LogVal<double>(-nLogMarginal, init_lnx());

	// prepare the schedule for visiting states in the next timestep
	iP1States.insert(toState);
      } 
    }

    // prepare for next timestep
    iStates = iP1States;
    iP1States.clear();
  }
  
}

// compute B(x,z) which can be indexed as: BXZ[y^*][z^*] to give B(x, z, z^*, y^*)
// assumptions: 
// - BXZ is cleared
// - fst, alphas, and betas are populated using BuildThetaLambdaFst
void LatentCrfModel::ComputeB(unsigned sentId, const vector<int> &z, 
			      const fst::VectorFst<FstUtils::LogArc> &fst, 
			      const vector<FstUtils::LogWeight> &alphas, const vector<FstUtils::LogWeight> &betas, 
			      map< std::pair<int, int>, map< int, LogVal<double> > > &BXZ) {
  // \sum_y [ \prod_i \theta_{z_i\mid y_i} e^{\lambda h(y_i, y_{i-1}, x, i)} ] \sum_i \delta_{y_i=y^*,z_i=z^*}
  assert(BXZ.size() == 0);

  const vector<int> &x = data[sentId];

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
      for(fst::ArcIterator< fst::VectorFst<FstUtils::LogArc> > aiter(fst, fromState); !aiter.Done(); aiter.Next()) {
	FstUtils::LogArc arc = aiter.Value();
	int yIM1 = arc.ilabel;
	int yI = arc.olabel;
	double arcWeight = arc.weight.Value();
	int toState = arc.nextstate;

	// compute marginal weight of passing on this arc
	double nLogMarginal = alphas[fromState].Value() + betas[toState].Value() + arcWeight;

	// update the corresponding B value
	std::pair<int, int> yIM1AndyI = std::pair<int, int>(yIM1, yI);
	if(BXZ.count(yIM1AndyI) == 0 || BXZ[yIM1AndyI].count(zI) == 0) {
	  BXZ[yIM1AndyI][zI] = 0;
	}
	BXZ[yIM1AndyI][zI] += LogVal<double>(-nLogMarginal, init_lnx());

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

double LatentCrfModel::GetNLogTheta(int yim1, int yi, int zi) {
  if(learningInfo.zIDependsOnYIM1) {
    return nLogThetaGivenTwoLabels[pair<int,int>(yim1, yi)][zi];
  } else {
    return nLogThetaGivenOneLabel[yi][zi];
  }
}

// build an FST which path sums to 
// -log \sum_y [ \prod_i \theta_{z_i\mid y_i} e^{\lambda h(y_i, y_{i-1}, x, i)} ]
void LatentCrfModel::BuildThetaLambdaFst(unsigned sentId, const vector<int> &z, fst::VectorFst<FstUtils::LogArc> &fst, vector<FstUtils::LogWeight> &alphas, vector<FstUtils::LogWeight> &betas) {

  clock_t timestamp = clock();

  const vector<int> &x = data[sentId];

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
	
	double nLogTheta_zI_y = GetNLogTheta(yIM1, yI, zI);

	// compute the weight of this transition: \lambda h(y_i, y_{i-1}, x, i), and multiply by -1 to be consistent with the -log probability representatio
	double nLambdaH = -1.0 * lambda->DotProduct(h);
	double weight = nLambdaH + nLogTheta_zI_y;

	// determine whether to add a new state or reuse an existing state which also represent label y_i and timestep i
	int toState;	
	if(yIToState.count(yI) == 0) {
	  toState = fst.AddState();
	  yIToState[yI] = toState;
	  // is it a final state?
	  if(i == x.size() - 1) {
	    fst.SetFinal(toState, FstUtils::LogWeight::One());
	  }
	} else {
	  toState = yIToState[yI];
	}
	// now add the arc
	fst.AddArc(fromState, FstUtils::LogArc(yIM1, yI, weight, toState));	
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
double LatentCrfModel::ComputeNLogPrYZGivenX(unsigned sentId, const vector<int>& y, const vector<int>& z) {
  const vector<int>& x = data[sentId];

  assert(x.size() == y.size());
  assert(x.size() == z.size());

  // initialize prob = 1.0
  double result = 0;

  // divide by Z_\lambda(x)
  result -= ComputeNLogZ_lambda(sentId);

  for(int i = 0; i < x.size(); i++) {

    // y_{i-1}
    int yIM1 = 
      i == 0? 
      START_OF_SENTENCE_Y_VALUE:
      y[i-1];

    // multiply \theta_{z_i|y_i} (which is already stored using in its -log value)
    result += GetNLogTheta(yIM1, y[i], z[i]);

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
double LatentCrfModel::ComputeNLogPrYGivenXZ(unsigned sentId, const vector<int> &y, const vector<int> &z) {

  const vector<int> &x = data[sentId]; 

  assert(x.size() == y.size());
  assert(x.size() == z.size());

  double result = 0;

  // multiply the numerator
  for(int i = 0; i < x.size(); i++) {

    // y_{i-1}
    int yIM1 = 
      i == 0? 
      START_OF_SENTENCE_Y_VALUE:
      y[i-1];

    // multiply \theta_{z_i|y_i} (which is already stored in its -log value)
    result += GetNLogTheta(yIM1, y[i], z[i]);

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
  fst::VectorFst<FstUtils::LogArc> fst;
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
	// \theta(z_i | y_i, y_{i-1})
	double nLogTheta_zI_y = GetNLogTheta(yIM1, yI, z[i]);

	// compute the weight of this transition: -log p_\theta(z_i|y_i) -log \exp \lambda h(y_i, y_{i-1}, x, i)
	// note: parameters theta[y_{i-1}][y_i] is already in the -log representation
	double weight = lambdaH + nLogTheta_zI_y;
	// determine whether to add a new state or reuse an existing state which also represent label y_i and timestep i
	int toState;	
	if(yIToState.count(yI) == 0) {
	  toState = fst.AddState();
	  yIToState[yI] = toState;
	  // is it a final state?
	  if(i == x.size() - 1) {
	    fst.SetFinal(toState, FstUtils::LogWeight::One());
	  }
	} else {
	  toState = yIToState[yI];
	}
	// now add the arc
	fst.AddArc(fromState, FstUtils::LogArc(yIM1, yI, weight, toState));	
      }
    }
    // now, that all states reached in step i have already been created, yIM1ToState has become irrelevant
    yIM1ToState = yIToState;
  }

  //  now compute the path sum, i.e. -\log [ \sum_y' \prod_i \theta_{z_i|y'_i} \exp \lambda h(y'_i, y'_{i-1}, x, i) ]
  vector<FstUtils::LogWeight> distancesToFinal;
  ShortestDistance(fst, &distancesToFinal, true);

  //  finally, divide by the denominator
  double denominator = distancesToFinal[startState].Value();
  result -= denominator;

  // return p(y | x, z)
  return result;
}

void LatentCrfModel::SupervisedTrain(string goldLabelsFilename) {
  // encode labels
  assert(goldLabelsFilename.size() != 0);
  VocabEncoder labelsEncoder(goldLabelsFilename, FIRST_ALLOWED_LABEL_VALUE);
  labels.clear();
  labelsEncoder.Read(goldLabelsFilename, labels);
  
  // use lbfgs to fit the lambda CRF parameters
  double *lambdasArray = lambda->GetParamWeightsArray();
  unsigned lambdasArrayLength = lambda->GetParamsCount();
  lbfgs_parameter_t lbfgsParams = SetLbfgsConfig();  
  lbfgsParams.max_iterations = 10;
  lbfgsParams.m = 50;
  lbfgsParams.max_linesearch = 20;
  double optimizedNLoglikelihoodYGivenX = 0;
  int allSents = -1;
  int lbfgsStatus = lbfgs(lambdasArrayLength, lambdasArray, &optimizedNLoglikelihoodYGivenX, 
			  EvaluateNLogLikelihoodYGivenXDerivativeWRTLambda, LbfgsProgressReport, &allSents, &lbfgsParams);
  if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH && learningInfo.mpiWorld->rank() == 0) {
    cerr << "master" << learningInfo.mpiWorld->rank() << ": lbfgsStatusCode = " << LbfgsUtils::LbfgsStatusIntToString(lbfgsStatus) << " = " << lbfgsStatus << endl;
  }
  if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH) {
    cerr << "rank #" << learningInfo.mpiWorld->rank() << ": loglikelihood_{p(y|x)}(\\lambda) = " << -optimizedNLoglikelihoodYGivenX << endl;
  }
  
  // optimize theta (i.e. multinomial) parameters to maximize the likeilhood of the data
  MultinomialParams::ConditionalMultinomialParam<int> thetaMle;
  MultinomialParams::MultinomialParam thetaMleMarginals;
  // for each sentence
  for(unsigned sentId = 0; sentId < data.size(); sentId++) {
    // collect number of times each theta parameter has been used
    vector<int> &z = data[sentId];
    vector<int> &y = labels[sentId];
    assert(z.size() == y.size());
    for(unsigned i = 0; i < z.size(); i++) {
      thetaMle[y[i]][z[i]] += 1;
      thetaMleMarginals[y[i]] += 1;
    }
  }
  // normalize thetas
  NormalizeThetaMle<int>(thetaMle, thetaMleMarginals);
  nLogThetaGivenOneLabel = thetaMle;
  // compute likelihood of \theta for z|y
  double nloglikelihoodZGivenY = 0; 
  for(unsigned sentId = 0; sentId < data.size(); sentId++) {
    vector<int> &z = data[sentId];
    vector<int> &y = labels[sentId];
    for(unsigned i = 0; i < z.size(); i++){ 
      nloglikelihoodZGivenY += nLogThetaGivenOneLabel[y[i]][z[i]];
    }
  } 
  if(learningInfo.debugLevel == DebugLevel::MINI_BATCH && learningInfo.mpiWorld->rank() == 0) {
    cerr << "master" << learningInfo.mpiWorld->rank() << ": loglikelihood_{p(z|y)}(\\theta) = " << - nloglikelihoodZGivenY << endl;
    cerr << "master" << learningInfo.mpiWorld->rank() << ": loglikelihood_{p(z|x)}(\\theta, \\lambda) = " << - optimizedNLoglikelihoodYGivenX - nloglikelihoodZGivenY << endl;
  }
}

void LatentCrfModel::Train() {
  switch(learningInfo.optimizationMethod.algorithm) {
  case BLOCK_COORD_DESCENT:
  case SIMULATED_ANNEALING:
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
  return objective;
}

FastSparseVector<double> LatentCrfModel::AccumulateDerivatives(const FastSparseVector<double> &v1, const FastSparseVector<double> &v2) {
  FastSparseVector<double> vTotal(v1);
  for(FastSparseVector<double>::const_iterator v2Iter = v2.begin(); v2Iter != v2.end(); ++v2Iter) {
    vTotal[v2Iter->first] += v2Iter->second;
  }
  return vTotal;
}

double LatentCrfModel::EvaluateNLogLikelihoodYGivenXDerivativeWRTLambda(void *uselessPtr,
									const double *lambdasArray,
									double *gradient,
									const int lambdasCount,
									const double step) {
  
  LatentCrfModel &model = LatentCrfModel::GetInstance();
  
  if(model.learningInfo.debugLevel  >= DebugLevel::REDICULOUS){
    cerr << "rank #" << model.learningInfo.mpiWorld->rank() << ": entered EvaluateNLogLikelihoodYGivenXDerivativeWRTLambda" << endl;
  }

  // important note: the parameters array manipulated by liblbfgs is the same one used in lambda. so, the new weights are already in effect

  double nlogLikelihood = 0;
  FastSparseVector<double> nDerivative;
  unsigned from = 0, to = model.data.size();
  assert(model.data.size() == model.labels.size());

  // for each training example (x, y)
  for(unsigned sentId = from; sentId < to; sentId++) {
    if(sentId % model.learningInfo.mpiWorld->size() != model.learningInfo.mpiWorld->rank()) {
      continue;
    }

    if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
      cerr << "rank #" << model.learningInfo.mpiWorld->rank() << ": proessing sentId " << sentId << endl;
    }

    // Make |y| = |x|
    assert(model.data[sentId].size() == model.labels[sentId].size());
    const vector<int> &x = model.data[sentId];
    vector<int> &y = model.labels[sentId];

    // build the FSTs
    fst::VectorFst<FstUtils::LogArc> lambdaFst;
    vector<FstUtils::LogWeight> lambdaAlphas, lambdaBetas;
    model.BuildLambdaFst(sentId, lambdaFst, lambdaAlphas, lambdaBetas);

    // compute the Z value for this sentence
    double nLogZ = model.ComputeNLogZ_lambda(lambdaFst, lambdaBetas);
    if(std::isnan(nLogZ) || std::isinf(nLogZ)) {
      if(model.learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
	cerr << "ERROR: nLogZ = " << nLogZ << ". my mistake. will halt!" << endl;
      }
      assert(false);
    } 
    
    if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
      cerr << "rank #" << model.learningInfo.mpiWorld->rank() << ": nLogZ = " << nLogZ << endl;
    }

    // compute the F map fro this sentence
    FastSparseVector<LogVal<double> > FSparseVector;
    model.ComputeF(sentId, lambdaFst, lambdaAlphas, lambdaBetas, FSparseVector);
    if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
      cerr << "rank #" << model.learningInfo.mpiWorld->rank() << ": F.size = " << FSparseVector.size();
    }

    // compute feature aggregate values on the gold labels of this sentence
    FastSparseVector<double> goldFeatures;
    for(unsigned i = 0; i < x.size(); i++) {
      model.lambda->FireFeatures(y[i], i==0?model.START_OF_SENTENCE_Y_VALUE:y[i-1], x, i, model.enabledFeatureTypes, goldFeatures);
    }
    if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
      cerr << "rank #" << model.learningInfo.mpiWorld->rank() << ": size of gold features = " << goldFeatures.size() << endl; 
    }

    // update the loglikelihood
    double dotProduct = model.lambda->DotProduct(goldFeatures);
    if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
      cerr << "rank #" << model.learningInfo.mpiWorld->rank() << ": dotProduct of gold features with crf params = " << dotProduct << endl;
    }
    if(nLogZ == 0 ||  dotProduct == 0 || nLogZ - dotProduct == 0) {
      cerr << "something is wrong! tell me more about lambdaFst." << endl << "lambdaFst has " << lambdaFst.NumStates() << "states. " << endl;
      if(model.learningInfo.mpiWorld->rank() == 0) {
	cerr << "lambda parameters are: ";
	model.lambda->PrintParams();
      }
    } 
    nlogLikelihood += - dotProduct - nLogZ;
    if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
      cerr << "rank #" << model.learningInfo.mpiWorld->rank() << ": nlogLikelihood = " << nlogLikelihood << endl; 
    }

    // update the gradient
    for(FastSparseVector<LogVal<double> >::iterator fIter = FSparseVector.begin(); fIter != FSparseVector.end(); ++fIter) {
      double nLogf = fIter->second.s_? fIter->second.v_ : -fIter->second.v_; // multiply the inner logF representation by -1.
      double nFOverZ = - MultinomialParams::nExp(nLogf - nLogZ);
      if(std::isnan(nFOverZ) || std::isinf(nFOverZ)) {
	if(model.learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
	  cerr << "ERROR: nFOverZ = " << nFOverZ << ", nLogf = " << nLogf << ". my mistake. will halt!" << endl;
	}
        assert(false);
      }
      nDerivative[fIter->first] += - goldFeatures[fIter->first] - nFOverZ;
    }
    if(model.learningInfo.debugLevel >= DebugLevel::SENTENCE) {
      cerr << "rank #" << model.learningInfo.mpiWorld->rank() << ": nDerivative size = " << nDerivative.size() << endl;
    }

    if(model.learningInfo.debugLevel >= DebugLevel::MINI_BATCH) {
      cerr << ".";
    }
  }

  // write the gradient in the array 'gradient' (which is pre-allocated by the lbfgs library)
  // init gradient to zero
  for(unsigned gradientIter = 0; gradientIter < model.lambda->GetParamsCount(); gradientIter++) {
    gradient[gradientIter] = 0;
  }
  // for each active feature 
  for(FastSparseVector<double>::iterator derivativeIter = nDerivative.begin(); 
      derivativeIter != nDerivative.end(); 
      ++derivativeIter) {
    // set active feature's value in the gradient
    gradient[derivativeIter->first] = derivativeIter->second;
  }

  // accumulate nloglikelihood from all processes
  mpi::all_reduce<double>(*model.learningInfo.mpiWorld, nlogLikelihood, nlogLikelihood, std::plus<double>());
  if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS && model.learningInfo.mpiWorld->rank() == 0) {
    cerr << "master" << model.learningInfo.mpiWorld->rank() << ": nloglikelihood after all_reduce = " << nlogLikelihood << endl;
  }

  // accumulate the gradient vectors from all processes
  vector<double> gradientVector(model.lambda->GetParamsCount());
  for(unsigned gradientIter = 0; gradientIter < model.lambda->GetParamsCount(); gradientIter++) {
    gradientVector[gradientIter] = gradient[gradientIter];
  }
  //  vector<double> gradientVector(gradient, gradient + model.lambda->GetParamsCount());
  mpi::all_reduce<vector<double> >(*model.learningInfo.mpiWorld, gradientVector, gradientVector, LatentCrfModel::AggregateVectors);
  assert(gradientVector.size() == lambdasCount);
  for(int i = 0; i < gradientVector.size(); i++) {
    gradient[i] = gradientVector[i];
  }
  
  if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
    cerr << "rank #" << model.learningInfo.mpiWorld->rank() << ": exiting EvaluateNLogLikelihoodYGivenXDerivativeWRTLambda" << endl;
  }

  if(model.learningInfo.debugLevel >= DebugLevel::MINI_BATCH && model.learningInfo.mpiWorld->rank() == 0) {
    cerr << "master" << model.learningInfo.mpiWorld->rank() << ": eval(y|x) = " << nlogLikelihood << endl;
  }
  return nlogLikelihood;
}

// a call back function that computes the gradient and the nloglikelihood function for the lbfgs minimizer
double LatentCrfModel::EvaluateNLogLikelihoodDerivativeWRTLambda(void *ptrFromSentId,
								 const double *lambdasArray,
								 double *gradient,
								 const int lambdasCount,
								 const double step) {
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
    fst::VectorFst<LogPairArc> thetaLambdaAndLambdaFst;
    vector<LogPairWeight> thetaLambdaAndLambdaAlphas, thetaLambdaAndLambdaBetas;
    model.BuildThetaLambdaAndLambdaFst(model.data[sentId], model.data[sentId], thetaLambdaAndLambdaFst, 
				       thetaLambdaAndLambdaAlphas, thetaLambdaAndLambdaBetas);
    */
    // build the FSTs
    fst::VectorFst<FstUtils::LogArc> thetaLambdaFst, lambdaFst;
    vector<FstUtils::LogWeight> thetaLambdaAlphas, lambdaAlphas, thetaLambdaBetas, lambdaBetas;
    model.BuildThetaLambdaFst(sentId, model.data[sentId], thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas);
    model.BuildLambdaFst(sentId, lambdaFst, lambdaAlphas, lambdaBetas);

    // compute the D map for this sentence
    FastSparseVector<LogVal<double> > DSparseVector;
    model.ComputeD(sentId, model.data[sentId], thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas, DSparseVector);
    // compute the C value for this sentence
    double nLogC = model.ComputeNLogC(thetaLambdaFst, thetaLambdaBetas);
    if(std::isnan(nLogC) || std::isinf(nLogC)) {
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
    for(FastSparseVector<LogVal<double> >::iterator dIter = DSparseVector.begin(); dIter != DSparseVector.end(); ++dIter) {
      double nLogd = dIter->second.s_? dIter->second.v_ : -dIter->second.v_; // multiply the inner logD representation by -1.
      double dOverC = MultinomialParams::nExp(nLogd - nLogC);
      if(std::isnan(dOverC) || std::isinf(dOverC)) {
	if(model.learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
	  cerr << "ERROR: dOverC = " << dOverC << ", nLogd = " << nLogd << ". my mistake. will halt!" << endl;
	}
        assert(false);
      }
      derivativeWRTLambdaSparseVector[dIter->first] -= dOverC;
    }
    // compute the F map fro this sentence
    FastSparseVector<LogVal<double> > FSparseVector;
    model.ComputeF(sentId, lambdaFst, lambdaAlphas, lambdaBetas, FSparseVector);
    // compute the Z value for this sentence
    double nLogZ = model.ComputeNLogZ_lambda(lambdaFst, lambdaBetas);
    // update the log likelihood
    if(std::isnan(nLogZ) || std::isinf(nLogZ)) {
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
      if(std::isnan(fOverZ) || std::isinf(fOverZ)) {
	if(model.learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
	  cerr << "ERROR: fOverZ = " << nLogZ << ", nLogf = " << nLogf << ". my mistake. will halt!" << endl;
	}
	assert(false);
      }
      derivativeWRTLambdaSparseVector[fIter->first] += fOverZ;
      if(std::isnan(derivativeWRTLambdaSparseVector[fIter->first]) || std::isinf(derivativeWRTLambdaSparseVector[fIter->first])) {
	cerr << "rank #" << model.learningInfo.mpiWorld->rank() << ": ERROR: fOverZ = " << nLogZ << ", nLogf = " << nLogf << ". my mistake. will halt!" << endl;
	assert(false);
      }
    }
    if(model.learningInfo.debugLevel >= DebugLevel::MINI_BATCH) {
      cerr << ".";
    }
    if(model.learningInfo.debugLevel >= DebugLevel::SENTENCE) {
      cerr << "rank #" << model.learningInfo.mpiWorld->rank() << ": EvaluateNLogLikelihoodDerivativeWRTLambda() for this sentence took " << (float) (clock() - timestamp2) / CLOCKS_PER_SEC << " sec." << endl;
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
  if(model.learningInfo.debugLevel >= DebugLevel::MINI_BATCH && model.learningInfo.mpiWorld->rank() == 0) {
    cerr << endl << "master" << model.learningInfo.mpiWorld->rank() << ": -eval(nloglikelihood=" << nlogLikelihood << ",totalMoveAwayPenalty=" << totalMoveAwayPenalty << ")\n";
  }

  if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
    cerr << endl << " EvaluateNLogLikelihoodDerivativeWRTLambda() for this minibatch took " << (float) (clock() - timestamp) / CLOCKS_PER_SEC << " sec. " << endl;
    cerr << "listing the current values in the gradient array (no problem enountered though): " << endl;
    for(int displacedIndex = 0; displacedIndex < model.lambda->GetParamsCount() - model.countOfConstrainedLambdaParameters; displacedIndex++) {
      cerr << gradient[displacedIndex] << " ";
    } 
    cerr << endl;
  }

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
  if(model.learningInfo.debugLevel >= DebugLevel::MINI_BATCH && model.learningInfo.mpiWorld->rank() == 0) {
    cerr << "master" << model.learningInfo.mpiWorld->rank() << ": -report sents:" << from << "-" << to;
    cerr << "\tlbfgs Iteration " << k;
    cerr << ":\tobjective = " << fx;
  }
  if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
    cerr << ",\txnorm = " << xnorm;
    cerr << ",\tgnorm = " << gnorm;
    cerr << ",\tstep = " << step;
  }
  if(model.learningInfo.debugLevel >= DebugLevel::MINI_BATCH && model.learningInfo.mpiWorld->rank() == 0) {
    cerr << endl << endl;
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

// add constrained features here and set their weights by hand. those weights will not be optimized.
void LatentCrfModel::AddConstrainedFeatures() {
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
}

// reduces two sets into one
set<string> LatentCrfModel::AggregateSets(const set<string> &v1, const set<string> &v2) {
  set<string> vTotal(v2);
  for(set<string>::const_iterator v1Iter = v1.begin(); v1Iter != v1.end(); ++v1Iter) {
    vTotal.insert(*v1Iter);
  }
  return vTotal;
}

// make sure all features which may fire on this training data have a corresponding parameter in lambda (member)
void LatentCrfModel::WarmUp() {
  if(learningInfo.debugLevel >= DebugLevel::CORPUS && learningInfo.mpiWorld->rank() == 0) {
    cerr << "master" << learningInfo.mpiWorld->rank() << ": warming up..." << endl;
  }

  // only the master adds constrained features with hand-crafted weights depending on the feature type
  if(learningInfo.mpiWorld->rank() == 0) {
    // but first, make sure no features are fired yet. we need the constrained features to be 
    assert(lambda->GetParamsCount() == 0);
    AddConstrainedFeatures();
  }

  // then, each process discovers the features that may show up in their sentences.
  for(int sentId = 0; sentId < data.size(); sentId++) {

    // skip sentences not assigned to this process
    if(sentId % learningInfo.mpiWorld->size() != learningInfo.mpiWorld->rank()) {
      continue;
    }
    // debug info
    if(learningInfo.debugLevel >= DebugLevel::SENTENCE) {
      cerr << "rank #" << learningInfo.mpiWorld->rank() << ": now processing sent# " << sentId << endl;
    }
    // build the FST
    fst::VectorFst<FstUtils::LogArc> lambdaFst;
    if(learningInfo.debugLevel >= DebugLevel::SENTENCE) {
      cerr << "rank #" << learningInfo.mpiWorld->rank() << ": before calling BuildLambdaFst(), |lambda| =  " << lambda->GetParamsCount() <<", |lambdaFst| =  " << lambdaFst.NumStates() << endl;
    }
    BuildLambdaFst(sentId, lambdaFst);
    if(learningInfo.debugLevel >= DebugLevel::SENTENCE) {
      cerr << "rank #" << learningInfo.mpiWorld->rank() << ": after calling BuildLambdaFst(), |lambda| =  " << lambda->GetParamsCount() << ", |lambdaFst| =  " << lambdaFst.NumStates() << endl;
    }
    // compute the F map from this sentence (implicitly adds the fired features to lambda parameters)
    FastSparseVector<double> activeFeatures;
    activeFeatures.clear();
    FireFeatures(sentId, lambdaFst, activeFeatures);
    // debug info
    if(learningInfo.debugLevel >= DebugLevel::SENTENCE) {
      cerr << "rank #" << learningInfo.mpiWorld->rank() << ": extracted " << activeFeatures.size() << " features from sent Id " << sentId;
      for(vector<int>::const_iterator tokenIter = data[sentId].begin(); tokenIter != data[sentId].end(); tokenIter++) {
	cerr << " " << *tokenIter;
      }
      cerr << endl;
    }
  }

  if(learningInfo.debugLevel >= DebugLevel::REDICULOUS){ 
    cerr << "rank #" << learningInfo.mpiWorld->rank() << ": done with my share of FireFeatures(sent)" << endl;
  }

  stringstream removemefilename;
  removemefilename << "khaled." << learningInfo.mpiWorld->rank();
  lambda->PersistParams(removemefilename.str());

  // master collects all feature ids fired on any sentence
  set<string> localParamIds(lambda->paramIds.begin(), lambda->paramIds.end()), allParamIds;
  mpi::reduce< set<string> >(*learningInfo.mpiWorld, localParamIds, allParamIds, LatentCrfModel::AggregateSets, 0);
  
  // master updates its lambda object adding all those features
  if(learningInfo.mpiWorld->rank() == 0) {
    for(set<string>::const_iterator paramIdIter = allParamIds.begin(); paramIdIter != allParamIds.end(); ++paramIdIter) {
      lambda->AddParam(*paramIdIter);
    }
  }
  
  // master broadcasts the full set of features to all slaves
  BroadcastLambdas();
  if(learningInfo.debugLevel == DebugLevel::REDICULOUS) {
    cerr << "rank #" << learningInfo.mpiWorld->rank() << ": sent/received the lambda parameters that includes everything" << endl;
  }
  double lambdaSum = 0;
  double* lambdaWeights = lambda->GetParamWeightsArray();
  for(unsigned i = 0; i < lambda->GetParamsCount(); i++) {
    lambdaSum += lambdaWeights[i];
  }

  // debug info
  if(learningInfo.debugLevel >= DebugLevel::REDICULOUS && learningInfo.mpiWorld->rank() == 0) {
    cerr << "lambdas initialized to: " << endl;
    lambda->PrintParams();
  }
  if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH) {
      //if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH && learningInfo.mpiWorld->rank() == 0) {
    cerr << "warmup done. Lambda params count:" << lambda->GetParamsCount() << endl;
  }
}

void LatentCrfModel::UpdateThetaMleForSent(const unsigned sentId, 
					   MultinomialParams::ConditionalMultinomialParam<int> &mleGivenOneLabel, 
					   map<int, double> &mleMarginalsGivenOneLabel,
					   MultinomialParams::ConditionalMultinomialParam< pair<int, int> > &mleGivenTwoLabels, 
					   map< pair<int, int>, double> &mleMarginalsGivenTwoLabels) {
  if(learningInfo.zIDependsOnYIM1) {
    UpdateThetaMleForSent(sentId, mleGivenTwoLabels, mleMarginalsGivenTwoLabels);
  } else {
    UpdateThetaMleForSent(sentId, mleGivenOneLabel, mleMarginalsGivenOneLabel);
  }
}

void LatentCrfModel::NormalizeThetaMleAndUpdateTheta(MultinomialParams::ConditionalMultinomialParam<int> &mleGivenOneLabel, 
						     map<int, double> &mleMarginalsGivenOneLabel,
						     MultinomialParams::ConditionalMultinomialParam< std::pair<int, int> > &mleGivenTwoLabels, 
						     map< std::pair<int, int>, double> &mleMarginalsGivenTwoLabels) {
  if(learningInfo.zIDependsOnYIM1) {
    NormalizeThetaMle(mleGivenTwoLabels, mleMarginalsGivenTwoLabels);
    nLogThetaGivenTwoLabels = mleGivenTwoLabels;
  } else {
    NormalizeThetaMle(mleGivenOneLabel, mleMarginalsGivenOneLabel);
    nLogThetaGivenOneLabel = mleGivenOneLabel;
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

void LatentCrfModel::BroadcastTheta(unsigned rankId) {
  if(learningInfo.zIDependsOnYIM1) {
    mpi::broadcast< map< pair<int,int>, MultinomialParams::MultinomialParam > >(*learningInfo.mpiWorld, nLogThetaGivenTwoLabels.params, rankId);
  } else {
    mpi::broadcast< map< int, MultinomialParams::MultinomialParam > >(*learningInfo.mpiWorld, nLogThetaGivenOneLabel.params, rankId);
  }
}

void LatentCrfModel::ReduceMleAndMarginals(MultinomialParams::ConditionalMultinomialParam<int> mleGivenOneLabel, 
					   MultinomialParams::ConditionalMultinomialParam< pair<int, int> > mleGivenTwoLabels,
					   map<int, double> mleMarginalsGivenOneLabel,
					   map<std::pair<int, int>, double> mleMarginalsGivenTwoLabels) {
  if(learningInfo.zIDependsOnYIM1) {
    mpi::reduce< map< pair<int,int>, MultinomialParams::MultinomialParam > >(*learningInfo.mpiWorld, 
									     mleGivenTwoLabels.params, mleGivenTwoLabels.params, 
									     MultinomialParams::AccumulateConditionalMultinomials< pair<int, int> >, 0);
    mpi::reduce< map< pair<int, int>, double > >(*learningInfo.mpiWorld, 
					       mleMarginalsGivenTwoLabels, mleMarginalsGivenTwoLabels, 
					       MultinomialParams::AccumulateMultinomials< pair<int,int> >, 0);
  } else {
    mpi::reduce< map< int, MultinomialParams::MultinomialParam > >(*learningInfo.mpiWorld, 
								   mleGivenOneLabel.params, mleGivenOneLabel.params, 
								   MultinomialParams::AccumulateConditionalMultinomials< int >, 0);
    mpi::reduce< map< int, double > >(*learningInfo.mpiWorld, 
				      mleMarginalsGivenOneLabel, mleMarginalsGivenOneLabel, 
				      MultinomialParams::AccumulateMultinomials<int>, 0);
  }
}

void LatentCrfModel::PersistTheta(string thetaParamsFilename) {
  if(learningInfo.zIDependsOnYIM1) {
    MultinomialParams::PersistParams(thetaParamsFilename, nLogThetaGivenTwoLabels, vocabEncoder);
  } else {
    MultinomialParams::PersistParams(thetaParamsFilename, nLogThetaGivenOneLabel, vocabEncoder);
  }
}

void LatentCrfModel::BlockCoordinateDescent() {  
  
  BroadcastTheta();
  BroadcastLambdas();

  // TRAINING ITERATIONS
  bool converged = false;
  do {

    // UPDATE THETAS by normalizing soft counts (i.e. the closed form MLE solution)
    // data structure to hold theta MLE estimates
    MultinomialParams::ConditionalMultinomialParam<int> mleGivenOneLabel;
    MultinomialParams::ConditionalMultinomialParam< pair<int, int> > mleGivenTwoLabels;
    map<int, double> mleMarginalsGivenOneLabel;
    map<std::pair<int, int>, double> mleMarginalsGivenTwoLabels;

    // debug info
    if(learningInfo.debugLevel >= DebugLevel::CORPUS && learningInfo.mpiWorld->rank() == 0) {
      cerr << "updating thetas...";
    }

    // update the mle for each sentence
    for(unsigned sentId = 0; sentId < data.size(); sentId++) {
      // sentId is assigned to the process # (sentId % world.size())
      if(sentId % learningInfo.mpiWorld->size() != learningInfo.mpiWorld->rank()) {
	continue;
      }
      UpdateThetaMleForSent(sentId, mleGivenOneLabel, mleMarginalsGivenOneLabel, mleGivenTwoLabels, mleMarginalsGivenTwoLabels);
    }

    // debug info
    if(learningInfo.debugLevel >= DebugLevel::CORPUS && learningInfo.mpiWorld->rank() == 0) {
      cerr << "accumulating mle counts from slaves...";
    }

    // accumulate mle counts from slaves
    ReduceMleAndMarginals(mleGivenOneLabel, mleGivenTwoLabels, mleMarginalsGivenOneLabel, mleMarginalsGivenTwoLabels);
    
    // debug info
    if(learningInfo.debugLevel >= DebugLevel::CORPUS && learningInfo.mpiWorld->rank() == 0) {
      cerr << "now master has all mle counts; normalize...";
    }

    // normalize mle and update nLogTheta on master
    if(learningInfo.mpiWorld->rank() == 0) {
      NormalizeThetaMleAndUpdateTheta(mleGivenOneLabel, mleMarginalsGivenOneLabel, mleGivenTwoLabels, mleMarginalsGivenTwoLabels);
    }

    // debug info
    if(learningInfo.debugLevel >= DebugLevel::CORPUS && learningInfo.mpiWorld->rank() == 0) {
      cerr << "master sends the normalized thetas to all slaves...";
    }

    // update nLogTheta on slaves
    BroadcastTheta();

    // debug info
    if(learningInfo.debugLevel >= DebugLevel::CORPUS && learningInfo.mpiWorld->rank() == 0) {
      cerr << "done" << endl;
    }

    // debug info
    if(learningInfo.iterationsCount % learningInfo.persistParamsAfterNIteration == 0 && learningInfo.mpiWorld->rank() == 0) {
      stringstream thetaParamsFilename;
      thetaParamsFilename << outputPrefix << "." << learningInfo.iterationsCount;
      thetaParamsFilename << ".theta";
      if(learningInfo.debugLevel >= DebugLevel::CORPUS) {
	cerr << "persisting theta parameters after iteration " << learningInfo.iterationsCount << " at " << thetaParamsFilename.str() << endl;
      }
      PersistTheta(thetaParamsFilename.str());
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
	if(learningInfo.mpiWorld->rank() == 0 && learningInfo.debugLevel >= DebugLevel::MINI_BATCH) { 
	  cerr << "master" << learningInfo.mpiWorld->rank() << ": calling lbfgs on sents " << sentId << "-" << to << endl;
	}
      }
      if(learningInfo.optimizationMethod.subOptMethod->algorithm == LBFGS) {

	int lbfgsStatus = lbfgs(lambdasArrayLength, lambdasArray, &optimizedMiniBatchNLogLikelihood, 
				EvaluateNLogLikelihoodDerivativeWRTLambda, LbfgsProgressReport, &sentId, &lbfgsParams);
	// debug
	if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH) {
	  cerr << "rank #" << learningInfo.mpiWorld->rank() << ": lbfgsStatusCode = " << LbfgsUtils::LbfgsStatusIntToString(lbfgsStatus) << " = " << lbfgsStatus << endl;
	}
	if(learningInfo.retryLbfgsOnRoundingErrors && 
	   (lbfgsStatus == LBFGSERR_ROUNDING_ERROR || lbfgsStatus == LBFGSERR_MAXIMUMLINESEARCH)) {
	  if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH && learningInfo.mpiWorld->rank() == 0) {
	    cerr << "master: rounding error (" << lbfgsStatus << "). my gradient might be buggy." << endl << "retry..." << endl;
	  }
	  lbfgsStatus = lbfgs(lambdasArrayLength, lambdasArray, &optimizedMiniBatchNLogLikelihood,
			      EvaluateNLogLikelihoodDerivativeWRTLambda, LbfgsProgressReport, &sentId, &lbfgsParams);
	  if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH && learningInfo.mpiWorld->rank() == 0) {
	    cerr << "master: lbfgsStatusCode = " << LbfgsUtils::LbfgsStatusIntToString(lbfgsStatus) << " = " << lbfgsStatus << endl;
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
	cerr << "rank #" << learningInfo.mpiWorld->rank() << ": optimized nloglikelihood is " << optimizedMiniBatchNLogLikelihood << endl;
      }
      
      // update iteration's nloglikelihood
      if(std::isnan(optimizedMiniBatchNLogLikelihood) || std::isinf(optimizedMiniBatchNLogLikelihood)) {
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
    if(learningInfo.iterationsCount % learningInfo.persistParamsAfterNIteration == 0 && learningInfo.mpiWorld->rank() == 0) {
      lambdaParamsFilename << outputPrefix << "." << learningInfo.iterationsCount << ".lambda";
      if(learningInfo.debugLevel >= DebugLevel::CORPUS && learningInfo.mpiWorld->rank() == 0) {
	cerr << "persisting lambda parameters after iteration " << learningInfo.iterationsCount << " at " << lambdaParamsFilename.str() << endl;
      }
      lambda->PersistParams(lambdaParamsFilename.str());
    }

    // debug info
    if(learningInfo.debugLevel >= DebugLevel::CORPUS && learningInfo.mpiWorld->rank() == 0) {
      cerr << endl << "master" << learningInfo.mpiWorld->rank() << ": finished coordinate descent iteration #" << learningInfo.iterationsCount << " nloglikelihood=" << nlogLikelihood << endl;
    }
    
    // update learningInfo
    learningInfo.logLikelihood.push_back(nlogLikelihood);
    learningInfo.iterationsCount++;

    // check convergence
    if(learningInfo.mpiWorld->rank() == 0) {
      converged = learningInfo.IsModelConverged();
    }
    
    // broadcast the convergence decision
    mpi::broadcast<bool>(*learningInfo.mpiWorld, converged, 0);
  
  } while(!converged);

  // debug
  if(learningInfo.mpiWorld->rank() == 0) {
    lambda->PersistParams(outputPrefix + string(".final.lambda"));
    PersistTheta(outputPrefix + string(".final.theta"));
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
  data.push_back(tokens);
  unsigned sentId = data.size() - 1;
  fst::VectorFst<FstUtils::LogArc> fst;
  vector<FstUtils::LogWeight> alphas, betas;
  BuildThetaLambdaFst(sentId, tokens, fst, alphas, betas);
  fst::VectorFst<FstUtils::StdArc> fst2, shortestPath;
  fst::ArcMap(fst, &fst2, FstUtils::LogToTropicalMapper());
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

double LatentCrfModel::ComputeManyToOne(string &aLabelsFilename, string &bLabelsFilename) {
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
  return ClustersComparer::ComputeManyToOne(clusteringA, clusteringB);
}
