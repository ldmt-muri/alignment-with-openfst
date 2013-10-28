#include "LatentCrfModel.h"

using namespace std;
using namespace OptAlgorithm;

// singlenton instance definition and trivial initialization
LatentCrfModel* LatentCrfModel::instance = 0;
int LatentCrfModel::START_OF_SENTENCE_Y_VALUE = -100;
unsigned LatentCrfModel::NULL_POSITION = -100;

LatentCrfModel& LatentCrfModel::GetInstance() {
  if(!instance) {
    assert(false);
  }
  return *instance;
}

LatentCrfModel::~LatentCrfModel() {
  delete &lambda->types;
  delete lambda;
}

// initialize model weights to zeros
LatentCrfModel::LatentCrfModel(const string &textFilename, 
			       const string &outputPrefix, 
			       LearningInfo &learningInfo, 
			       unsigned FIRST_LABEL_ID,
			       LatentCrfModel::Task task) : gaussianSampler(0.0, 10.0),
							    UnsupervisedSequenceTaggingModel(textFilename) {
  
  
  AddEnglishClosedVocab();
  
  if(learningInfo.mpiWorld->rank() == 0) {
    vocabEncoder.PersistVocab(outputPrefix + string(".vocab"));
  }

  // all processes will now read from the .vocab file master is writing. so, lets wait for the master before we continue.
  bool syncAllProcesses;
  mpi::broadcast<bool>(*learningInfo.mpiWorld, syncAllProcesses, 0);

  lambda = new LogLinearParams(vocabEncoder);
  
  // set member variables
  this->textFilename = textFilename;
  this->outputPrefix = outputPrefix;
  this->learningInfo = learningInfo;
  this->lambda->SetLearningInfo(learningInfo);
  
  // by default, we are operating in the training (not testing) mode
  testingMode = false;

  // what task is this core being used for? pos tagging? word alignment?
  this->task = task;
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

// compute the partition function Z_\lambda(x)
// assumptions:
// - fst and betas are populated using BuildLambdaFst()
double LatentCrfModel::ComputeNLogZ_lambda(const fst::VectorFst<FstUtils::LogArc> &fst, const vector<FstUtils::LogWeight> &betas) {
  return betas[fst.Start()].Value();
}

// builds an FST to compute Z(x) = \sum_y \prod_i \exp \lambda h(y_i, y_{i-1}, x, i), but doesn't not compute the potentials
void LatentCrfModel::BuildLambdaFst(unsigned sentId, fst::VectorFst<FstUtils::LogArc> &fst) {

  PrepareExample(sentId);

  const vector<int> &x = GetObservableSequence(sentId);
  // arcs represent a particular choice of y_i at time step i
  // arc weights are -\lambda h(y_i, y_{i-1}, x, i)
  assert(fst.NumStates() == 0);
  int startState = fst.AddState();
  fst.SetStart(startState);
  int finalState = fst.AddState();
  fst.SetFinal(finalState, FstUtils::LogWeight::One());

  // map values of y_{i-1} and y_i to fst states
   boost::unordered_map<int, int> yIM1ToState, yIToState;
  assert(yIM1ToState.size() == 0);
  assert(yIToState.size() == 0);
  yIM1ToState[LatentCrfModel::START_OF_SENTENCE_Y_VALUE] = startState;

  // for each timestep
  for(int i = 0; i < x.size(); i++){

    // timestep i hasn't reached any states yet
    yIToState.clear();
    // from each state reached in the previous timestep
    for(auto prevStateIter = yIM1ToState.begin();
        prevStateIter != yIM1ToState.end();
        prevStateIter++) {

      int fromState = prevStateIter->second;
      int yIM1 = prevStateIter->first;
      // to each possible value of y_i
      for(auto yDomainIter = yDomain.begin();
          yDomainIter != yDomain.end();
          yDomainIter++) {

        int yI = *yDomainIter;
	
        // skip special classes
        if(yI == LatentCrfModel::START_OF_SENTENCE_Y_VALUE || yI == LatentCrfModel::END_OF_SENTENCE_Y_VALUE) {
          continue;
      	}

        // compute h(y_i, y_{i-1}, x, i)
        FastSparseVector<double> h;
      	FireFeatures(yI, yIM1, sentId, i, h);
        // compute the weight of this transition:
        // \lambda h(y_i, y_{i-1}, x, i), and multiply by -1 to be consistent with the -log probability representation
        double nLambdaH = -1.0 * lambda->DotProduct(h);
        // determine whether to add a new state or reuse an existing state which also represent label y_i and timestep i
        int toState;
        if(yIToState.count(yI) == 0) {
          toState = fst.AddState();
          // separate state for each previous label?
          if(learningInfo.hiddenSequenceIsMarkovian) {
            yIToState[yI] = toState;
          } else {
            // same state for all labels used for previous observation
            for(auto yDomainIter2 = yDomain.begin();
                yDomainIter2 != yDomain.end();
                yDomainIter2++) {
              yIToState[*yDomainIter2] = toState;
            }
          }
          // is it a final state?
          if(i == x.size() - 1) {
            fst.AddArc(toState, FstUtils::LogArc(FstUtils::EPSILON, FstUtils::EPSILON, FstUtils::LogWeight::One(), finalState));
          }
        } else {
      	  toState = yIToState[yI];
        }
        
        // now add the arc
        fst.AddArc(fromState, FstUtils::LogArc(yIM1, yI, nLambdaH, toState));
      } 
   
      if(!learningInfo.hiddenSequenceIsMarkovian) {
        break;
      }
    }
    // now, that all states reached in step i have already been created, yIM1ToState has become irrelevant
    yIM1ToState = yIToState;
  }  
}

// builds an FST to compute Z(x) = \sum_y \prod_i \exp \lambda h(y_i, y_{i-1}, x, i), and computes the potentials
void LatentCrfModel::BuildLambdaFst(unsigned sentId, fst::VectorFst<FstUtils::LogArc> &fst, vector<FstUtils::LogWeight> &alphas, vector<FstUtils::LogWeight> &betas) {
  clock_t timestamp = clock();

  const vector<int> &x = GetObservableSequence(sentId);

  // first, build the fst
  BuildLambdaFst(sentId, fst);

  // then, compute potentials
  assert(alphas.size() == 0);
  ShortestDistance(fst, &alphas, false);
  assert(betas.size() == 0);
  ShortestDistance(fst, &betas, true);

}

// assumptions: 
// - fst is populated using BuildLambdaFst()
// - FXk is cleared
void LatentCrfModel::ComputeF(unsigned sentId,
			      const fst::VectorFst<FstUtils::LogArc> &fst,
			      const vector<FstUtils::LogWeight> &alphas, const vector<FstUtils::LogWeight> &betas,
			      FastSparseVector<LogVal<double> > &FXk) {
  clock_t timestamp = clock();
  
  const vector<int> &x = GetObservableSequence(sentId);

  assert(FXk.size() == 0);
  assert(fst.NumStates() > 0);
  
  // a schedule for visiting states such that we know the timestep for each arc
  std::tr1::unordered_set<int> iStates, iP1States;
  iStates.insert(fst.Start());

  // for each timestep
  for(int i = 0; i < x.size(); i++) {
    int xI = x[i];
    
    // from each state at timestep i
    for(auto iStatesIter = iStates.begin(); 
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
	FireFeatures(yI, yIM1, sentId, i, h);
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
}			   

void LatentCrfModel::FireFeatures(int yI, int yIM1, unsigned sentId, int i, 
				  FastSparseVector<double> &activeFeatures) { 
  if(task == Task::POS_TAGGING) {
    // fire the pos tagger features
    assert(false); // fix the implementation of FireFeatures for pOS tagging replacing string feature ids with FeatureId structs
    //lambda->FireFeatures(yI, yIM1, GetObservableSequence(sentId), i, enabledFeatureTypes, activeFeatures);
  } else if(task == Task::WORD_ALIGNMENT) {
    // fire the word aligner features
    int firstPos = learningInfo.allowNullAlignments? NULL_POSITION : NULL_POSITION + 1;
    lambda->FireFeatures(yI, yIM1, GetObservableSequence(sentId), GetObservableContext(sentId), i, 
			 LatentCrfModel::START_OF_SENTENCE_Y_VALUE, firstPos, 
			 activeFeatures);
    assert(GetObservableSequence(sentId).size() > 0);
  } else {
    assert(false);
  }
}

void LatentCrfModel::FireFeatures(unsigned sentId,
				  const fst::VectorFst<FstUtils::LogArc> &fst,
				  FastSparseVector<double> &h) {
  clock_t timestamp = clock();
  
  const vector<int> &x = GetObservableSequence(sentId);

  assert(fst.NumStates() > 0);
  
  // a schedule for visiting states such that we know the timestep for each arc
  set<int> iStates, iP1States;
  iStates.insert(fst.Start());

  // for each timestep
  for(int i = 0; i < x.size(); i++) {
    int xI = x[i];
    
    // from each state at timestep i
    for(auto iStatesIter = iStates.begin(); 
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
	FireFeatures(yI, yIM1, sentId, i, h);

	// prepare the schedule for visiting states in the next timestep
	iP1States.insert(toState);
      } 
    }

    // prepare for next timestep
    iStates = iP1States;
    iP1States.clear();
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

  const vector<int> &x = GetObservableSequence(sentId);
  // enforce assumptions
  assert(DXZk.size() == 0);

  // schedule for visiting states such that we know the timestep for each arc
  std::tr1::unordered_set<int> iStates, iP1States;
  iStates.insert(fst.Start());

  // for each timestep
  for(int i = 0; i < x.size(); i++) {
    int xI = x[i];
    int zI = z[i];
    
    // from each state at timestep i
    for(std::tr1::unordered_set<int>::const_iterator iStatesIter = iStates.begin(); 
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
	FireFeatures(yI, yIM1, sentId, i, h);
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
			      boost::unordered_map< int, boost::unordered_map< int, LogVal<double> > > &BXZ) {
  // \sum_y [ \prod_i \theta_{z_i\mid y_i} e^{\lambda h(y_i, y_{i-1}, x, i)} ] \sum_i \delta_{y_i=y^*,z_i=z^*}
  assert(BXZ.size() == 0);

  const vector<int> &x = GetObservableSequence(sentId);

  // schedule for visiting states such that we know the timestep for each arc
  std::tr1::unordered_set<int> iStates, iP1States;
  iStates.insert(fst.Start());

  // for each timestep
  for(int i = 0; i < x.size(); i++) {
    int xI = x[i];
    int zI = z[i];
    
    // from each state at timestep i
    for(auto iStatesIter = iStates.begin(); 
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
			      boost::unordered_map< std::pair<int, int>, boost::unordered_map< int, LogVal<double> > > &BXZ) {
  // \sum_y [ \prod_i \theta_{z_i\mid y_i} e^{\lambda h(y_i, y_{i-1}, x, i)} ] \sum_i \delta_{y_i=y^*,z_i=z^*}
  assert(BXZ.size() == 0);

  const vector<int> &x = GetObservableSequence(sentId);

  // schedule for visiting states such that we know the timestep for each arc
  std::tr1::unordered_set<int> iStates, iP1States;
  iStates.insert(fst.Start());

  // for each timestep
  for(int i = 0; i < x.size(); i++) {
    int xI = x[i];
    int zI = z[i];
    
    // from each state at timestep i
    for(auto iStatesIter = iStates.begin(); 
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


double LatentCrfModel::GetNLogTheta(const pair<int,int> context, int event) {
  return nLogThetaGivenTwoLabels[context][event];
}


double LatentCrfModel::GetNLogTheta(int context, int event) {
  return nLogThetaGivenOneLabel[context][event];
}

double LatentCrfModel::GetNLogTheta(int yim1, int yi, int zi, unsigned exampleId) {
  if(task == Task::POS_TAGGING) {
    return nLogThetaGivenOneLabel[yi][zi]; 
  } else if(task == Task::WORD_ALIGNMENT) {
    vector<int> &srcSent = GetObservableContext(exampleId);
    vector<int> &tgtSent = GetObservableSequence(exampleId);
    assert(find(tgtSent.begin(), tgtSent.end(), zi) != tgtSent.end());
    unsigned FIRST_POSITION = learningInfo.allowNullAlignments? NULL_POSITION: NULL_POSITION+1;
    yi -= FIRST_POSITION;
    yim1 -= FIRST_POSITION;
    // identify and explain a pathological situation
    if(nLogThetaGivenOneLabel.params.count( srcSent[yi] ) == 0) {
      cerr << "yi = " << yi << ", srcSent[yi] == " << srcSent[yi] << \
        ", nLogThetaGivenOneLabel.params.count(" << srcSent[yi] << ")=0" << \
        " although nLogThetaGivenOneLabel.params.size() = " << \
        nLogThetaGivenOneLabel.params.size() << endl << \
        "keys available are: " << endl;
      for(auto contextIter = nLogThetaGivenOneLabel.params.begin();
          contextIter != nLogThetaGivenOneLabel.params.end();
          ++contextIter) {
        cerr << " " << contextIter->first << endl;
      }
    }
    assert(nLogThetaGivenOneLabel.params.count( srcSent[yi] ) > 0);
    return nLogThetaGivenOneLabel[ srcSent[yi] ][zi];
  } else {
    assert(false);
  }
}

// build an FST which path sums to 
// -log \sum_y [ \prod_i \theta_{z_i\mid y_i} e^{\lambda h(y_i, y_{i-1}, x, i)} ]
void LatentCrfModel::BuildThetaLambdaFst(unsigned sentId, const vector<int> &z, 
					 fst::VectorFst<FstUtils::LogArc> &fst, 
					 vector<FstUtils::LogWeight> &alphas, vector<FstUtils::LogWeight> &betas) {

  clock_t timestamp = clock();
  //  cerr << "starting LatentCrfModel::BuildThetaLambdaFst" << endl;
  PrepareExample(sentId);

  const vector<int> &x = GetObservableSequence(sentId);

  // arcs represent a particular choice of y_i at time step i
  // arc weights are -log \theta_{z_i|y_i} - \lambda h(y_i, y_{i-1}, x, i)
  assert(fst.NumStates() == 0);
  int startState = fst.AddState();
  fst.SetStart(startState);
  int finalState = fst.AddState();
  fst.SetFinal(finalState, FstUtils::LogWeight::One());
  
  // map values of y_{i-1} and y_i to fst states
  boost::unordered_map<int, int> yIM1ToState, yIToState;

  yIM1ToState[LatentCrfModel::START_OF_SENTENCE_Y_VALUE] = startState;

  // for each timestep
  for(int i = 0; i < x.size(); i++) {

    // timestep i hasn't reached any states yet
    yIToState.clear();
    // from each state reached in the previous timestep
    for(auto prevStateIter = yIM1ToState.begin();
      	prevStateIter != yIM1ToState.end();
        prevStateIter++) {

      int fromState = prevStateIter->second;
      int yIM1 = prevStateIter->first;
      // to each possible value of y_i
      for(auto yDomainIter = yDomain.begin();
          yDomainIter != yDomain.end();
          yDomainIter++) {

        int yI = *yDomainIter;

        // skip special classes
        if(yI == LatentCrfModel::START_OF_SENTENCE_Y_VALUE || yI == END_OF_SENTENCE_Y_VALUE) {
          continue;
        }

      	// compute h(y_i, y_{i-1}, x, i)
        FastSparseVector<double> h;
        FireFeatures(yI, yIM1, sentId, i, h);

        // prepare -log \theta_{z_i|y_i}
        int zI = z[i];
        
        double nLogTheta_zI_y = GetNLogTheta(yIM1, yI, zI, sentId);
        assert(!std::isnan(nLogTheta_zI_y) && !std::isinf(nLogTheta_zI_y));

        // compute the weight of this transition: \lambda h(y_i, y_{i-1}, x, i), and multiply by -1 to be consistent with the -log probability representatio
        double nLambdaH = -1.0 * lambda->DotProduct(h);
        assert(!std::isnan(nLambdaH) && !std::isinf(nLambdaH));
        double weight = nLambdaH + nLogTheta_zI_y;
      	assert(!std::isnan(weight) && !std::isinf(weight));

        // determine whether to add a new state or reuse an existing state which also represent label y_i and timestep i
        int toState;
        if(yIToState.count(yI) == 0) {
          toState = fst.AddState();
          // when each variable in the hidden sequence directly depends on the previous one:
          if(learningInfo.hiddenSequenceIsMarkovian) {
            yIToState[yI] = toState;
          } else {
            // when variables in the hidden sequence are independent given observed sequence x:
            for(auto yDomainIter2 = yDomain.begin();
                yDomainIter2 != yDomain.end();
                yDomainIter2++) {
              yIToState[*yDomainIter2] = toState;
            }
          }
          // is it a final state?
          if(i == x.size() - 1) {
            fst.AddArc(toState, FstUtils::LogArc(FstUtils::EPSILON, FstUtils::EPSILON, FstUtils::LogWeight::One(), finalState));
          }
      	} else {
          toState = yIToState[yI];
        }
        // now add the arc
        fst.AddArc(fromState, FstUtils::LogArc(yIM1, yI, weight, toState));
        
      }
      
      // if hidden labels are independent given observation, then there's only one unique state in the previous timestamp
      if(!learningInfo.hiddenSequenceIsMarkovian) {
        break;
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

}

void LatentCrfModel::SupervisedTrain(string goldLabelsFilename) {
  assert(task != Task::WORD_ALIGNMENT); // the latent variable y_ needs to be re-interpreted for the word alignment task while using mle[] or theta[]
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
  double optimizedNllYGivenX = 0;
  int allSents = -1;
  
  int lbfgsStatus = lbfgs(lambdasArrayLength, lambdasArray, &optimizedNllYGivenX, 
			  LbfgsCallbackEvalYGivenXLambdaGradient, LbfgsProgressReport, &allSents, &lbfgsParams);
  if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH && learningInfo.mpiWorld->rank() == 0) {
    cerr << "master" << learningInfo.mpiWorld->rank() << ": lbfgsStatusCode = " << LbfgsUtils::LbfgsStatusIntToString(lbfgsStatus) << " = " << lbfgsStatus << endl;
  }
  if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH) {
    cerr << "rank #" << learningInfo.mpiWorld->rank() << ": loglikelihood_{p(y|x)}(\\lambda) = " << -optimizedNllYGivenX << endl;
  }
  
  // optimize theta (i.e. multinomial) parameters to maximize the likeilhood of the data
  MultinomialParams::ConditionalMultinomialParam<int> thetaMle;
  // for each sentence
  for(unsigned sentId = 0; sentId < examplesCount; sentId++) {
    // collect number of times each theta parameter has been used
    vector<int> &x_s = GetObservableContext(sentId);
    vector<int> &z = GetObservableSequence(sentId);
    vector<int> &y = labels[sentId];
    assert(z.size() == y.size());
    for(unsigned i = 0; i < z.size(); i++) {
      if(task == Task::POS_TAGGING) {
	thetaMle[y[i]][z[i]] += 1;
      } else if(task == Task::WORD_ALIGNMENT) {
	thetaMle[ x_s[y[i]] ][ z[i] ] += 1;
      } else {
	assert(false);
      }
    }
  }
  // normalize thetas
  MultinomialParams::NormalizeParams(thetaMle, learningInfo.multinomialSymmetricDirichletAlpha, false, true);

  // update nLogThetaGivenOneLabel
  for(auto contextIter = thetaMle.params.begin();
      contextIter != thetaMle.params.end();
      ++contextIter) {
    for(MultinomialParams::MultinomialParam::const_iterator probIter = contextIter->second.begin();
        probIter != contextIter->second.end();
        ++probIter) {
      nLogThetaGivenOneLabel[contextIter->first][probIter->first] = probIter->second;
    }
  }

  // compute likelihood of \theta for z|y
  double NllZGivenY = 0; 
  for(unsigned sentId = 0; sentId < examplesCount; sentId++) {
    vector<int> &z = GetObservableSequence(sentId);
    vector<int> &y = labels[sentId];
    for(unsigned i = 0; i < z.size(); i++){ 
      int DONT_CARE = -100;
      NllZGivenY += GetNLogTheta(DONT_CARE, y[i], z[i], sentId);
    }
  } 
  if(learningInfo.debugLevel == DebugLevel::MINI_BATCH && learningInfo.mpiWorld->rank() == 0) {
    cerr << "master" << learningInfo.mpiWorld->rank() << ": loglikelihood_{p(z|y)}(\\theta) = " << - NllZGivenY << endl;
    cerr << "master" << learningInfo.mpiWorld->rank() << ": loglikelihood_{p(z|x)}(\\theta, \\lambda) = " << - optimizedNllYGivenX - NllZGivenY << endl;
  }
}

void LatentCrfModel::Train() {
  testingMode = false;
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

// when l2 is specified, the regularized objective is returned. when l1 or none is specified, the unregualrized objective is returned
double LatentCrfModel::EvaluateNll() {  
  vector<double> gradientPiece(lambda->GetParamsCount(), 0.0);
  double nllPiece = ComputeNllZGivenXAndLambdaGradient(gradientPiece, 0, examplesCount);
  double nllTotal = -1;
  mpi::all_reduce<double>(*learningInfo.mpiWorld, nllPiece, nllTotal, std::plus<double>());
  assert(nllTotal != -1);
  if(learningInfo.optimizationMethod.subOptMethod->regularizer == Regularizer::L2) {
    nllTotal = AddL2Term(nllTotal);
  }
  return nllTotal;
}

// to interface with the simulated annealing library at http://www.taygeta.com/annealing/simanneal.html
float LatentCrfModel::EvaluateNll(float *lambdasArray) {
  // singleton
  LatentCrfModel &model = LatentCrfModel::GetInstance();
  // unconstrained lambda parameters count
  unsigned lambdasCount = model.lambda->GetParamsCount();
  // which sentences to work on?
  static int fromSentId = 0;
  if(fromSentId >= model.examplesCount) {
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
  float objective = (float)LbfgsCallbackEvalZGivenXLambdaGradient(ptrFromSentId, dblLambdasArray, dummy, lambdasCount, 1.0);
  return objective;
}
 
// lbfgs' callback function for evaluating -logliklihood(y|x) and its d/d_\lambda
// this is needed for supervised training of the CRF
double LatentCrfModel::LbfgsCallbackEvalYGivenXLambdaGradient(void *uselessPtr,
							       const double *lambdasArray,
							      double *gradient,
							      const int lambdasCount,
							      const double step) {
  // this method needs to be reimplemented/modified according to https://github.com/ldmt-muri/alignment-with-openfst/issues/83
  assert(false);
  
  LatentCrfModel &model = LatentCrfModel::GetInstance();
  
  if(model.learningInfo.debugLevel  >= DebugLevel::REDICULOUS){
    cerr << "rank #" << model.learningInfo.mpiWorld->rank() << ": entered LbfgsCallbackEvalYGivenXLambdaGradient" << endl;
  }

  // important note: the parameters array manipulated by liblbfgs is the same one used in lambda. so, the new weights are already in effect

  double Nll = 0;
  FastSparseVector<double> nDerivative;
  unsigned from = 0, to = model.examplesCount;
  assert(model.examplesCount == model.labels.size());

  // for each training example (x, y)
  for(unsigned sentId = from; sentId < to; sentId++) {
    if(sentId % model.learningInfo.mpiWorld->size() != model.learningInfo.mpiWorld->rank()) {
      continue;
    }

    if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
      cerr << "rank #" << model.learningInfo.mpiWorld->rank() << ": proessing sentId " << sentId << endl;
    }

    // Make |y| = |x|
    assert(model.GetObservableSequence(sentId).size() == model.labels[sentId].size());
    const vector<int> &x = model.GetObservableSequence(sentId);
    if(x.size() > model.learningInfo.maxSequenceLength) {
      continue;
    }
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
      model.FireFeatures(y[i], i==0? LatentCrfModel::START_OF_SENTENCE_Y_VALUE:y[i-1], sentId, i, goldFeatures);
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
    Nll += - dotProduct - nLogZ;
    if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
      cerr << "rank #" << model.learningInfo.mpiWorld->rank() << ": Nll = " << Nll << endl; 
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
      if(sentId % model.learningInfo.nSentsPerDot == 0) {
	cerr << "." << model.learningInfo.mpiWorld->rank();
      }
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

  // accumulate Nll from all processes

  // the all_reduce way => Nll
  mpi::all_reduce<double>(*model.learningInfo.mpiWorld, Nll, Nll, std::plus<double>());

  
  if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS /*&& model.learningInfo.mpiWorld->rank() == 0*/) {
    cerr << "rank" << model.learningInfo.mpiWorld->rank() << ": Nll after all_reduce = " << Nll << endl;
  }

  // accumulate the gradient vectors from all processes
  vector<double> gradientVector(model.lambda->GetParamsCount());
  for(unsigned gradientIter = 0; gradientIter < model.lambda->GetParamsCount(); gradientIter++) {
    gradientVector[gradientIter] = gradient[gradientIter];
  }

  mpi::all_reduce<vector<double> >(*model.learningInfo.mpiWorld, gradientVector, gradientVector, AggregateVectors2());
  assert(gradientVector.size() == lambdasCount);
  for(int i = 0; i < gradientVector.size(); i++) {
    gradient[i] = gradientVector[i];
    assert(!std::isnan(gradient[i]) || !std::isinf(gradient[i]));
  }
  
  if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
    cerr << "rank #" << model.learningInfo.mpiWorld->rank() << ": exiting LbfgsCallbackEvalYGivenXLambdaGradient" << endl;
  }

  if(model.learningInfo.debugLevel >= DebugLevel::MINI_BATCH && model.learningInfo.mpiWorld->rank() == 0) {
    cerr << "master" << model.learningInfo.mpiWorld->rank() << ": eval(y|x) = " << Nll << endl;
  }
  return Nll;
}

// adds l2 terms to both the objective and the gradient). return value is the 
// the objective after adding the l2 term.
double LatentCrfModel::AddL2Term(const vector<double> &unregularizedGradient, 
  double *regularizedGradient, double unregularizedObjective) {
  double l2RegularizedObjective = unregularizedObjective;
  // this is where the L2 term is added to both the gradient and objective function
  assert(lambda->GetParamsCount() == unregularizedGradient.size());
  for(unsigned i = 0; i < lambda->GetParamsCount(); i++) {
    double lambda_i = lambda->GetParamWeight(i);
    regularizedGradient[i] = unregularizedGradient[i] + 2.0 * learningInfo.optimizationMethod.subOptMethod->regularizationStrength * lambda_i;
    l2RegularizedObjective += learningInfo.optimizationMethod.subOptMethod->regularizationStrength * lambda_i * lambda_i;
    assert(!std::isnan(unregularizedGradient[i]) || !std::isinf(unregularizedGradient[i]));
  } 
  return l2RegularizedObjective;
}

// adds the l2 term to the objective. return value is the the objective after adding the l2 term.
double LatentCrfModel::AddL2Term(double unregularizedObjective) {
  double l2RegularizedObjective = unregularizedObjective;
  for(unsigned i = 0; i < lambda->GetParamsCount(); i++) {
    double lambda_i = lambda->GetParamWeight(i);
    l2RegularizedObjective += learningInfo.optimizationMethod.subOptMethod->regularizationStrength * lambda_i * lambda_i;
  } 
  return l2RegularizedObjective;
}

// the callback function lbfgs calls to compute the -log likelihood(z|x) and its d/d_\lambda
// this function is not expected to be executed by any slave; only the master process with rank 0
double LatentCrfModel::LbfgsCallbackEvalZGivenXLambdaGradient(void *ptrFromSentId,
							      const double *lambdasArray,
							      double *gradient,
  							      const int lambdasCount,
							      const double step) {
  
  LatentCrfModel &model = LatentCrfModel::GetInstance();
  // only the master executes the lbfgs() call and therefore only the master is expected to come here
  assert(model.learningInfo.mpiWorld->rank() == 0);

  // important note: the parameters array manipulated by liblbfgs is the same one used in lambda. so, the new weights are already in effect

  // the master tells the slaves that he needs their help to collectively compute the gradient
  bool NEED_HELP = true;
  mpi::broadcast<bool>(*model.learningInfo.mpiWorld, NEED_HELP, 0);

  // even the master needs to process its share of sentences
  vector<double> gradientPiece(model.lambda->GetParamsCount(), 0.0), reducedGradient;
  int fromSentId = *( (int*)ptrFromSentId );
  int toSentId = min(fromSentId + model.learningInfo.optimizationMethod.subOptMethod->miniBatchSize, 
                     (int)model.examplesCount);
      
  
  double NllPiece = model.ComputeNllZGivenXAndLambdaGradient(gradientPiece, fromSentId, toSentId);
  double reducedNll = -1;

  // now, the master aggregates gradient pieces computed by the slaves
  mpi::reduce< vector<double> >(*model.learningInfo.mpiWorld, gradientPiece, reducedGradient, AggregateVectors2(), 0);
  mpi::reduce<double>(*model.learningInfo.mpiWorld, NllPiece, reducedNll, std::plus<double>(), 0);
  assert(reducedNll != -1);

  // fill in the gradient array allocated by lbfgs
  if(model.learningInfo.optimizationMethod.subOptMethod->regularizer == Regularizer::L2) {
    reducedNll = model.AddL2Term(reducedGradient, gradient, reducedNll);
  } else {
    assert(gradientPiece.size() == reducedGradient.size() && gradientPiece.size() == model.lambda->GetParamsCount());
    for(unsigned i = 0; i < model.lambda->GetParamsCount(); i++) {
      gradient[i] = reducedGradient[i];
      assert(!std::isnan(gradient[i]) || !std::isinf(gradient[i]));
    } 
  }

  if(model.learningInfo.debugLevel == DebugLevel::MINI_BATCH) {
    if(model.learningInfo.optimizationMethod.subOptMethod->regularizer == Regularizer::L2) {
      cerr << " l2 reg. objective = " << reducedNll << endl;
    } else {
      cerr << " unregularized objective = " << reducedNll << endl;	
    }
  }
  
  return reducedNll;
}

// -loglikelihood is the return value
double LatentCrfModel::ComputeNllZGivenXAndLambdaGradient(
  vector<double> &derivativeWRTLambda, int fromSentId, int toSentId) {

  //  cerr << "starting LatentCrfModel::ComputeNllZGivenXAndLambdaGradient" << endl;
  // for each sentence in this mini batch, aggregate the Nll and its derivatives across sentences
  double objective = 0;

  bool ignoreThetaTerms = this->optimizingLambda &&
    learningInfo.fixPosteriorExpectationsAccordingToPZGivenXWhileOptimizingLambdas &&
    learningInfo.iterationsCount >= 2;
  
  assert(derivativeWRTLambda.size() == lambda->GetParamsCount());
  
  // for each training example
  for(int sentId = fromSentId; sentId < toSentId; sentId++) {
    
    // sentId is assigned to the process with rank = sentId % world.size()
    if(sentId % learningInfo.mpiWorld->size() != learningInfo.mpiWorld->rank()) {
      continue;
    }

    // prune long sequences
    if( GetObservableSequence(sentId).size() > learningInfo.maxSequenceLength ) {
      continue;
    }
    
    // build the FSTs
    fst::VectorFst<FstUtils::LogArc> thetaLambdaFst, lambdaFst;
    vector<FstUtils::LogWeight> thetaLambdaAlphas, lambdaAlphas, thetaLambdaBetas, lambdaBetas;
    if(!ignoreThetaTerms) {
      BuildThetaLambdaFst(sentId, GetObservableSequence(sentId), thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas);
    }
    BuildLambdaFst(sentId, lambdaFst, lambdaAlphas, lambdaBetas);

    // compute the D map for this sentence
    FastSparseVector<LogVal<double> > DSparseVector;
    if(!ignoreThetaTerms) {
      ComputeD(sentId, GetObservableSequence(sentId), thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas, DSparseVector);
    }
    
    // compute the C value for this sentence
    double nLogC = 0;
    if(!ignoreThetaTerms) {
      nLogC = ComputeNLogC(thetaLambdaFst, thetaLambdaBetas);
    }
    if(std::isnan(nLogC) || std::isinf(nLogC)) {
      if(learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
        cerr << "ERROR: nLogC = " << nLogC << ". my mistake. will halt!" << endl;
        cerr << "thetaLambdaFst summary:" << endl;
        cerr << FstUtils::PrintFstSummary(thetaLambdaFst);
      }
      assert(false);
    }
    
    // update the loglikelihood
    if(!ignoreThetaTerms) {
      objective += nLogC;

      // add D/C to the gradient
      for(FastSparseVector<LogVal<double> >::iterator dIter = DSparseVector.begin(); 
          dIter != DSparseVector.end(); ++dIter) {
        double nLogd = dIter->second.s_? dIter->second.v_ : -dIter->second.v_; // multiply the inner logD representation by -1.
        double dOverC = MultinomialParams::nExp(nLogd - nLogC);
        if(std::isnan(dOverC) || std::isinf(dOverC)) {
          if(learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
            cerr << "ERROR: dOverC = " << dOverC << ", nLogd = " << nLogd << ". my mistake. will halt!" << endl;
          }
          assert(false);
        }
        if(derivativeWRTLambda.size() <= dIter->first) {
          cerr << "problematic feature index is " << dIter->first << " cuz derivativeWRTLambda.size() = " << derivativeWRTLambda.size() << endl;
        }
        assert(derivativeWRTLambda.size() > dIter->first);
        derivativeWRTLambda[dIter->first] -= dOverC;
      }
    }
    
    // compute the F map fro this sentence
    FastSparseVector<LogVal<double> > FSparseVector;
    ComputeF(sentId, lambdaFst, lambdaAlphas, lambdaBetas, FSparseVector);

    // compute the Z value for this sentence
    double nLogZ = ComputeNLogZ_lambda(lambdaFst, lambdaBetas);

    // keep an eye on bad numbers
    if(std::isnan(nLogZ) || std::isinf(nLogZ)) {
      if(learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
        cerr << "ERROR: nLogZ = " << nLogZ << ". my mistake. will halt!" << endl;
      }
      assert(false);
    } 

    // update the log likelihood
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
      derivativeWRTLambda[fIter->first] += fOverZ;
      if(std::isnan(derivativeWRTLambda[fIter->first]) || 
          std::isinf(derivativeWRTLambda[fIter->first])) {
        cerr << "rank #" << learningInfo.mpiWorld->rank() \
          << ": ERROR: fOverZ = " << nLogZ << ", nLogf = " << nLogf \
          << ". my mistake. will halt!" << endl;
        assert(false);
      }
    }

    // debug info
    if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH && sentId % learningInfo.nSentsPerDot == 0) {
      cerr << ".";
    }
  } // end of training examples 

  cerr << learningInfo.mpiWorld->rank() << "|";

  //  cerr << "ending LatentCrfModel::ComputeNllZGivenXAndLambdaGradient" << endl;

  return objective;
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
  
  //  cerr << "starting LatentCrfModel::LbfgsProgressReport" << endl;
  LatentCrfModel &model = LatentCrfModel::GetInstance();

  int index = *((int*)ptrFromSentId), from, to;
  if(index == -1) {
    from = 0;
    to = model.examplesCount;
  } else {
    from = index;
    to = min((int)model.examplesCount, from + model.learningInfo.optimizationMethod.subOptMethod->miniBatchSize);
  }
  
  // show progress
  if(model.learningInfo.debugLevel >= DebugLevel::MINI_BATCH /* && model.learningInfo.mpiWorld->rank() == 0*/) {
    cerr << endl << "rank" << model.learningInfo.mpiWorld->rank() << ": -report: coord-descent iteration # " << model.learningInfo.iterationsCount;
    cerr << " sents(" << from << "-" << to;
    cerr << ")\tlbfgs Iteration " << k;
    if(model.learningInfo.optimizationMethod.subOptMethod->regularizer == Regularizer::NONE) {
      cerr << ":\t";
    } else {
      cerr << ":\tregularized ";
    }
    cerr << "objective = " << fx;
  }
  if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
    cerr << ",\txnorm = " << xnorm;
    cerr << ",\tgnorm = " << gnorm;
    cerr << ",\tstep = " << step;
  }
  if(model.learningInfo.debugLevel >= DebugLevel::MINI_BATCH /* && model.learningInfo.mpiWorld->rank() == 0*/) {
    cerr << endl << endl;
  }

  if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
    cerr << "done" << endl;
  }

  //  cerr << "ending LatentCrfModel::LbfgsProgressReport" << endl;
  return 0;
}

double LatentCrfModel::UpdateThetaMleForSent(const unsigned sentId, 
					     MultinomialParams::ConditionalMultinomialParam<int> &mleGivenOneLabel, 
					     boost::unordered_map<int, double> &mleMarginalsGivenOneLabel,
					     MultinomialParams::ConditionalMultinomialParam< pair<int, int> > &mleGivenTwoLabels, 
					     boost::unordered_map< pair<int, int>, double> &mleMarginalsGivenTwoLabels) {
  
  // in the word alignment model, yDomain depends on the example
  PrepareExample(sentId);
  
  double nll = -1;
  nll = UpdateThetaMleForSent(sentId, mleGivenOneLabel, mleMarginalsGivenOneLabel);
  return nll;
}

void LatentCrfModel::NormalizeThetaMleAndUpdateTheta(
    MultinomialParams::ConditionalMultinomialParam<int> &mleGivenOneLabel, 
    boost::unordered_map<int, double> &mleMarginalsGivenOneLabel,
    MultinomialParams::ConditionalMultinomialParam< std::pair<int, int> > &mleGivenTwoLabels, 
    boost::unordered_map< std::pair<int, int>, double> &mleMarginalsGivenTwoLabels) {
  
  MultinomialParams::NormalizeParams(mleGivenOneLabel, learningInfo.multinomialSymmetricDirichletAlpha, false, true);
  nLogThetaGivenOneLabel = mleGivenOneLabel;
}


lbfgs_parameter_t LatentCrfModel::SetLbfgsConfig() {
  // lbfgs configurations
  lbfgs_parameter_t lbfgsParams;
  lbfgs_parameter_init(&lbfgsParams);
  assert(learningInfo.optimizationMethod.subOptMethod != 0);
  lbfgsParams.max_iterations = learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxIterations;
  lbfgsParams.m = learningInfo.optimizationMethod.subOptMethod->lbfgsParams.memoryBuffer;
  if(learningInfo.mpiWorld->rank() == 0 && learningInfo.debugLevel >= DebugLevel::CORPUS) {
    cerr << "rank #" << learningInfo.mpiWorld->rank() << ": m = " << lbfgsParams.m  << endl;
  }
  lbfgsParams.xtol = learningInfo.optimizationMethod.subOptMethod->lbfgsParams.precision;
  if(learningInfo.mpiWorld->rank() == 0 && learningInfo.debugLevel >= DebugLevel::CORPUS) {
    cerr << "rank #" << learningInfo.mpiWorld->rank() << ": xtol = " << lbfgsParams.xtol  << endl;
  }
  lbfgsParams.max_linesearch = learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxEvalsPerIteration;
  if(learningInfo.mpiWorld->rank() == 0 && learningInfo.debugLevel >= DebugLevel::CORPUS) {
    cerr << "rank #" << learningInfo.mpiWorld->rank() << ": max_linesearch = " << lbfgsParams.max_linesearch  << endl;
  }
  switch(learningInfo.optimizationMethod.subOptMethod->regularizer) {
  case Regularizer::L1:
    lbfgsParams.orthantwise_c = learningInfo.optimizationMethod.subOptMethod->regularizationStrength;
    if(learningInfo.mpiWorld->rank() == 0 && learningInfo.debugLevel >= DebugLevel::CORPUS) {
      cerr << "rank #" << learningInfo.mpiWorld->rank() << ": orthantwise_c = " << lbfgsParams.orthantwise_c  << endl;
    }
    // this is the only linesearch algorithm that seems to work with orthantwise lbfgs
    lbfgsParams.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
    if(learningInfo.mpiWorld->rank() == 0 && learningInfo.debugLevel >= DebugLevel::CORPUS) {
      cerr << "rank #" << learningInfo.mpiWorld->rank() << ": linesearch = " << lbfgsParams.linesearch  << endl;
    }
    break;
  case Regularizer::L2:
    // nothing to be done now. l2 is implemented in the lbfgs callback evaluate function.
    break;
  case Regularizer::NONE:
    // do nothing
    break;
  default:
    cerr << "regularizer not supported" << endl;
    assert(false);
    break;
  }

  return lbfgsParams;
}

void LatentCrfModel::BroadcastTheta(unsigned rankId) {
  if(learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
    cerr << "rank #" << learningInfo.mpiWorld->rank() << ": before calling BroadcastTheta()" << endl;
  }

  mpi::broadcast< boost::unordered_map< int, MultinomialParams::MultinomialParam > >(*learningInfo.mpiWorld, nLogThetaGivenOneLabel.params, rankId);
  
  if(learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
    cerr << "rank #" << learningInfo.mpiWorld->rank() << ": after calling BroadcastTheta()" << endl;
  }
}

void LatentCrfModel::ReduceMleAndMarginals(MultinomialParams::ConditionalMultinomialParam<int> &mleGivenOneLabel, 
					   MultinomialParams::ConditionalMultinomialParam< pair<int, int> > &mleGivenTwoLabels,
					   boost::unordered_map<int, double> &mleMarginalsGivenOneLabel,
					   boost::unordered_map<std::pair<int, int>, double> &mleMarginalsGivenTwoLabels) {
  if(learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
    cerr << "rank" << learningInfo.mpiWorld->rank() << ": before calling ReduceMleAndMarginals()" << endl;
  }
  
  mpi::reduce< boost::unordered_map< int, MultinomialParams::MultinomialParam > >(*learningInfo.mpiWorld, 
								   mleGivenOneLabel.params, mleGivenOneLabel.params, 
								   MultinomialParams::AccumulateConditionalMultinomials< int >, 0);
  mpi::reduce< boost::unordered_map< int, double > >(*learningInfo.mpiWorld, 
				      mleMarginalsGivenOneLabel, mleMarginalsGivenOneLabel, 
				      MultinomialParams::AccumulateMultinomials<int>, 0);
  
  // debug info
  if(learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
    cerr << "rank" << learningInfo.mpiWorld->rank() << ": after calling ReduceMleAndMarginals()" << endl;
  }
  
}

void LatentCrfModel::PersistTheta(string thetaParamsFilename) {
  MultinomialParams::PersistParams(thetaParamsFilename, nLogThetaGivenOneLabel, 
    vocabEncoder, true, true);
}

void LatentCrfModel::BlockCoordinateDescent() {  
  assert(lambda->IsSealed());
  
  // if you're not using mini batch, set the minibatch size to data.size()
  if(learningInfo.optimizationMethod.subOptMethod->miniBatchSize <= 0) {
    learningInfo.optimizationMethod.subOptMethod->miniBatchSize = examplesCount;
  }
  
  // set lbfgs configurations
  lbfgs_parameter_t lbfgsParams = SetLbfgsConfig();
  
  // variables used for adagrad
  vector<double> gradient(lambda->GetParamsCount());
  double *u = new double[lambda->GetParamsCount()],
    *h = new double[lambda->GetParamsCount()];
  for(int paramId = 0; paramId < lambda->GetParamsCount(); ++paramId) {
    u[paramId] = h[paramId] = 0;
  }
  int adagradIter = 1;

  // TRAINING ITERATIONS
  bool converged = false;
  do {
    
    // debug info
    if(learningInfo.debugLevel >= DebugLevel::CORPUS && learningInfo.mpiWorld->rank() == 0) {
      cerr << "master" << learningInfo.mpiWorld->rank() << ": ====================== ITERATION " << learningInfo.iterationsCount << " =====================" << endl << endl;
      cerr << "master" << learningInfo.mpiWorld->rank() << ": ========== first, update thetas using a few EM iterations: =========" << endl << endl;
    }
    
    if(learningInfo.thetaOptMethod->algorithm == EXPECTATION_MAXIMIZATION) {



      // run a few EM iterations to update thetas
      for(int emIter = 0; emIter < learningInfo.emIterationsCount; ++emIter) {
    
        lambda->GetParamsCount();
        // skip EM updates of the first block-coord-descent iteration
        //if(learningInfo.iterationsCount == 0) {
        //  break;
        //}
        
        // UPDATE THETAS by normalizing soft counts (i.e. the closed form MLE solution)
        // data structure to hold theta MLE estimates
        MultinomialParams::ConditionalMultinomialParam<int> mleGivenOneLabel;
        MultinomialParams::ConditionalMultinomialParam< pair<int, int> > mleGivenTwoLabels;
        boost::unordered_map<int, double> mleMarginalsGivenOneLabel;
        boost::unordered_map<std::pair<int, int>, double> mleMarginalsGivenTwoLabels;
        
        // update the mle for each sentence
        assert(examplesCount > 0);
        if(learningInfo.mpiWorld->rank() == 0) {
          cerr << endl << "aggregating soft counts for each theta parameter...";
        }
        double unregularizedObjective = 0;
        for(unsigned sentId = 0; sentId < examplesCount; sentId++) {
          // sentId is assigned to the process # (sentId % world.size())
          if(sentId % learningInfo.mpiWorld->size() != learningInfo.mpiWorld->rank()) {
            continue;
          }
                    
          double sentLoglikelihood = UpdateThetaMleForSent(sentId, mleGivenOneLabel, mleMarginalsGivenOneLabel, mleGivenTwoLabels, mleMarginalsGivenTwoLabels);
          unregularizedObjective += sentLoglikelihood;
          
          if(sentId % learningInfo.nSentsPerDot == 0) {
            cerr << ".";
          }
        }
        
        // debug info
        cerr << learningInfo.mpiWorld->rank() << "|";
        
        // accumulate mle counts from slaves
        ReduceMleAndMarginals(mleGivenOneLabel, mleGivenTwoLabels, mleMarginalsGivenOneLabel, mleMarginalsGivenTwoLabels);
        mpi::all_reduce<double>(*learningInfo.mpiWorld, unregularizedObjective, unregularizedObjective, std::plus<double>());
        
        double regularizedObjective = learningInfo.optimizationMethod.subOptMethod->regularizer == Regularizer::L2?
          AddL2Term(unregularizedObjective):
          unregularizedObjective;
        
        if(learningInfo.mpiWorld->rank() == 0) {
          if(learningInfo.optimizationMethod.subOptMethod->regularizer == Regularizer::L2) {
            cerr << "l2 reg. objective = " << regularizedObjective << endl;
          } else { 
            cerr << "unregularized objective = " << unregularizedObjective << endl;
          }
        }	
        
        // normalize mle and update nLogTheta on master
        if(learningInfo.mpiWorld->rank() == 0) {
          NormalizeThetaMleAndUpdateTheta(mleGivenOneLabel, mleMarginalsGivenOneLabel, 
                                          mleGivenTwoLabels, mleMarginalsGivenTwoLabels);
        }
        
        // update nLogTheta on slaves
        BroadcastTheta(0);
        
      } // end of EM iterations
      
      // debug info
      if( (learningInfo.iterationsCount % learningInfo.persistParamsAfterNIteration == 0) && (learningInfo.mpiWorld->rank() == 0) ) {
        stringstream thetaParamsFilename;
        thetaParamsFilename << outputPrefix << "." << learningInfo.iterationsCount;
        thetaParamsFilename << ".theta";
        if(learningInfo.debugLevel >= DebugLevel::CORPUS) {
          cerr << "master" << learningInfo.mpiWorld->rank() << ": persisting theta parameters in iteration " \
               << learningInfo.iterationsCount << " at " << thetaParamsFilename.str() << endl;
        }
        PersistTheta(thetaParamsFilename.str());
      }
      
      // end of if(thetaOptMethod->algorithm == EM)
    } else if (learningInfo.thetaOptMethod->algorithm == GRADIENT_DESCENT) {
      assert(learningInfo.mpiWorld->size() == 1); // this method is only supported for single-threaded runs
      
      for(int gradientDescentIter = 0; gradientDescentIter < 10; ++gradientDescentIter) {
        
        cerr << "at the beginning of gradient descent iteration " << gradientDescentIter << ", EvaluateNll() = " << EvaluateNll() << endl;
        
        MultinomialParams::ConditionalMultinomialParam<int> gradientOfNll;
        ComputeNllZGivenXThetaGradient(gradientOfNll);
        for(boost::unordered_map< int, boost::unordered_map<int, double> >::iterator yIter = nLogThetaGivenOneLabel.params.begin(); 
            yIter != nLogThetaGivenOneLabel.params.end(); 
            ++yIter) {
          double marginal = 0.0;
          for(boost::unordered_map<int, double>::iterator zIter = yIter->second.begin(); zIter != yIter->second.end(); ++zIter) {
            double oldTheta = MultinomialParams::nExp(zIter->second);
            double newTheta = oldTheta - learningInfo.thetaOptMethod->learningRate * gradientOfNll[yIter->first][zIter->first];
            if(newTheta <= 0) {
              newTheta = 0.00001;
              cerr << "^";
            }
            marginal += newTheta;
            zIter->second = newTheta;
          } // end of theta updates for a particular event
          
          // now project (i.e. renormalize)
          for(boost::unordered_map<int, double>::iterator zIter = yIter->second.begin(); zIter != yIter->second.end(); ++zIter) {
            double newTheta = zIter->second;
            double projectedNewTheta = newTheta / marginal;
            double nlogProjectedNewTheta = MultinomialParams::nLog(projectedNewTheta);
            zIter->second = nlogProjectedNewTheta;
          }
          
        } // end of theta updates for a particular context
      } // end of gradient descent iterations
      // end of if(thetaOptMethod->algorithm == GRADIENT DESCENT)
    } else {
      // other optimization methods of theta are not implemented
      assert(false);
    }
    
    // update the lambdas
    this->optimizingLambda = true;
    // debug info
    if(learningInfo.debugLevel >= DebugLevel::CORPUS && learningInfo.mpiWorld->rank() == 0) {
      cerr << endl << "master" << learningInfo.mpiWorld->rank() << ": ========== second, update lambdas ==========" << endl << endl;
    }
    
    // make a copy of the lambda weights converged to in the previous iteration to use as
    // an initialization for ADAGRAD
    auto endIterator = learningInfo.optimizationMethod.subOptMethod->algorithm == ADAGRAD?
      lambda->paramWeightsPtr->end() : lambda->paramWeightsPtr->begin();
    vector<double> prevLambdaWeights(lambda->paramWeightsPtr->begin(), endIterator);
    
    double Nll = 0;
    // note: batch == minibatch with size equals to data.size()
    for(int sentId = 0; sentId < examplesCount; sentId += learningInfo.optimizationMethod.subOptMethod->miniBatchSize) {
      
      int fromSentId = sentId;
      int toSentId = min(sentId+learningInfo.optimizationMethod.subOptMethod->miniBatchSize, (int)examplesCount);
        
      // debug info
      double optimizedMiniBatchNll = 0;
      if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH && learningInfo.mpiWorld->rank() == 0) {
        cerr << "master" << learningInfo.mpiWorld->rank() << ": optimizing lambda weights to max likelihood(z|x) for sents " \
             << fromSentId << "-" << toSentId << endl;
      }
      
      // use LBFGS to update lambdas
      if(learningInfo.optimizationMethod.subOptMethod->algorithm == LBFGS) {
        
        if(learningInfo.debugLevel >= DebugLevel::REDICULOUS && learningInfo.mpiWorld->rank() == 0) {
          cerr << "master" << learningInfo.mpiWorld->rank() << ": we'll use LBFGS to update the lambda parameters" << endl;
        }
      
        // parallelizing the lbfgs callback function is complicated
        if(learningInfo.mpiWorld->rank() == 0) {
          
          // populate lambdasArray and lambasArrayLength
          // don't optimize all parameters. only optimize unconstrained ones
          double* lambdasArray;
          int lambdasArrayLength;
          lambdasArray = lambda->GetParamWeightsArray();
          lambdasArrayLength = lambda->GetParamsCount();
          
          // only the master executes lbfgs
          int lbfgsStatus = lbfgs(lambdasArrayLength, lambdasArray, &optimizedMiniBatchNll, 
                                  LbfgsCallbackEvalZGivenXLambdaGradient, LbfgsProgressReport, &sentId, &lbfgsParams);
          
          bool NEED_HELP = false;
          mpi::broadcast<bool>(*learningInfo.mpiWorld, NEED_HELP, 0);
          
          // debug
          if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH) {
            cerr << "rank #" << learningInfo.mpiWorld->rank() << ": lbfgsStatusCode = " \
                 << LbfgsUtils::LbfgsStatusIntToString(lbfgsStatus) << " = " << lbfgsStatus << endl;
          }
          
        } else {
          
          // be loyal to your master
          while(true) {
            
            // does the master need help computing the gradient? this line always "receives" rather than broacasts
            bool masterNeedsHelp = false;
            mpi::broadcast<bool>(*learningInfo.mpiWorld, masterNeedsHelp, 0);
            if(!masterNeedsHelp) {
              break;
            }
            
            // process your share of examples
            vector<double> gradientPiece(lambda->GetParamsCount(), 0.0), dummy;
            double nllPiece = ComputeNllZGivenXAndLambdaGradient(gradientPiece, fromSentId, toSentId);
            
            // merge your gradient with other slaves
            mpi::reduce< vector<double> >(*learningInfo.mpiWorld, gradientPiece, dummy, 
                                          AggregateVectors2(), 0);
            
            // aggregate the loglikelihood computation as well
            double dummy2;
            mpi::reduce<double>(*learningInfo.mpiWorld, nllPiece, dummy2, std::plus<double>(), 0);
            
            // for debug
            if(learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
              cerr << "rank" << learningInfo.mpiWorld->rank() << ": i'm trapped in this loop, repeatedly helping master evaluate likelihood and gradient for lbfgs." << endl;
            }
          }
        } // end if master => run lbfgs() else help master
        
      } else if(learningInfo.optimizationMethod.subOptMethod->algorithm == SIMULATED_ANNEALING) {
        // use simulated annealing to optimize likelihood
        
        // populate lambdasArray and lambasArrayLength
        // don't optimize all parameters. only optimize unconstrained ones
        double* lambdasArray;
        int lambdasArrayLength;
        lambdasArray = lambda->GetParamWeightsArray();
        lambdasArrayLength = lambda->GetParamsCount();
        
        simulatedAnnealer.set_up(EvaluateNll, lambdasArrayLength);
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
      } else if (learningInfo.optimizationMethod.subOptMethod->algorithm == ADAGRAD) {
        bool adagradConverged = false;
        // in each adagrad iter
        while(!adagradConverged) {    
          // compute the loss and its gradient
          double* lambdasArray = lambda->GetParamWeightsArray();

          // process your share of examples
          vector<double> gradientPiece(lambda->GetParamsCount(), 0.0);
          double nllPiece = ComputeNllZGivenXAndLambdaGradient(gradientPiece, fromSentId, toSentId);

          // merge your gradient with other slaves
          mpi::reduce< vector<double> >(*learningInfo.mpiWorld, gradientPiece, gradient, 
                                        AggregateVectors2(), 0);

          // aggregate the loglikelihood computation as well
          mpi::reduce<double>(*learningInfo.mpiWorld, nllPiece, optimizedMiniBatchNll, std::plus<double>(), 0);

          // add l2 regularization terms to objective and gradient
          if(learningInfo.optimizationMethod.subOptMethod->regularizer == Regularizer::L2) {
            optimizedMiniBatchNll = this->AddL2Term(gradient, gradient.data(), optimizedMiniBatchNll);
          }

          // log
          if(learningInfo.mpiWorld->rank() == 0) { cerr << " -- nll = " << optimizedMiniBatchNll << endl; }
          
          // l1 strength?
          double l1 = 0.0;
          if(learningInfo.optimizationMethod.subOptMethod->regularizer == Regularizer::L1) {
            l1 = learningInfo.optimizationMethod.subOptMethod->regularizationStrength; 
          }

          // core of adagrad algorithm
          // loop over params
          for(int paramId = 0; paramId < lambda->GetParamsCount(); ++paramId) {
            // the u array accumulates the gradient across iterations
            u[paramId] += gradient[paramId];
            // the h array accumulates the squared gradient across iterations
            h[paramId] += gradient[paramId] * gradient[paramId];
            // absolute (average derivative value) of this parameter minus l1 strength
            double z = (fabs(u[paramId]) / adagradIter) - l1;
            // sign of accumulated derivative value of this parameter
            double s = u[paramId] > 0? -1 : 1;
            // update param weight
            double eta = 1000.0 / (learningInfo.iterationsCount+1);
            int miniBatchSize = toSentId - fromSentId;
            if (z > 0 && h[paramId] && gradient[paramId]) {
              lambdasArray[paramId] = prevLambdaWeights[paramId] + \
                (eta * s * z * adagradIter) / (sqrt(h[paramId]) * miniBatchSize);
            } else {
              lambdasArray[paramId] = prevLambdaWeights[paramId];
            }
          }
          
          // convergence criterion for adagrad 
          int maxAdagradIter = 1;//learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxIterations;
          adagradConverged = (adagradIter++ % maxAdagradIter == 0);
        }
        
      } else {
        assert(false);
      }
      
      // debug info
      if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH && learningInfo.mpiWorld->rank() == 0) {
        cerr << "master" << learningInfo.mpiWorld->rank() << ": optimized Nll is " << optimizedMiniBatchNll << endl;
      }
      
      // update iteration's Nll
      if(std::isnan(optimizedMiniBatchNll) || std::isinf(optimizedMiniBatchNll)) {
        if(learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
          cerr << "ERROR: optimizedMiniBatchNll = " << optimizedMiniBatchNll << ". didn't add this batch's likelihood to the total likelihood. will halt!" << endl;
        }
        assert(false);
      } else {
        Nll += optimizedMiniBatchNll;
      }
      
    } // for each minibatch
    
    // in the next block coordinate descent iteration, theta will change, which means that the function 
    // you're optimizing with adagrad will change. But the lambda parameter values will have a pretty good
    // initial value (i.e. the optimal values of the likelihood function from this iteration). when provided
    // a good initialization, and starts with an empty h,u vectors, adagrad fails to improve the objective
    // for a few iterations, until h and u aggregate enough gradient values. So, instead of zeroing h,u vectors
    // at the beginning of each optimization, we intialize them with the average value from the previous run,
    // and pretend that these gradient values are obtained during the first pass of the new optimization run, 
    // hence we set adagradIter = 2, h = average(gradient), u = average(gradient^2)
    for(int paramId = 0; paramId < lambda->GetParamsCount(); ++paramId) {
      h[paramId] /= adagradIter; 
      u[paramId] /= adagradIter;
    }
    adagradIter = 2;

    // done optimizing lambdas
    this->optimizingLambda = false;
    
    // persist updated lambda params
    stringstream lambdaParamsFilename;
    if(learningInfo.iterationsCount % learningInfo.persistParamsAfterNIteration == 0 && learningInfo.mpiWorld->rank() == 0) {
      lambdaParamsFilename << outputPrefix << "." << learningInfo.iterationsCount << ".lambda";
      if(learningInfo.debugLevel >= DebugLevel::CORPUS && learningInfo.mpiWorld->rank() == 0) {
        cerr << "persisting lambda parameters after iteration " << learningInfo.iterationsCount << " at " << lambdaParamsFilename.str() << endl;
      }
      lambda->PersistParams(lambdaParamsFilename.str(), false);
      lambdaParamsFilename << ".humane";
      lambda->PersistParams(lambdaParamsFilename.str(), true);
    }
    
    // label the first K examples from the training set (i.e. the test set)
    if(learningInfo.iterationsCount % learningInfo.invokeCallbackFunctionEveryKIterations == 0 && \
       learningInfo.endOfKIterationsCallbackFunction != 0 /* &&         \
                                                             learningInfo.mpiWorld->rank() == 0*/) {
      // call the call back function
      (*learningInfo.endOfKIterationsCallbackFunction)();
    }
    
    // debug info
    if(learningInfo.debugLevel >= DebugLevel::CORPUS && learningInfo.mpiWorld->rank() == 0) {
      cerr << endl << "master" << learningInfo.mpiWorld->rank() << ": finished coordinate descent iteration #" << learningInfo.iterationsCount << " Nll=" << Nll << endl;
    }
    
    // update learningInfo
    mpi::broadcast<double>(*learningInfo.mpiWorld, Nll, 0);
    learningInfo.logLikelihood.push_back(Nll);
    learningInfo.iterationsCount++;
    
    // check convergence
    if(learningInfo.mpiWorld->rank() == 0) {
      converged = learningInfo.IsModelConverged();
    }
    
    if(learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
      cerr << "rank" << learningInfo.mpiWorld->rank() << ": coord descent converged = " << converged << endl;
    }
    
    // broadcast the convergence decision
    mpi::broadcast<bool>(*learningInfo.mpiWorld, converged, 0);    
    
  } while(!converged);
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
  boost::unordered_map<int, boost::unordered_map<string, int> > labelToTypesAndCounts;
  boost::unordered_map<string, boost::unordered_map<int, int> > typeToLabelsAndCounts;
  for(int sentId = 0; sentId < tokens.size(); sentId++) {
    for(int i = 0; i < tokens[sentId].size(); i++) {
      labelToTypesAndCounts[labels[sentId][i]][tokens[sentId][i]]++;
      typeToLabelsAndCounts[tokens[sentId][i]][labels[sentId][i]]++;
    }
  }
  // write the number of tokens of each labels
  std::ofstream outputFile(outputFilename.c_str(), std::ios::out);
  outputFile << "# LABEL HISTOGRAM #" << endl;
  for(boost::unordered_map<int, boost::unordered_map<string, int> >::const_iterator labelIter = labelToTypesAndCounts.begin(); labelIter != labelToTypesAndCounts.end(); labelIter++) {
    outputFile << "label:" << labelIter->first;
    int totalCount = 0;
    for(boost::unordered_map<string, int>::const_iterator typeIter = labelIter->second.begin(); typeIter != labelIter->second.end(); typeIter++) {
      totalCount += typeIter->second;
    }
    outputFile << " tokenCount:" << totalCount << endl;
  }
  // write the types of each label
  outputFile << endl << "# LABEL -> TYPES:COUNTS #" << endl;
  for(boost::unordered_map<int, boost::unordered_map<string, int> >::const_iterator labelIter = labelToTypesAndCounts.begin(); labelIter != labelToTypesAndCounts.end(); labelIter++) {
    outputFile << "label:" << labelIter->first << endl << "\ttypes: " << endl;
    for(boost::unordered_map<string, int>::const_iterator typeIter = labelIter->second.begin(); typeIter != labelIter->second.end(); typeIter++) {
      outputFile << "\t\t" << typeIter->first << ":" << typeIter->second << endl;
    }
  }
  // write the labels of each type
  outputFile << endl << "# TYPE -> LABELS:COUNT #" << endl;
  for(boost::unordered_map<string, boost::unordered_map<int, int> >::const_iterator typeIter = typeToLabelsAndCounts.begin(); typeIter != typeToLabelsAndCounts.end(); typeIter++) {
    outputFile << "type:" << typeIter->first << "\tlabels: ";
    for(boost::unordered_map<int, int>::const_iterator labelIter = typeIter->second.begin(); labelIter != typeIter->second.end(); labelIter++) {
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


// make sure all features which may fire on this training data have a corresponding parameter in lambda (member)
void LatentCrfModel::InitLambda() {
  if(learningInfo.mpiWorld->rank() == 0) {
    cerr << "master" << learningInfo.mpiWorld->rank() << ": initializing lambdas..." << endl;
  }

  // then, each process discovers the features that may show up in their sentences.
  for(int sentId = 0; sentId < examplesCount; sentId++) {
    
    // skip sentences not assigned to this process
    if(sentId % learningInfo.mpiWorld->size() != learningInfo.mpiWorld->rank()) {
      continue;
    }
    
    // build the FST
    fst::VectorFst<FstUtils::LogArc> lambdaFst;
    BuildLambdaFst(sentId, lambdaFst);
  }

  if(learningInfo.mpiWorld->rank() == 0) {
    cerr << "master" << learningInfo.mpiWorld->rank() << ": each process extracted features from its respective examples. Now, master will reduce all of them...";
  }

  // master collects all feature ids fired on any sentence
  assert(!lambda->IsSealed());
  unordered_set_featureId localParamIds(lambda->paramIdsTemp.begin(), lambda->paramIdsTemp.end()), allParamIds;
  mpi::reduce< unordered_set_featureId >(*learningInfo.mpiWorld, localParamIds, allParamIds, AggregateSets2(), 0);

  if(learningInfo.mpiWorld->rank() == 0) {
    cerr << "done. |lambda| = " << allParamIds.size() << endl; 
  }

  // master updates its lambda object adding all those features
  if(learningInfo.mpiWorld->rank() == 0) {
    for(auto paramIdIter = allParamIds.begin(); paramIdIter != allParamIds.end(); ++paramIdIter) {
      lambda->AddParam(*paramIdIter);
    }
  }
  
  // master seals his lambda params creating shared memory 
  if(learningInfo.mpiWorld->rank() == 0) {
    assert(lambda->paramIdsTemp.size() == lambda->paramWeightsTemp.size());
    assert(lambda->paramIdsTemp.size() > 0);
    assert(lambda->paramIdsTemp.size() == lambda->paramIndexes.size());
    assert(lambda->paramIdsPtr == 0 && lambda->paramWeightsPtr == 0);
    lambda->Seal(true);
    assert(lambda->paramIdsTemp.size() == 0 && lambda->paramWeightsTemp.size() == 0);
    assert(lambda->paramIdsPtr != 0 && lambda->paramWeightsPtr != 0);
    assert(lambda->paramIdsPtr->size() == lambda->paramWeightsPtr->size() && \
           lambda->paramIdsPtr->size() == lambda->paramIndexes.size());
  }

  // paramIndexes is out of sync. master must send it
  mpi::broadcast<unordered_map_featureId_int>(*learningInfo.mpiWorld, lambda->paramIndexes, 0);

  // slaves seal their lambda params, consuming the shared memory created by master
  if(learningInfo.mpiWorld->rank() != 0) {
    assert(lambda->paramIdsTemp.size() == lambda->paramWeightsTemp.size());
    assert(lambda->paramIdsTemp.size() > 0);
    assert(lambda->paramIdsPtr == 0 && lambda->paramWeightsPtr == 0);
    lambda->Seal(false);
    assert(lambda->paramIdsTemp.size() == 0 && lambda->paramWeightsTemp.size() == 0);
    assert(lambda->paramIdsPtr != 0 && lambda->paramWeightsPtr != 0 \
           && lambda->paramIdsPtr->size() == lambda->paramWeightsPtr->size() \
           && lambda->paramIdsPtr->size() == lambda->paramIndexes.size());    
  }
}

// returns -log p(z|x)
double LatentCrfModel::UpdateThetaMleForSent(const unsigned sentId, 
					     MultinomialParams::ConditionalMultinomialParam<int> &mle, 
					     boost::unordered_map<int, double> &mleMarginals) {
  if(learningInfo.debugLevel >= DebugLevel::SENTENCE) {
    std::cerr << "sentId = " << sentId << endl;
  }
  assert(sentId < examplesCount);
  // build the FSTs
  fst::VectorFst<FstUtils::LogArc> thetaLambdaFst;
  fst::VectorFst<FstUtils::LogArc> lambdaFst;
  std::vector<FstUtils::LogWeight> thetaLambdaAlphas, lambdaAlphas, thetaLambdaBetas, lambdaBetas;
  BuildThetaLambdaFst(sentId, GetObservableSequence(sentId), thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas);
  BuildLambdaFst(sentId, lambdaFst, lambdaAlphas, lambdaBetas);
  // compute the B matrix for this sentence
  boost::unordered_map< int, boost::unordered_map< int, LogVal<double> > > B;
  B.clear();
  ComputeB(sentId, this->GetObservableSequence(sentId), thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas, B);
  // compute the C value for this sentence
  double nLogC = ComputeNLogC(thetaLambdaFst, thetaLambdaBetas);
  //  cerr << "nLogC=" << nLogC << endl;
  double nLogZ = ComputeNLogZ_lambda(lambdaFst, lambdaBetas);
  double nLogP_ZGivenX = nLogC - nLogZ;
  // update mle for each z^*|y^* fired
  for(typename boost::unordered_map< int, boost::unordered_map<int, LogVal<double> > >::const_iterator yIter = B.begin(); yIter != B.end(); yIter++) {
    int context = GetContextOfTheta(sentId, yIter->first);
    for(boost::unordered_map<int, LogVal<double> >::const_iterator zIter = yIter->second.begin(); zIter != yIter->second.end(); zIter++) {
      int z_ = zIter->first;
      double nLogb = -log<double>(zIter->second);
      assert(zIter->second.s_ == false); //  all B values are supposed to be positive
      double bOverC = MultinomialParams::nExp(nLogb - nLogC);
      assert(bOverC > -0.001);
      mle[context][z_] += bOverC;
      mleMarginals[context] += bOverC;
      //      cerr << "-log(b[" << vocabEncoder.Decode(context) << "][" << vocabEncoder.Decode(z_) << "]) = -log(b[" << yIter->first << "][" << z_  << "]) = " << nLogb << endl;
      //      cerr << "bOverC[" << context << "][" << z_ << "] += " << bOverC << endl;
    }
  }
  return nLogP_ZGivenX;
}

// returns -log p(z|x)
// TODO: we don't need the lambdaFst. the return value of this function is just used for debugging.
double LatentCrfModel::UpdateThetaMleForSent(const unsigned sentId, 
					     MultinomialParams::ConditionalMultinomialParam<pair<int,int> > &mle, 
					     boost::unordered_map< pair<int, int> , double> &mleMarginals) {
  if(learningInfo.debugLevel >= DebugLevel::SENTENCE) {
    std::cerr << "sentId = " << sentId << endl;
  }
  assert(sentId < examplesCount);
  // build the FSTs
  fst::VectorFst<FstUtils::LogArc> thetaLambdaFst;
  fst::VectorFst<FstUtils::LogArc> lambdaFst;
  std::vector<FstUtils::LogWeight> thetaLambdaAlphas, lambdaAlphas, thetaLambdaBetas, lambdaBetas;
  BuildThetaLambdaFst(sentId, GetObservableSequence(sentId), thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas);
  BuildLambdaFst(sentId, lambdaFst, lambdaAlphas, lambdaBetas);
  // compute the B matrix for this sentence
  boost::unordered_map< pair<int, int>, boost::unordered_map< int, LogVal<double> > > B;
  B.clear();
  ComputeB(sentId, this->GetObservableSequence(sentId), thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas, B);
  // compute the C value for this sentence
  double nLogC = ComputeNLogC(thetaLambdaFst, thetaLambdaBetas);
  //  cerr << "C = " << MultinomialParams::nExp(nLogC) << endl;
  double nLogZ = ComputeNLogZ_lambda(lambdaFst, lambdaBetas);
  double nLogP_ZGivenX = nLogC - nLogZ;
  //cerr << "nloglikelihood += " << nLogC << endl;
  // update mle for each z^*|y^* fired
  for(typename boost::unordered_map< pair<int, int>, boost::unordered_map<int, LogVal<double> > >::const_iterator yIter = B.begin(); yIter != B.end(); yIter++) {
  const pair<int, int> &y_ = yIter->first;
    for(boost::unordered_map<int, LogVal<double> >::const_iterator zIter = yIter->second.begin(); zIter != yIter->second.end(); zIter++) {
      int z_ = zIter->first;
      double nLogb = -log<double>(zIter->second);
      assert(zIter->second.s_ == false); //  all B values are supposed to be positive
      double bOverC = MultinomialParams::nExp(nLogb - nLogC);
      assert(bOverC > -0.001);
      mle[y_][z_] += bOverC;
      mleMarginals[y_] += bOverC;
    }
  }
  return nLogP_ZGivenX;
}
