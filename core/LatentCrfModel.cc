#include "LatentCrfModel.h"

namespace mpi = boost::mpi;
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

void LatentCrfModel::EncodeTgtWordClasses() {
  if(learningInfo.mpiWorld->rank() != 0 || learningInfo.tgtWordClassesFilename.size() == 0) { return; }
  std::ifstream infile(learningInfo.tgtWordClassesFilename.c_str());
  string classString, wordString;
  int frequency;
  while(infile >> classString >> wordString >> frequency) {
    vocabEncoder.Encode(classString);
    vocabEncoder.Encode(wordString);
  }
  vocabEncoder.Encode("?");
}

vector<int64_t> LatentCrfModel::GetTgtWordClassSequence(vector<int64_t> &x_t) {
  assert(learningInfo.tgtWordClassesFilename.size() > 0);
  vector<int64_t> classSequence;
  for(auto tgtToken = x_t.begin(); tgtToken != x_t.end(); tgtToken++) {
    if( tgtWordToClass.count(*tgtToken) == 0 ) {
      classSequence.push_back( vocabEncoder.ConstEncode("?") );
    } else {
      classSequence.push_back( tgtWordToClass[*tgtToken] );
    }
  }
  return classSequence;
}

void LatentCrfModel::LoadTgtWordClasses(std::vector<std::vector<int64_t> > &tgtSents) {
  // read the word class file and store it in a map
  if(learningInfo.tgtWordClassesFilename.size() == 0) { return; }
  tgtWordToClass.clear();
  std::ifstream infile(learningInfo.tgtWordClassesFilename.c_str());
  string classString, wordString;
  int frequency;
  while(infile >> classString >> wordString >> frequency) {
    int64_t wordClass = vocabEncoder.ConstEncode(classString);
    int64_t wordType = vocabEncoder.ConstEncode(wordString);
    tgtWordToClass[wordType] = wordClass;
  }
  infile.close();
  
  // now read each tgt sentence and create a corresponding sequence of tgt word clusters
  for(auto tgtSent = tgtSents.begin(); tgtSent != tgtSents.end(); tgtSent++) {
    classTgtSents.push_back( GetTgtWordClassSequence(*tgtSent) );
  }
  
  if(learningInfo.mpiWorld->rank() == 0) {
    cerr << "master: finished reading " << learningInfo.tgtWordClassesFilename << ". now, classTgtSents.size() = " << classTgtSents.size() << ", tgtWordToClass.size() = " << tgtWordToClass.size() << endl;
  }

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
			       LatentCrfModel::Task task) : UnsupervisedSequenceTaggingModel(textFilename, learningInfo),
                                          learningInfo(learningInfo),
                                          gaussianSampler(0.0, 10.0) {
  
  
  //AddEnglishClosedVocab();
  
  // all processes will now read from the .vocab file master is writing. so, lets wait for the master before we continue.
  bool syncAllProcesses;
  mpi::broadcast<bool>(*learningInfo.mpiWorld, syncAllProcesses, 0);

  lambda = new LogLinearParams(vocabEncoder);

  // set member variables
  this->textFilename = textFilename;
  this->outputPrefix = outputPrefix;
  this->learningInfo = learningInfo;
  this->lambda->SetLearningInfo(learningInfo, (task==POS_TAGGING));

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
void LatentCrfModel::BuildLambdaFst(unsigned sentId, fst::VectorFst<FstUtils::LogArc> &fst, vector<double> *derivativeWRTLambda, double *objective) {

  //cerr << "Inside BuildLambdaFst(sentId=" << sentId << "...)" << endl;

  PrepareExample(sentId);

  const vector<int64_t> &x = GetObservableSequence(sentId);
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
  for(unsigned i = 0; i < x.size(); i++){

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

        // also, if this observation appears in a tag dictionary, we only allow the corresponding word classes
        //if(tagDict.count(x[i]) > 0 && tagDict[x[i]].count(yI) == 0) {
        //  continue;
        //}

        // compute h(y_i, y_{i-1}, x, i)
        FastSparseVector<double> h;
        FireFeatures(yI, yIM1, sentId, i, h);
        
        // compute the weight of this transition:
        // \lambda h(y_i, y_{i-1}, x, i), and multiply by -1 to be consistent with the -log probability representation
        double nLambdaH = -1.0 * lambda->DotProduct(h);
        // determine whether to add a new state or reuse an existing state which also represent label y_i and timestep i

        // WeightedL2
        AddWeightedL2Term(derivativeWRTLambda, objective, h);

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
void LatentCrfModel::BuildLambdaFst(unsigned sentId, fst::VectorFst<FstUtils::LogArc> &fst, vector<FstUtils::LogWeight> &alphas, vector<FstUtils::LogWeight> &betas, vector<double> *derivativeWRTLambda, double *objective) {
  
  // first, build the fst
  BuildLambdaFst(sentId, fst, derivativeWRTLambda, objective);

  // then, compute potentials
  assert(alphas.size() == 0);
  ShortestDistance(fst, &alphas, false);
  assert(betas.size() == 0);
  ShortestDistance(fst, &betas, true);

}

// assumptions: 
// - fst is populated using BuildLambdaFst()
// - FXk is cleared
void LatentCrfModel::ComputeFOverZ(unsigned sentId,
			      const fst::VectorFst<FstUtils::LogArc> &fst,
			      const vector<FstUtils::LogWeight> &alphas, const vector<FstUtils::LogWeight> &betas,
			      FastSparseVector<double> &FOverZk) {

  const vector<int64_t> &x = GetObservableSequence(sentId);

  assert(FOverZk.size() == 0);
  assert(fst.NumStates() > 0);
  
  double nLogZ = ComputeNLogZ_lambda(fst, betas);

  // a schedule for visiting states such that we know the timestep for each arc
  std::tr1::unordered_set<int> iStates, iP1States;
  iStates.insert(fst.Start());
  
  // for each timestep
  for(unsigned i = 0; i < x.size(); i++) {
    
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
          if(FOverZk.find(h_k->first) == FOverZk.end()) {
            FOverZk[h_k->first] = 0.0;
          }
          double arcNLogProb = nLogMarginal - nLogZ;
          double arcProb = MultinomialParams::nExp(arcNLogProb);
          FOverZk[h_k->first] += h_k->second * arcProb;
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

void LatentCrfModel::FireFeatures(const unsigned sentId,
                                  FastSparseVector<double> &h) {
  fst::VectorFst<FstUtils::LogArc> fst;
  vector<double> derivativeWRTLambda;
  double objective;
  BuildLambdaFst(sentId, fst, &derivativeWRTLambda, &objective);
  cerr << "Error: Method not properly implemented. h is not populated." << endl;
  assert(false);
}

void LatentCrfModel::FireFeatures(unsigned sentId,
				  const fst::VectorFst<FstUtils::LogArc> &fst,
				  FastSparseVector<double> &h) {
  //clock_t timestamp = clock();
  
  const vector<int64_t> &x = GetObservableSequence(sentId);

  assert(fst.NumStates() > 0);
  
  // a schedule for visiting states such that we know the timestep for each arc
  set<int> iStates, iP1States;
  iStates.insert(fst.Start());

  // for each timestep
  for(unsigned i = 0; i < x.size(); i++) {
    
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
void LatentCrfModel::ComputeDOverC(unsigned sentId, const vector<int64_t> &z, 
			      const fst::VectorFst<FstUtils::LogArc> &fst,
			      const vector<FstUtils::LogWeight> &alphas, const vector<FstUtils::LogWeight> &betas,
			      FastSparseVector<double> &DOverCk) {
  //clock_t timestamp = clock();

  const vector<int64_t> &x = GetObservableSequence(sentId);
  // enforce assumptions
  assert(DOverCk.size() == 0);

  double nLogC = ComputeNLogC(fst, betas);
  
  // schedule for visiting states such that we know the timestep for each arc
  std::tr1::unordered_set<int> iStates, iP1States;
  iStates.insert(fst.Start());
  
  // for each timestep
  for(unsigned i = 0; i < x.size(); i++) {
    
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
          if(DOverCk.find(h_k->first) == DOverCk.end()) {
            DOverCk[h_k->first] = 0;
          }
          double arcNLogProb = nLogMarginal - nLogC;
          double arcProb = MultinomialParams::nExp(arcNLogProb);
          DOverCk[h_k->first] += h_k->second * arcProb;
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
void LatentCrfModel::ComputeB(unsigned sentId, const vector<int64_t> &z, 
			      const fst::VectorFst<FstUtils::LogArc> &fst, 
			      const vector<FstUtils::LogWeight> &alphas, const vector<FstUtils::LogWeight> &betas, 
			      boost::unordered_map< int64_t, boost::unordered_map< int64_t, LogVal<double> > > &BXZ) {
  // \sum_y [ \prod_i \theta_{z_i\mid y_i} e^{\lambda h(y_i, y_{i-1}, x, i)} ] \sum_i \delta_{y_i=y^*,z_i=z^*}
  assert(BXZ.size() == 0);

  const vector<int64_t> &x = GetObservableSequence(sentId);

  // schedule for visiting states such that we know the timestep for each arc
  std::tr1::unordered_set<int> iStates, iP1States;
  iStates.insert(fst.Start());

  // for each timestep
  for(unsigned i = 0; i < x.size(); i++) {
    int64_t zI = z[i];
    
    // from each state at timestep i
    for(auto iStatesIter = iStates.begin(); 
	iStatesIter != iStates.end(); 
	iStatesIter++) {
      int fromState = *iStatesIter;

      // for each arc leaving this state
      for(fst::ArcIterator< fst::VectorFst<FstUtils::LogArc> > aiter(fst, fromState); !aiter.Done(); aiter.Next()) {
	FstUtils::LogArc arc = aiter.Value();
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
void LatentCrfModel::ComputeB(unsigned sentId, const vector<int64_t> &z, 
			      const fst::VectorFst<FstUtils::LogArc> &fst, 
			      const vector<FstUtils::LogWeight> &alphas, const vector<FstUtils::LogWeight> &betas, 
			      boost::unordered_map< std::pair<int64_t, int64_t>, boost::unordered_map< int64_t, LogVal<double> > > &BXZ) {
  // \sum_y [ \prod_i \theta_{z_i\mid y_i} e^{\lambda h(y_i, y_{i-1}, x, i)} ] \sum_i \delta_{y_i=y^*,z_i=z^*}
  assert(BXZ.size() == 0);

  const vector<int64_t> &x = GetObservableSequence(sentId);

  // schedule for visiting states such that we know the timestep for each arc
  std::tr1::unordered_set<int> iStates, iP1States;
  iStates.insert(fst.Start());

  // for each timestep
  for(unsigned i = 0; i < x.size(); i++) {
    int64_t zI = z[i];
    
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


double LatentCrfModel::GetNLogTheta(const pair<int64_t,int64_t> context, int64_t event) {
  return nLogThetaGivenTwoLabels[context][event];
}


double LatentCrfModel::GetNLogTheta(int64_t context, int64_t event) {
  return nLogThetaGivenOneLabel[context][event];
}

double LatentCrfModel::GetNLogTheta(int yim1, int yi, int64_t zi, unsigned exampleId) {
  if(task == Task::POS_TAGGING) {
    return nLogThetaGivenOneLabel[yi][zi]; 
  } else if(task == Task::WORD_ALIGNMENT) {
    vector<int64_t> &srcSent = GetObservableContext(exampleId);
    vector<int64_t> &reconstructedSent = GetReconstructedObservableSequence(exampleId);
    assert(find(reconstructedSent.begin(), reconstructedSent.end(), zi) != reconstructedSent.end());
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
void LatentCrfModel::BuildThetaLambdaFst(unsigned sentId, const vector<int64_t> &z, 
					 fst::VectorFst<FstUtils::LogArc> &fst, 
					 vector<FstUtils::LogWeight> &alphas, vector<FstUtils::LogWeight> &betas) {

  //clock_t timestamp = clock();
  PrepareExample(sentId);

  const vector<int64_t> &x = GetObservableSequence(sentId);

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
  for(unsigned i = 0; i < x.size(); i++) {

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

        // also, if this observation appears in a tag dictionary, we only allow the corresponding word classes
        if(tagDict.count(x[i]) > 0 && tagDict[x[i]].count(yI) == 0) {
          continue;
        }

      	// compute h(y_i, y_{i-1}, x, i)
        FastSparseVector<double> h;
        FireFeatures(yI, yIM1, sentId, i, h);

        // prepare -log \theta_{z_i|y_i}
        int64_t zI = z[i];
        
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

void LatentCrfModel::SupervisedTrainTheta() {
  cerr << "void LatentCrfModel::SupervisedTrainTheta() is not implemented" << endl;
  assert(false);
}

void LatentCrfModel::SupervisedTrain(bool fitLambdas, bool fitThetas) {
  
  if(fitLambdas) {

    if(learningInfo.mpiWorld->rank() == 0) {
      cerr << "started supervised training of lambda parameters.. " << endl;
    }
    
    // use lbfgs to fit the lambda CRF parameters
    // parallelizing the lbfgs callback function is complicated
    double nll;
    if(learningInfo.mpiWorld->rank() == 0) {

      // populate lambdasArray and lambasArrayLength
      double* lambdasArray;
      int lambdasArrayLength;
      lambdasArray = lambda->GetParamWeightsArray();
      lambdasArrayLength = lambda->GetParamsCount();

      if(learningInfo.mpiWorld->rank() == 0) {
        cerr << "will start LBFGS " <<  " at " << time(0) << endl;    
      }

      // check the analytic gradient computation by computing the derivatives numerically 
      // using method of finite differenes for a subset of the features
      int testIndexesCount = 20;
      double epsilon = 0.00000001;
      int granularities = 1;
      vector<int> testIndexes;
      if(false && learningInfo.checkGradient) {
        testIndexes = lambda->SampleFeatures(testIndexesCount);
        cerr << "calling CheckGradient() *before* running lbfgs for supervised training" << endl; 
        for(int granularity = 0; granularity < granularities; epsilon /= 10, granularity++) {
          CheckGradient(LbfgsCallbackEvalYGivenXLambdaGradient, testIndexes, epsilon);
        }
      }
      
      // only the master executes lbfgs
      int dummy = 0;
      bool supervised=true;
      lbfgs_parameter_t lbfgsParams = SetLbfgsConfig(supervised);
      if(learningInfo.mpiWorld->rank() == 0) {
        PrintLbfgsConfig(lbfgsParams);
      }
      
      int lbfgsStatus = lbfgs(lambdasArrayLength, lambdasArray, &nll, 
                              LbfgsCallbackEvalYGivenXLambdaGradient, LbfgsProgressReport, &dummy, &lbfgsParams);

      // check the analytic gradient computation by computing the derivatives numerically 
      // using method of finite differenes for a subset of the features
      if(false && learningInfo.checkGradient) {
        cerr << "calling CheckGradient() *after* running lbfgs for supervised training" << endl; 
        for(int granularity = 0; granularity < granularities; epsilon /= 10, granularity++) {
          CheckGradient(LbfgsCallbackEvalYGivenXLambdaGradient, testIndexes, epsilon);
        }
      }
      
      bool NEED_HELP = false;
      mpi::broadcast<bool>(*learningInfo.mpiWorld, NEED_HELP, 0);

      if(learningInfo.mpiWorld->rank() == 0) {
        cerr << "done with LBFGS " <<  " at " << time(0) << endl;    
      }

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
        int fromSentId = 0;
        LatentCrfModel &model = LatentCrfModel::GetInstance();
        int toSentId = (int)model.examplesCount;
        double nllPiece = ComputeNllYGivenXAndLambdaGradient(gradientPiece, fromSentId, toSentId);

        // merge your gradient with other slaves
        mpi::reduce< vector<double> >(*learningInfo.mpiWorld, gradientPiece, dummy, 
                                      AggregateVectors2(), 0);

        // aggregate the loglikelihood computation as well
        double dummy2;
        mpi::reduce<double>(*learningInfo.mpiWorld, nllPiece, dummy2, std::plus<double>(), 0);

      }
    } // end if master => run lbfgs() else help master

    if(learningInfo.mpiWorld->rank() == 0) {
      cerr << "supervised training of lambda parameters finished. " << endl;
      lambda->PersistParams(learningInfo.outputFilenamePrefix + ".supervised.lambda", false);
      lambda->PersistParams(learningInfo.outputFilenamePrefix + ".supervised.lambda.humane", true);
      cerr << "parameters can be found at " << learningInfo.outputFilenamePrefix << ".supervised.lambda" << endl;
    }
    
  } 
  
  if(fitThetas) {

    if(learningInfo.mpiWorld->rank() == 0) {
      cerr << "started supervised training of theta parameters..." << endl;
    }

    // optimize theta (i.e. multinomial) parameters to maximize the likeilhood
    SupervisedTrainTheta();

    // broadcast
    BroadcastTheta(0);

    if(learningInfo.mpiWorld->rank() == 0) {
      cerr << "supervised training of theta parameters finished." << endl;
      PersistTheta(learningInfo.outputFilenamePrefix + ".supervised.theta");
      cerr << "parameters can be found at " << learningInfo.outputFilenamePrefix << ".supervised.theta" << endl;
    }

  }
}

void LatentCrfModel::Train() {
  testingMode = false;
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

// when l2 is specified, the regularized objective is returned. when l1 or none is specified, the unregualrized objective is returned
double LatentCrfModel::EvaluateNll() {  
  vector<double> gradientPiece(lambda->GetParamsCount(), 0.0);
  double devSetNllPiece = 0.0;
  double nllPiece = ComputeNllZGivenXAndLambdaGradient(gradientPiece, 0, examplesCount, &devSetNllPiece);
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
  static unsigned fromSentId = 0;
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

double LatentCrfModel::CheckGradient(lbfgs_evaluate_t proc_evaluate, vector<int> &testIndexes, double epsilon) {

  // first, use the lbfgs callback function to analytically compute the objective and gradient at the current lambdas
  int fromSentId = 0;
  void *uselessPtr = &fromSentId;
  double* lambdasArray = lambda->GetParamWeightsArray();
  int lambdasArrayLength = lambda->GetParamsCount();
  double* analyticGradient = new double[lambdasArrayLength];
  double originalObjective = proc_evaluate(uselessPtr, lambdasArray, analyticGradient, 
                                                                   lambdasArrayLength, 0);
  
  // copy the derivatives we need to compare (the gradient vector will be overwritten)
  vector<double> analyticDerivatives;
  for(auto testIndexIter = testIndexes.begin();
      testIndexIter != testIndexes.end();
      ++testIndexIter) {
    analyticDerivatives.push_back(analyticGradient[*testIndexIter]);
  }
  
  // test each test index
  vector<double> numericDerivatives;
  for(auto testIndexIter = testIndexes.begin(); 
      testIndexIter != testIndexes.end();
      ++testIndexIter) {

    // by first modifying the corresponding feature weight, 
    lambdasArray[*testIndexIter] += epsilon;
    
    // computing the new objective
    double modifiedObjective = proc_evaluate(uselessPtr, lambdasArray, analyticGradient, 
                                                                      lambdasArrayLength, 0);
    
    // compute the derivative numerically,
    double objectiveDiff = modifiedObjective - originalObjective;
    double derivative = objectiveDiff / epsilon;
    numericDerivatives.push_back(derivative);
    
    // reset this feature's weight
    lambdasArray[*testIndexIter] -= epsilon;
  }
  
  // summarize your findings
  cerr << "======================" << endl;
  cerr << "CheckGradient summary (with epsilon = " << epsilon << "):" << endl;

  cerr << "feature\tfeature\tfeature\tanalytic\tnumeric\tdiff" << endl;
  cerr << "index\tid\tvalue\tderivative\tderivative\tsquared" << endl;
  double sumOfDiffSquared = 0.0;
  for(int i = 0; i < testIndexes.size(); ++i) { 
    double diff = analyticDerivatives[i] - numericDerivatives[i];
    double diffSquared = diff * diff;
    sumOfDiffSquared += diffSquared;
    cerr << testIndexes[i] << "\t" << (*lambda->paramIdsPtr)[testIndexes[i]] << "\t" << lambdasArray[testIndexes[i]] << "\t";
    cerr << analyticDerivatives[i] << "\t" << numericDerivatives[i] << "\t" << diffSquared << endl;    
  }
  cerr << "\\sum_i (analytic - numeric)^2 = " << sumOfDiffSquared << endl;
  cerr << "=====================" << endl;
}

// lbfgs' callback function for evaluating -logliklihood(y|x) and its d/d_\lambda
// this is needed for supervised training of the CRF
// this function is not expected to be executed by any slave; only the master process with rank 0
double LatentCrfModel::LbfgsCallbackEvalYGivenXLambdaGradient(void *uselessPtr,
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
  int fromSentId = 0;
  int toSentId = model.goldLabelSequences.size();
  
  double NllPiece = model.ComputeNllYGivenXAndLambdaGradient(gradientPiece, fromSentId, toSentId);
  double reducedNll = -1;

  // now, the master aggregates gradient pieces computed by the slaves
  mpi::reduce< vector<double> >(*model.learningInfo.mpiWorld, gradientPiece, reducedGradient, AggregateVectors2(), 0);
  mpi::reduce<double>(*model.learningInfo.mpiWorld, NllPiece, reducedNll, std::plus<double>(), 0);
  assert(reducedNll != -1);
  
  // fill in the gradient array allocated by lbfgs
  cerr << ">>> before l2 reg: reducednll = " << reducedNll;
  double gradientL2Norm = 0.0;
  if(model.learningInfo.optimizationMethod.subOptMethod->regularizer == Regularizer::L2) {
    reducedNll = model.AddL2Term(reducedGradient, gradient, reducedNll, gradientL2Norm);
  } else {
    assert(gradientPiece.size() == reducedGradient.size() && gradientPiece.size() == model.lambda->GetParamsCount());
    assert((unsigned)lambdasCount == model.lambda->GetParamsCount());
    for(unsigned i = 0; i < model.lambda->GetParamsCount(); i++) {
      gradient[i] = reducedGradient[i];
      gradientL2Norm += gradient[i] * gradient[i];
      assert(!std::isnan(gradient[i]) || !std::isinf(gradient[i]));
    } 
  }
  cerr << ", after l2 reg: reducednll = " << reducedNll;
  cerr << ", gradient l2 norm = " << gradientL2Norm << endl;
  
  // useful for inspecting weight/gradient vectors // for debugging
  if(false && model.learningInfo.checkGradient && model.learningInfo.mpiWorld->rank() == 0) {
    cerr << endl << endl << "index\tid\tweight\tderivative" << endl;
    for(int i = 0; i < model.lambda->paramIdsPtr->size(); ++i) {
      cerr << i << "\t" << (*model.lambda->paramIdsPtr)[i] << "\t" << (*model.lambda->paramWeightsPtr)[i] << "\t" << gradient[i] << endl;
    }
    cerr << endl << endl;
  }

  return reducedNll;
}

double LatentCrfModel::ComputeNllYGivenXAndLambdaGradient(
							  vector<double> &derivativeWRTLambda, int fromSentId, int toSentId) {
  cerr << "method not implemented: double LatentCrfParser::ComputeNllYGivenXAndLambdaGradient" << endl;
  assert(false); 
}

void LatentCrfModel::AddWeightedL2Term(vector<double> *gradient, double *objective, FastSparseVector<double> &activeFeatures) {

  if(learningInfo.optimizationMethod.subOptMethod->regularizer != Regularizer::WeightedL2) return;
  if(gradient == NULL || objective == NULL) return;
  for(auto activeFeatureIter = activeFeatures.begin(); 
      activeFeatureIter != activeFeatures.end();
      ++activeFeatureIter) {
    unsigned temp = activeFeatureIter->first;
    double lambda_i = lambda->GetParamWeight(temp);
    double distance = 
      lambda->featureGaussianMeans.find( lambda->GetParamId(temp) ) == lambda->featureGaussianMeans.end()?
      lambda_i: 
      lambda_i - lambda->featureGaussianMeans[ lambda->GetParamId(temp) ];
    (*gradient)[activeFeatureIter->first] += 2.0 * learningInfo.optimizationMethod.subOptMethod->regularizationStrength * distance;
    *objective += learningInfo.optimizationMethod.subOptMethod->regularizationStrength * distance * distance;
  }
}

// adds l2 terms to both the objective and the gradient). return value is the 
// the objective after adding the l2 term.
double LatentCrfModel::AddL2Term(const vector<double> &unregularizedGradient, 
                                 double *regularizedGradient, double unregularizedObjective, double &gradientL2Norm) {
  double l2RegularizedObjective = unregularizedObjective;
  gradientL2Norm = 0;
  // this is where the L2 term is added to both the gradient and objective function
  assert(lambda->GetParamsCount() == unregularizedGradient.size());
  double l2term = 0;
  for(unsigned i = 0; i < lambda->GetParamsCount(); i++) {
    double lambda_i = lambda->GetParamWeight(i);
    double distance = 
      lambda->featureGaussianMeans.find( lambda->GetParamId(i) ) == lambda->featureGaussianMeans.end()?
      lambda_i: 
      lambda_i - lambda->featureGaussianMeans[ lambda->GetParamId(i) ];
    
    regularizedGradient[i] = unregularizedGradient[i] + 2.0 * learningInfo.optimizationMethod.subOptMethod->regularizationStrength * distance;
    l2RegularizedObjective += learningInfo.optimizationMethod.subOptMethod->regularizationStrength * distance * distance;
    l2term += learningInfo.optimizationMethod.subOptMethod->regularizationStrength * distance * distance;
    gradientL2Norm += regularizedGradient[i] * regularizedGradient[i];
    assert(!std::isnan(unregularizedGradient[i]) || !std::isinf(unregularizedGradient[i]));
  } 
  return l2RegularizedObjective;
}

// adds the l2 term to the objective. return value is the the objective after adding the l2 term.
double LatentCrfModel::AddL2Term(double unregularizedObjective) {
  double l2RegularizedObjective = unregularizedObjective;
  for(unsigned i = 0; i < lambda->GetParamsCount(); i++) {
    double lambda_i = lambda->GetParamWeight(i);
    double distance = 
      lambda->featureGaussianMeans.find( lambda->GetParamId(i) ) == lambda->featureGaussianMeans.end()?
      lambda_i: 
      lambda_i - lambda->featureGaussianMeans[ lambda->GetParamId(i) ];
    l2RegularizedObjective += learningInfo.optimizationMethod.subOptMethod->regularizationStrength * distance * distance;
  } 
  return l2RegularizedObjective;
}

// the callback function lbfgs calls to compute the -log likelihood(z|x) and its d/d_\lambda
// this function is not expected to be executed by any slave; only the master process with rank 0
double LatentCrfModel::LbfgsCallbackEvalZGivenXLambdaGradient(void *dummy,
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
  int supervisedFromSentId = 0;
  int supervisedToSentId = model.goldLabelSequences.size();
  int fromSentId = model.goldLabelSequences.size();
  int toSentId = model.examplesCount;
  cerr << "computing the supervised objective for sentIds: " << supervisedFromSentId << "-" << supervisedToSentId << endl;
  cerr << "computing the unsupervised objective for sentIds: " << fromSentId << "-" << toSentId << endl;
  
  double devSetNllPiece = 0;
  double NllPiece = model.ComputeNllZGivenXAndLambdaGradient(gradientPiece, fromSentId, toSentId, &devSetNllPiece);
  double reducedNll = -1, reducedDevSetNll = 0;
  
  // for semi-supervised learning, we need to also collect the gradient from labeled data
  // note this is supposed to *add to the gradient of the unsupervised objective*, but only 
  // return *the value of the supervised objective*
  double SupervisedNllPiece = 
    supervisedFromSentId < supervisedToSentId?
    model.ComputeNllYGivenXAndLambdaGradient(gradientPiece, supervisedFromSentId, supervisedToSentId):
    0.0;
  
  // now, the master aggregates gradient pieces computed by the slaves
  mpi::reduce< vector<double> >(*model.learningInfo.mpiWorld, gradientPiece, reducedGradient, AggregateVectors2(), 0);
  // now the master aggregates the unsupervised objective
  mpi::reduce<double>(*model.learningInfo.mpiWorld, NllPiece + SupervisedNllPiece, reducedNll, std::plus<double>(), 0);
  assert(reducedNll != -1);
  mpi::reduce<double>(*model.learningInfo.mpiWorld, devSetNllPiece, reducedDevSetNll, std::plus<double>(), 0);
  
  /*if(model.learningInfo.mpiWorld->rank() == 0) {
    for(unsigned i = 0; i < reducedGradient.size(); ++i) {
      cerr << "gradient(" << (*model.lambda->paramIdsPtr)[i] << ") = " << reducedGradient[i] << endl;
    }
    }*/
    
  // fill in the gradient array allocated by lbfgs
  cerr << "before l2 reg, reducednll = " << reducedNll;
  double gradientL2Norm = 0.0;
  double featuresL2Norm = 0.0;
  if(model.learningInfo.optimizationMethod.subOptMethod->regularizer == Regularizer::L2) {
    double temp = reducedNll;
    reducedNll = model.AddL2Term(reducedGradient, gradient, reducedNll, gradientL2Norm);
    featuresL2Norm = reducedNll - temp;
  } else {
    assert(gradientPiece.size() == reducedGradient.size() && gradientPiece.size() == model.lambda->GetParamsCount());
    assert((unsigned)lambdasCount == model.lambda->GetParamsCount());
    for(unsigned i = 0; i < model.lambda->GetParamsCount(); i++) {
      gradient[i] = reducedGradient[i];
      gradientL2Norm += gradient[i] * gradient[i];
      assert(!std::isnan(gradient[i]) || !std::isinf(gradient[i]));
    } 
  }
  cerr << endl << "features l2 norm = " << featuresL2Norm;
  cerr << endl << "gradient l2 norm = " << gradientL2Norm;
  cerr << endl << "after l2 reg, reducednll = " << reducedNll << endl;
  
  if(model.learningInfo.useEarlyStopping) {
    cerr << " dev set negative loglikelihood = " << reducedDevSetNll << endl;
  }

  // useful for inspecting weight/gradient vectors // for debugging
  if(false && model.learningInfo.checkGradient && model.learningInfo.mpiWorld->rank() == 0) {
    cerr << endl;
    cerr << "=============================================" << endl;
    cerr << "            GRADIENT" << endl;
    cerr << "index\tid\tweight\tderivative" << endl;
    cerr << "=============================================" << endl;
    for(int i = 0; i < model.lambda->paramIdsPtr->size(); ++i) {
      cerr << i << "\t" << (*model.lambda->paramIdsPtr)[i] << "\t" << (*model.lambda->paramWeightsPtr)[i] << "\t" << gradient[i] << endl;
    }
    cerr << endl << endl;
  }

  
  return reducedNll;
}

// -loglikelihood is the return value
double LatentCrfModel::ComputeNllZGivenXAndLambdaGradient(
							  vector<double> &derivativeWRTLambda, int fromSentId, int toSentId, double *devSetNll) {

  assert(*devSetNll == 0.0);
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
    if(learningInfo.maxSequenceLength > 0 && GetObservableSequence(sentId).size() > learningInfo.maxSequenceLength) {
      cerr << "sentId = " << sentId << " was pruned because of its length = " << GetObservableSequence(sentId).size() << endl;
      continue;
    }
    
    
    // build the FSTs
    fst::VectorFst<FstUtils::LogArc> thetaLambdaFst, lambdaFst;
    vector<FstUtils::LogWeight> thetaLambdaAlphas, lambdaAlphas, thetaLambdaBetas, lambdaBetas;
    if(!ignoreThetaTerms) {
      BuildThetaLambdaFst(sentId, 
                          GetReconstructedObservableSequence(sentId), 
                          thetaLambdaFst, 
                          thetaLambdaAlphas, 
                          thetaLambdaBetas);
    }
    BuildLambdaFst(sentId, lambdaFst, lambdaAlphas, lambdaBetas, &derivativeWRTLambda, &objective);
    
    // compute the D map for this sentence
    FastSparseVector<double> DOverCSparseVector;
    if(!ignoreThetaTerms) {
      ComputeDOverC(sentId, GetObservableSequence(sentId), thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas, DOverCSparseVector);
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
      
      if(learningInfo.useEarlyStopping && sentId % 10 == 0) {
        *devSetNll += nLogC;
      } else {
        objective += nLogC;
        // add D/C to the gradient
        for(auto dIter = DOverCSparseVector.begin(); 
            dIter != DOverCSparseVector.end(); ++dIter) {
          double dOverC = dIter->second;
          if(std::isnan(dOverC) || std::isinf(dOverC)) {
            if(learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
              cerr << "ERROR: dOverC = " << dOverC << ". my mistake. will halt!" << endl;
            }
            assert(false);
          }
          assert(derivativeWRTLambda.size() > dIter->first);
          derivativeWRTLambda[dIter->first] -= dOverC;
        }
      }
    }
    
    // compute the F map fro this sentence
    FastSparseVector<double> FOverZSparseVector;
    ComputeFOverZ(sentId, lambdaFst, lambdaAlphas, lambdaBetas, FOverZSparseVector);

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
    if(learningInfo.useEarlyStopping && sentId % 10 == 0) {
      cerr << "useEarlyStopping = " << learningInfo.useEarlyStopping << endl;
      cerr << "sentId = " << sentId << endl;
      *devSetNll -= nLogZ;
      cerr << "devSetNll = " << *devSetNll << endl;
    } else {
      if(nLogC < nLogZ) {
        cerr << "this must be a bug. nLogC always be >= nLogZ. " << endl;
        cerr << "nLogC = " << nLogC << endl;
        cerr << "nLogZ = " << nLogZ << endl;
      }
      objective -= nLogZ;
      
      // subtract F/Z from the gradient
      for(auto fOverZIter = FOverZSparseVector.begin(); 
          fOverZIter != FOverZSparseVector.end(); ++fOverZIter) {
        double fOverZ = fOverZIter->second;
        if(std::isnan(fOverZ) || std::isinf(fOverZ)) {
          if(learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
            cerr << "ERROR: fOverZ = " << fOverZ << ". my mistake. will halt!" << endl;
          }
          assert(false);
        }
        assert(fOverZIter->first < derivativeWRTLambda.size());
        derivativeWRTLambda[fOverZIter->first] += fOverZ;
        if(std::isnan(derivativeWRTLambda[fOverZIter->first]) || 
           std::isinf(derivativeWRTLambda[fOverZIter->first])) {
          cerr << "rank #" << learningInfo.mpiWorld->rank()        \
               << ": ERROR: fOverZ = " << fOverZ                   \
               << ". my mistake. will halt!" << endl;
          assert(false);
        }
      }
    }
    
    // debug info
    if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH && sentId % learningInfo.nSentsPerDot == 0) {
      cerr << ".";
    }
  } // end of training examples 

  cerr << learningInfo.mpiWorld->rank() << "|";

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
    cerr << endl << "<<< " << model.learningInfo.mpiWorld->rank() << "report: coord-descent iteration # " << model.learningInfo.iterationsCount;
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
  if(model.learningInfo.debugLevel >= DebugLevel::MINI_BATCH) {
    cerr << endl << endl;
  }
  
  if(model.learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
    cerr << "done" << endl;
  }

  //  cerr << "ending LatentCrfModel::LbfgsProgressReport" << endl;
  //

  model.learningInfo.hackK = k;
  
  return 0;
}

double LatentCrfModel::UpdateThetaMleForSent(const unsigned sentId, 
					     MultinomialParams::ConditionalMultinomialParam<int64_t> &mleGivenOneLabel, 
					     boost::unordered_map<int64_t, double> &mleMarginalsGivenOneLabel,
					     MultinomialParams::ConditionalMultinomialParam< pair<int64_t, int64_t> > &mleGivenTwoLabels, 
					     boost::unordered_map< pair<int64_t, int64_t>, double> &mleMarginalsGivenTwoLabels) {
  
  // in the word alignment model, yDomain depends on the example
  PrepareExample(sentId);
  
  double nll = UpdateThetaMleForSent(sentId, mleGivenOneLabel, mleMarginalsGivenOneLabel);
  return nll;
}

void LatentCrfModel::NormalizeThetaMleAndUpdateTheta(
    MultinomialParams::ConditionalMultinomialParam<int64_t> &mleGivenOneLabel, 
    boost::unordered_map<int64_t, double> &mleMarginalsGivenOneLabel,
    MultinomialParams::ConditionalMultinomialParam< std::pair<int64_t, int64_t> > &mleGivenTwoLabels, 
    boost::unordered_map< std::pair<int64_t, int64_t>, double> &mleMarginalsGivenTwoLabels) {
  
  bool unnormalizedParamsAreInNLog = false;
  bool normalizedParamsAreInNLog = true;
  MultinomialParams::NormalizeParams(mleGivenOneLabel, 
				     learningInfo.multinomialSymmetricDirichletAlpha, 
				     unnormalizedParamsAreInNLog,
				     normalizedParamsAreInNLog,
				     learningInfo.variationalInferenceOfMultinomials);
  nLogThetaGivenOneLabel = mleGivenOneLabel;
}

void LatentCrfModel::PrintLbfgsConfig(lbfgs_parameter_t &lbfgsParams) {
  cerr << "============================" << endl;
  cerr << "configurations for liblbfgs:" << endl;
  cerr << "============================" << endl;
  cerr << "lbfgsParams.m = " << lbfgsParams.m << endl;
  cerr << "lbfgsParams.epsilon = " << lbfgsParams.epsilon << endl;
  cerr << "lbfgsParams.past = " << lbfgsParams.past << endl;
  cerr << "lbfgsParams.delta = " << lbfgsParams.delta << endl;
  cerr << "lbfgsParams.max_iterations = " << lbfgsParams.max_iterations << endl;
  cerr << "lbfgsParams.linesearch = " << lbfgsParams.linesearch << endl;
  cerr << "lbfgsParams.max_linesearch = " << lbfgsParams.max_linesearch << endl;
  cerr << "lbfgsParams.min_step = " << lbfgsParams.min_step << endl;
  cerr << "lbfgsParams.max_step = " << lbfgsParams.max_step << endl;
  cerr << "lbfgsParams.ftol = " << lbfgsParams.ftol << endl;
  cerr << "lbfgsParams.wolfe = " << lbfgsParams.wolfe << endl;
  cerr << "lbfgsParams.gtol = " << lbfgsParams.gtol << endl;
  cerr << "lbfgsParams.xtol = " << lbfgsParams.xtol << endl;
  cerr << "lbfgsParams.orthantwise_c = " << lbfgsParams.orthantwise_c << endl;
  cerr << "lbfgsParams.orthantwise_start = " << lbfgsParams.orthantwise_start << endl;
  cerr << "lbfgsParams.orthantwise_end = " << lbfgsParams.orthantwise_end << endl;
  cerr << "============================" << endl;
}

lbfgs_parameter_t LatentCrfModel::SetLbfgsConfig(bool supervised) {
  // lbfgs configurations
  lbfgs_parameter_t lbfgsParams;
  lbfgs_parameter_init(&lbfgsParams);
  assert(learningInfo.optimizationMethod.subOptMethod != 0);
  if(supervised) {
    lbfgsParams.max_iterations = learningInfo.supervisedMaxLbfgsIterCount;
  } else {
    lbfgsParams.max_iterations = learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxIterations;
  }
  lbfgsParams.m = learningInfo.optimizationMethod.subOptMethod->lbfgsParams.memoryBuffer;
  //lbfgsParams.xtol = learningInfo.optimizationMethod.subOptMethod->lbfgsParams.precision;
  lbfgsParams.max_linesearch = learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxEvalsPerIteration;
  switch(learningInfo.optimizationMethod.subOptMethod->regularizer) {
  case Regularizer::L1:
    lbfgsParams.orthantwise_c = learningInfo.optimizationMethod.subOptMethod->lbfgsParams.l1Strength;
    assert(learningInfo.optimizationMethod.subOptMethod->lbfgsParams.l1Strength == 
           learningInfo.optimizationMethod.subOptMethod->regularizationStrength);
    // this is the only linesearch algorithm that seems to work with orthantwise lbfgs
    lbfgsParams.linesearch = 0;//LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
    break;
  case Regularizer::L2:
  case Regularizer::WeightedL2:
    // nothing to be done now. l2 is implemented in the lbfgs callback evaluate function. 
    // weighted l2 is implemented in BuildLambdaFst()
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

  mpi::broadcast< boost::unordered_map< int64_t, MultinomialParams::MultinomialParam > >(*learningInfo.mpiWorld, nLogThetaGivenOneLabel.params, rankId);
  
  if(learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
    cerr << "rank #" << learningInfo.mpiWorld->rank() << ": after calling BroadcastTheta()" << endl;
  }
}

void LatentCrfModel::ReduceMleAndMarginals(
             MultinomialParams::ConditionalMultinomialParam<int64_t> &mleGivenOneLabel, 
					   MultinomialParams::ConditionalMultinomialParam< pair<int64_t, int64_t> > &mleGivenTwoLabels,
					   boost::unordered_map<int64_t, double> &mleMarginalsGivenOneLabel,
					   boost::unordered_map<std::pair<int64_t, int64_t>, double> &mleMarginalsGivenTwoLabels) {
  if(learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
    cerr << "rank" << learningInfo.mpiWorld->rank() << ": before calling ReduceMleAndMarginals()" << endl;
  }
  
  mpi::reduce< boost::unordered_map< int64_t, MultinomialParams::MultinomialParam > >(
                   *learningInfo.mpiWorld, 
								   mleGivenOneLabel.params, mleGivenOneLabel.params, 
								   MultinomialParams::AccumulateConditionalMultinomials< int64_t >, 0);
  mpi::reduce< boost::unordered_map< int64_t, double > >(*learningInfo.mpiWorld, 
				      mleMarginalsGivenOneLabel, mleMarginalsGivenOneLabel, 
				      MultinomialParams::AccumulateMultinomials<int64_t>, 0);
  
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
  bool supervised=false;
  lbfgs_parameter_t lbfgsParams = SetLbfgsConfig(supervised);
  if(learningInfo.mpiWorld->rank() == 0) {
    PrintLbfgsConfig(lbfgsParams);
  }
  
  // variables used for adagrad
  vector<double> gradient(lambda->GetParamsCount());
  vector<double> u(lambda->GetParamsCount());
  vector<double> h(lambda->GetParamsCount());
  for(unsigned paramId = 0; paramId < lambda->GetParamsCount(); ++paramId) {
    u[paramId] = 0;
    h[paramId] = 0;
  }
  int adagradIter = 1;

  // fix learningInfo.firstKExamplesToLabel
  if(learningInfo.firstKExamplesToLabel == 1) {
    learningInfo.firstKExamplesToLabel = examplesCount;
  }
  
  // baby steps
  unsigned originalMaxSequenceLength = learningInfo.maxSequenceLength;

  // TRAINING ITERATIONS
  bool converged = false;
  do {

    if(learningInfo.mpiWorld->rank() == 0) {
      cerr << "starting coordinate descent iteration #" << learningInfo.iterationsCount <<  " at " << time(0) << endl;
    }

    if(learningInfo.useMaxIterationsCount && learningInfo.maxIterationsCount == 0) {
      // no training at all!
      break;
    }

    if(learningInfo.babySteps) {
      learningInfo.maxSequenceLength = min(learningInfo.iterationsCount + 3, originalMaxSequenceLength);
    }

    unsigned firstSentIdUsedForTraining = learningInfo.inductive? learningInfo.firstKExamplesToLabel: 0;
    if(learningInfo.mpiWorld->rank() == 0) {
      cerr << "training starts at sentId = " << firstSentIdUsedForTraining << endl;
    }
    
    // debug info
    if(learningInfo.mpiWorld->rank() == 0) {
      cerr << "master" << learningInfo.mpiWorld->rank() << ": ====================== ITERATION " << learningInfo.iterationsCount << " =====================" << endl << endl;
      cerr << "master" << learningInfo.mpiWorld->rank() << ": ========== first, update thetas using a few EM iterations: =========" << endl << endl;
      
      /*
        cerr << "lambda->PrintParams()" << endl;
        lambda->PrintParams();
        cerr << endl << "nLogThetaGivenOneLabel.params.PrintParams()" << endl;
        nLogThetaGivenOneLabel.PrintParams(vocabEncoder, true, true);
      */
    }
  
    if(learningInfo.iterationsCount == 0 && learningInfo.optimizeLambdasFirst) {
      // don't touch theta parameters in the first iteration
    } else if(learningInfo.thetaOptMethod->algorithm == EXPECTATION_MAXIMIZATION) {
      // run a few EM iterations to update thetas
      for(unsigned emIter = 0; emIter < learningInfo.emIterationsCount; ++emIter) {
        
        if(learningInfo.mpiWorld->rank() == 0) {
          cerr << "starting EM iteration #" << emIter <<  " at " << time(0) << endl;    
        }

        lambda->GetParamsCount();
        // skip EM updates of the first block-coord-descent iteration
        //if(learningInfo.iterationsCount == 0) {
        //  break;
        //}
        
        // UPDATE THETAS by normalizing soft counts (i.e. the closed form MLE solution)
        // data structure to hold theta MLE estimates
        MultinomialParams::ConditionalMultinomialParam<int64_t> mleGivenOneLabel;
        MultinomialParams::ConditionalMultinomialParam< pair<int64_t, int64_t> > mleGivenTwoLabels;
        boost::unordered_map<int64_t, double> mleMarginalsGivenOneLabel;
        boost::unordered_map<std::pair<int64_t, int64_t>, double> mleMarginalsGivenTwoLabels;
        
        // update the mle for each sentence
        assert(examplesCount > 0);
        if(learningInfo.mpiWorld->rank() == 0) {
          cerr << endl << "aggregating soft counts for each theta parameter...";
        }
        double unregularizedObjective = 0;
        for(unsigned sentId = firstSentIdUsedForTraining; sentId < examplesCount; sentId++) {

          // sentId is assigned to the process # (sentId % world.size())
          if(sentId % learningInfo.mpiWorld->size() != (unsigned)learningInfo.mpiWorld->rank()) {
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
          if(learningInfo.optimizationMethod.subOptMethod->regularizer == Regularizer::L2 || 
             learningInfo.optimizationMethod.subOptMethod->regularizer == Regularizer::WeightedL2) {
            cerr << " l2 reg. objective = " << regularizedObjective << " " << endl;
          } else { 
            cerr << " unregularized objective = " << unregularizedObjective << " " << endl;
          }
        }	
        
        // normalize mle and update nLogTheta on master
        if(learningInfo.mpiWorld->rank() == 0) {
          NormalizeThetaMleAndUpdateTheta(mleGivenOneLabel, mleMarginalsGivenOneLabel, 
                                          mleGivenTwoLabels, mleMarginalsGivenTwoLabels);
        }
        
        // update nLogTheta on slaves
        BroadcastTheta(0);

        if(learningInfo.mpiWorld->rank() == 0) {
          cerr << "ending EM iteration #" << emIter <<  " at " << time(0) << endl;    
        }

      } // end of EM iterations

      // for debugging
      //(*learningInfo.endOfKIterationsCallbackFunction)();
    
      // end of if(thetaOptMethod->algorithm == EM)
    } else if (learningInfo.thetaOptMethod->algorithm == GRADIENT_DESCENT) {
      assert(learningInfo.mpiWorld->size() == 1); // this method is only supported for single-threaded runs
      
      for(int gradientDescentIter = 0; gradientDescentIter < 10; ++gradientDescentIter) {
        
        cerr << "at the beginning of gradient descent iteration " << gradientDescentIter << ", EvaluateNll() = " << EvaluateNll() << endl;
        
        MultinomialParams::ConditionalMultinomialParam<int64_t> gradientOfNll;
        ComputeNllZGivenXThetaGradient(gradientOfNll);
        for(auto yIter = nLogThetaGivenOneLabel.params.begin(); 
            yIter != nLogThetaGivenOneLabel.params.end(); 
            ++yIter) {
          double marginal = 0.0;
          for(auto zIter = yIter->second.begin(); zIter != yIter->second.end(); ++zIter) {
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
          for(auto zIter = yIter->second.begin(); zIter != yIter->second.end(); ++zIter) {
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

    // debug info
    if( (learningInfo.iterationsCount % learningInfo.persistParamsAfterNIteration == 0) && (learningInfo.mpiWorld->rank() == 0) ) {
      PersistTheta(GetThetaFilename(learningInfo.iterationsCount));
    }
    // label the first K examples from the training set (i.e. the test set)
    /*if(learningInfo.iterationsCount % learningInfo.invokeCallbackFunctionEveryKIterations == 0 && \
    //   learningInfo.endOfKIterationsCallbackFunction != 0) {
    //  // call the call back function
    //  (*learningInfo.endOfKIterationsCallbackFunction)();
    }*/

    // update the lambdas
    this->optimizingLambda = true;
    // debug info
    if(learningInfo.debugLevel >= DebugLevel::CORPUS && learningInfo.mpiWorld->rank() == 0) {
      cerr << endl << "master" << learningInfo.mpiWorld->rank() << ": ========== second, update lambdas ==========" << endl << endl;
    }
    
    double Nll = 0, devSetNll = 0;
    // note: batch == minibatch with size equals to data.size()
    //for(unsigned sentId = firstSentIdUsedForTraining; sentId < examplesCount; sentId += learningInfo.optimizationMethod.subOptMethod->miniBatchSize) {
      
    //int fromSentId = sentId;
    //  int toSentId = min(sentId+learningInfo.optimizationMethod.subOptMethod->miniBatchSize, examplesCount);
        
      // debug info
      double optimizedMiniBatchNll = 0, miniBatchDevSetNll = 0;
      if(learningInfo.mpiWorld->rank() == 0) {
        cerr << "master" << learningInfo.mpiWorld->rank() << ": optimizing lambda weights to maximize both unsupervised and supervised objectives." << endl;
      }
      
      // use LBFGS to update lambdas
      if(learningInfo.optimizationMethod.subOptMethod->algorithm == LBFGS && learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxIterations > 0) {
        
        // parallelizing the lbfgs callback function is complicated
        if(learningInfo.mpiWorld->rank() == 0) {
          
          // populate lambdasArray and lambasArrayLength
          // don't optimize all parameters. only optimize unconstrained ones
          double* lambdasArray;
          int lambdasArrayLength;
          lambdasArray = lambda->GetParamWeightsArray();
          lambdasArrayLength = lambda->GetParamsCount();
          
          // check the analytic gradient computation by computing the derivatives numerically 
          // using method of finite differenes for a subset of the features
          int testIndexesCount = 20;
          double epsilon = 0.00000001;
          int granularities = 1;
          vector<int> testIndexes;
          if(true && learningInfo.checkGradient) {
            testIndexes = lambda->SampleFeatures(testIndexesCount);
            cerr << "calling CheckGradient() before running lbfgs inside coordinate descent" << endl; 
            for(int granularity = 0; granularity < granularities; epsilon /= 10, granularity++) {
              CheckGradient(LbfgsCallbackEvalZGivenXLambdaGradient, testIndexes, epsilon);
            }
          }
          
          // only the master executes lbfgs
          assert(learningInfo.mpiWorld->rank() == 0);
          cerr << "will start LBFGS " <<  " at " << time(0) << endl;    
          int dummy=0;
          int lbfgsStatus = lbfgs(lambdasArrayLength, lambdasArray, &optimizedMiniBatchNll, 
                                  LbfgsCallbackEvalZGivenXLambdaGradient, LbfgsProgressReport, &dummy, &lbfgsParams);
          
          bool NEED_HELP = false;
          mpi::broadcast<bool>(*learningInfo.mpiWorld, NEED_HELP, 0);

          if(learningInfo.mpiWorld->rank() == 0) {
            cerr << "done with LBFGS " <<  " at " << time(0) << endl;    
          }
          
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

            // determine sentIds
            int supervisedFromSentId = 0;
            int supervisedToSentId = goldLabelSequences.size();
            int fromSentId = goldLabelSequences.size();
            int toSentId = examplesCount;
            if(learningInfo.mpiWorld->rank() == 1) {
              cerr << "slave: computing the supervised objective for sentIds: " << supervisedFromSentId << "-" << supervisedToSentId << endl;
              cerr << "slave: computing the unsupervised objective for sentIds: " << fromSentId << "-" << toSentId << endl;
            }
            
            // process your share of examples
            vector<double> gradientPiece(lambda->GetParamsCount(), 0.0), dummy;
            double devSetNllPiece = 0.0;
            double nllPiece = ComputeNllZGivenXAndLambdaGradient(gradientPiece, fromSentId, toSentId, &devSetNllPiece);
            
            // for semi-supervised learning, we need to also collect the gradient from labeled data
            // note this is supposed to *add to the gradient of the unsupervised objective*, but only 
            // return *the value of the supervised objective*
            double SupervisedNllPiece = 
              supervisedFromSentId < supervisedToSentId?
              ComputeNllYGivenXAndLambdaGradient(gradientPiece, supervisedFromSentId, supervisedToSentId):
              0.0;
            
            // merge your gradient with other slaves
            mpi::reduce< vector<double> >(*learningInfo.mpiWorld, gradientPiece, dummy, 
                                          AggregateVectors2(), 0);
            
            // aggregate the loglikelihood computation as well
            double dummy2;
            mpi::reduce<double>(*learningInfo.mpiWorld, nllPiece + SupervisedNllPiece, dummy2, std::plus<double>(), 0);
            
            // aggregate the loglikelihood computation as well
            mpi::reduce<double>(*learningInfo.mpiWorld, devSetNllPiece, dummy2, std::plus<double>(), 0);
            
            // for debug
            if(learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
              cerr << "rank" << learningInfo.mpiWorld->rank() << ": i'm trapped in this loop, repeatedly helping master evaluate likelihood and gradient for lbfgs." << endl;
            }
          }
        } // end if master => run lbfgs() else help master
        
      } else if (learningInfo.optimizationMethod.subOptMethod->algorithm == ADAGRAD) {
        
        if(goldLabelSequences.size() > 0) {
          cerr << "ADAGRAD IS NOT AVAILABLE FOR SEMI-SUPERVISED LEARNING" << endl;
          assert(false);
        }

        if(learningInfo.mpiWorld->rank() == 0) {
          cerr << "master" << learningInfo.mpiWorld->rank() << ": we'll use adagrad to update the lambda parameters" << endl;
        }
        
        // sync.
        double dummy4;
        mpi::all_reduce<double>(*learningInfo.mpiWorld, dummy4, dummy4, std::plus<double>());
        
        bool adagradConverged = false;
        // in each adagrad iter
        while(!adagradConverged) {    

          if(learningInfo.mpiWorld->rank() == 0) {
            cerr << "new adagrad iteration " <<  " at " << time(0) << endl;    
          }

          // compute the loss and its gradient
          double* lambdasArray = lambda->GetParamWeightsArray();
          
          // set sentIds
          int supervisedFromSentId = 0;
          int supervisedToSentId = goldLabelSequences.size();
          int fromSentId = goldLabelSequences.size();
          int toSentId = examplesCount;
          
          // process your share of examples
          vector<double> gradientPiece(lambda->GetParamsCount(), 0.0);
          double devSetNllPiece = 0.0;
          double nllPiece = ComputeNllZGivenXAndLambdaGradient(gradientPiece, fromSentId, toSentId, &devSetNllPiece);

          // merge your gradient with other slaves
          mpi::reduce< vector<double> >(*learningInfo.mpiWorld, gradientPiece, gradient, 
                                        AggregateVectors2(), 0);

          // for debugging (remove it later to speed things up)
          if(false && learningInfo.mpiWorld->rank() == 0) {
            double gradientL2 = 0.0;
            for(auto gradientIter = gradient.begin(); gradientIter != gradient.end(); gradientIter++) {
              gradientL2 += (*gradientIter) * (*gradientIter);
              //cerr << "*gradientIter = " << *gradientIter << endl;
            }
            cerr << endl << "gradientL2 = " << gradientL2 << ", adagradIter = " << adagradIter << endl;
          }
            
          // aggregate the loglikelihood computation as well
          mpi::reduce<double>(*learningInfo.mpiWorld, nllPiece, optimizedMiniBatchNll, std::plus<double>(), 0);
          
          // aggregate the devset loglikelihood computation as well
          mpi::reduce<double>(*learningInfo.mpiWorld, devSetNllPiece, miniBatchDevSetNll, std::plus<double>(), 0);
          
          // add l2 regularization terms to objective and gradient
          if(learningInfo.mpiWorld->rank() == 0 &&
             learningInfo.optimizationMethod.subOptMethod->regularizer == Regularizer::L2) {
            cerr << "actually adding an l2 term.. before: " << optimizedMiniBatchNll << endl;
            double gradientL2Norm = 0;
            optimizedMiniBatchNll = this->AddL2Term(gradient, gradient.data(), optimizedMiniBatchNll, gradientL2Norm);
            cerr << "..after: " << optimizedMiniBatchNll;
            cerr << ", gradientL2Norm = " << gradientL2Norm << endl;
          }
          
          // log
          if(learningInfo.mpiWorld->rank() == 0) { cerr << " -- nll = " << optimizedMiniBatchNll << endl; }
          if(learningInfo.mpiWorld->rank() == 0 && miniBatchDevSetNll != 0) { cerr << " -- devset nll = " << miniBatchDevSetNll << endl; }
          
          // l1 strength?
          double l1 = learningInfo.optimizationMethod.subOptMethod->regularizer == Regularizer::L1? 
            learningInfo.optimizationMethod.subOptMethod->regularizationStrength : 0.0;
          
          // core of adagrad algorithm
          // loop over params
          if(learningInfo.mpiWorld->rank() == 0) {
            // update param weight
            assert(toSentId >= fromSentId);
            double eta = 0.1;
            for(unsigned paramId = 0; paramId < lambda->GetParamsCount(); ++paramId) {
              if(gradient[paramId] == 0.0 && lambdasArray[paramId] == 0.0) {continue;}
              // add l1 term
              optimizedMiniBatchNll += fabs(lambdasArray[paramId]);
              // the h array accumulates the squared gradient across iterations
              u[paramId] += gradient[paramId];
              h[paramId] += gradient[paramId] * gradient[paramId];
              if(l1 > 0.0) {
                double discountedAvgDerivative = fabs(u[paramId]/adagradIter) - l1;
                if(discountedAvgDerivative <= 0.0) {
                  lambdasArray[paramId] = 0.0;
                } else {
                  double uSign = u[paramId] / fabs(u[paramId]);
                  lambdasArray[paramId] = - uSign * eta / sqrt(h[paramId]) * discountedAvgDerivative;
                }
              } else {
                if(h[paramId] > 0.0) {
                  lambdasArray[paramId] -= eta / sqrt(h[paramId]) * gradient[paramId];
                } 
              }
            }
          }
          adagradIter++;
          
          double dummy3;
          mpi::all_reduce<double>(*learningInfo.mpiWorld, dummy3, dummy3, std::plus<double>());
          if(learningInfo.mpiWorld->rank() == 0) {
            cerr << "adagrad is done for this minibatch" << endl;
          } 
          
          // convergence criterion for adagrad 
          //int maxAdagradIter = 1; //learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxIterations;
          adagradConverged = true;
        }
        
        if(learningInfo.mpiWorld->rank() == 0) {
          cerr << "done with adagrad " << " at " << time(0) << endl;    
        }
      } else if(learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxIterations == 0) {
        cerr << "Lambdas are not optimized due to ";
        cerr << "learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxIterations == 0" << endl; 
      } else {
        assert(false);
      }
      
      // debug info
      if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH && learningInfo.mpiWorld->rank() == 0) {
        cerr << "master" << learningInfo.mpiWorld->rank() << ": optimized Nll is " << optimizedMiniBatchNll << endl;
        cerr << "master" << learningInfo.mpiWorld->rank() << ": optimized dev set Nll is " << miniBatchDevSetNll << endl;
      }
      
      // update iteration's Nll
      if(std::isnan(optimizedMiniBatchNll) || std::isinf(optimizedMiniBatchNll)) {
        if(learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
          cerr << "ERROR: optimizedMiniBatchNll = " << optimizedMiniBatchNll << ". didn't add this batch's likelihood to the total likelihood. will halt!" << endl;
        }
        assert(false);
      } else {
        Nll += optimizedMiniBatchNll;
        devSetNll += miniBatchDevSetNll;
      }
      
      //} // for each minibatch

    // done optimizing lambdas
    this->optimizingLambda = false;
    
    // persist updated lambda params
    if(learningInfo.iterationsCount % learningInfo.persistParamsAfterNIteration == 0 && 
       learningInfo.mpiWorld->rank() == 0) {
      lambda->PersistParams(GetLambdaFilename(learningInfo.iterationsCount, false), false);
      lambda->PersistParams(GetLambdaFilename(learningInfo.iterationsCount, true), true);
    }

    double dummy5;
    mpi::all_reduce<double>(*learningInfo.mpiWorld, dummy5, dummy5, std::plus<double>());    
    
    // label the first K examples from the training set (i.e. the test set)
    if(learningInfo.iterationsCount % learningInfo.invokeCallbackFunctionEveryKIterations == 0 && \
       learningInfo.endOfKIterationsCallbackFunction != 0) {
      // call the call back function
      (*learningInfo.endOfKIterationsCallbackFunction)();
    }
    
    // debug info
    if(learningInfo.debugLevel >= DebugLevel::CORPUS && learningInfo.mpiWorld->rank() == 0) {
      cerr << endl << "master" << learningInfo.mpiWorld->rank() << ": finished coordinate descent iteration #" << learningInfo.iterationsCount << " Nll=" << Nll << endl;
    }
    
    // update learningInfo
    mpi::broadcast<double>(*learningInfo.mpiWorld, Nll, 0);
    learningInfo.logLikelihood.push_back(-Nll);
    if(learningInfo.useEarlyStopping) {
      learningInfo.validationLogLikelihood.push_back(-devSetNll);
    }
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

  // after convergence, set the model parameters to those obtained in the "best iteration". Unfortunately,
  // this is only possible when we persist the parameters after each iteration, and we don't use variational
  // inference
  if(learningInfo.iterationsCount > 0 && learningInfo.persistParamsAfterNIteration == 1 && !learningInfo.variationalInferenceOfMultinomials) {
    unsigned bestIteration = learningInfo.GetBestIterationNumber();
    if(bestIteration != learningInfo.iterationsCount - 1 && bestIteration != 0) {
      if(learningInfo.mpiWorld->rank() == 0) {
        cerr << "best iteration is found to be #" << bestIteration << endl;
      }    
      if(learningInfo.mpiWorld->rank() == 0) { cerr << "Now, lets load the parameters of that iteration to produce output labels." << endl; }
      MultinomialParams::LoadParams(GetThetaFilename(bestIteration), nLogThetaGivenOneLabel, vocabEncoder, true, true);
      if(learningInfo.mpiWorld->rank() == 0) { cerr << "unsealing..."; }
      lambda->Unseal();
      if(learningInfo.mpiWorld->rank() == 0) { cerr << "done." << endl << "loading params..."; }
      lambda->LoadParams(GetLambdaFilename(bestIteration, false));
      if(learningInfo.mpiWorld->rank() == 0) { cerr << "done." << endl << "sealing..."; }
      lambda->Seal();
      if(learningInfo.mpiWorld->rank() == 0) { cerr << "done." << endl; }
    }
  }
}

void LatentCrfModel::Label(vector<string> &tokens, vector<int> &labels) {
  assert(labels.size() == 0);
  assert(tokens.size() > 0);
  vector<int64_t> tokensInt;
  for(unsigned i = 0; i < tokens.size(); i++) {
    tokensInt.push_back(vocabEncoder.Encode(tokens[i]));
  }
  Label(tokensInt, labels);
}

void LatentCrfModel::Label(vector<vector<int64_t> > &tokens, vector<vector<int> > &labels) {
  assert(labels.size() == 0);
  labels.resize(tokens.size());
  for(unsigned i = 0; i < tokens.size(); i++) {
    Label(tokens[i], labels[i]);
  }
}

void LatentCrfModel::Label(vector<vector<string> > &tokens, vector<vector<int> > &labels) {
  assert(labels.size() == 0);
  labels.resize(tokens.size());
  for(unsigned i = 0 ; i <tokens.size(); i++) {
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
  for(unsigned sentId = 0; sentId < tokens.size(); sentId++) {
    for(unsigned i = 0; i < tokens[sentId].size(); i++) {
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

// make sure all features which may fire on this training data have a corresponding parameter in lambda (member)
void LatentCrfModel::InitLambda() {
  if(learningInfo.mpiWorld->rank() == 0) {
    cerr << "examplesCount = " << examplesCount << endl;
    cerr << "master" << learningInfo.mpiWorld->rank() << ": initializing lambdas..." << endl;
  }

  assert(examplesCount > 0);

  // then, each process discovers the features that may show up in their sentences.
  for(unsigned sentId = 0; sentId < examplesCount; sentId++) {

    assert(learningInfo.mpiWorld->size() > 0);
    
    // skip sentences not assigned to this process
    if(sentId % learningInfo.mpiWorld->size() != (unsigned)learningInfo.mpiWorld->rank()) {
      continue;
    }

    // set learningInfo.currentSentId
    // TODO-REFACTOR: reconsider the need for this currentSentId variable. 
    // It's hard to remember to set it everywhere it's supposed to be set.
    lambda->learningInfo->currentSentId = sentId;
    //GetObservableSequence(sentId);
    
    //    FastSparseVector<double> h;
    //    FireFeatures(sentId, h);
    fst::VectorFst<FstUtils::LogArc> fst;
    vector<double> derivativeWRTLambda;
    double objective;
    BuildLambdaFst(sentId, fst, &derivativeWRTLambda, &objective);
  }

  if(learningInfo.mpiWorld->rank() == 0) {
    cerr << "master" << learningInfo.mpiWorld->rank() << ": each process extracted features from its respective examples. Now, master will reduce all of them...";
  }

  // master collects all feature ids fired on any sentence
  assert(!lambda->IsSealed());

  if (learningInfo.mpiWorld->rank() == 0) {
    std::vector< std::vector< FeatureId > > localFeatureVectors;
    cerr << "master: collecting lambdas from all slaves ... ";
    mpi::gather<std::vector< FeatureId > >(*learningInfo.mpiWorld, lambda->paramIdsTemp, localFeatureVectors, 0);
    for (int proc = 0; proc < learningInfo.mpiWorld->size(); ++proc) {
      lambda->AddParams(localFeatureVectors[proc]);
    }
    cerr << "master: done collecting all lambda features.  |lambda| = " << lambda->paramIndexes.size() << endl; 
  } else {
    //    cerr << "rank " << learningInfo.mpiWorld->rank() << ": sending my |paramIdsTemp| = " << lambda->paramIdsTemp.size() << "  to master ... ";
    mpi::gather< std::vector< FeatureId > >(*learningInfo.mpiWorld, lambda->paramIdsTemp, 0);
  }

  // master seals his lambda params creating shared memory 
  if(learningInfo.mpiWorld->rank() == 0) {
    assert(lambda->paramIdsTemp.size() == lambda->paramWeightsTemp.size());
    assert(lambda->paramIdsTemp.size() > 0);
    assert(lambda->paramIdsTemp.size() == lambda->paramIndexes.size());
    assert(lambda->paramIdsPtr == 0 && lambda->paramWeightsPtr == 0);
    lambda->Seal();
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
    assert(lambda->paramIdsPtr == 0 && lambda->paramWeightsPtr == 0);
    lambda->Seal();
    assert(lambda->paramIdsTemp.size() == 0 && lambda->paramWeightsTemp.size() == 0);
    assert(lambda->paramIdsPtr != 0 && lambda->paramWeightsPtr != 0 \
           && lambda->paramIdsPtr->size() == lambda->paramWeightsPtr->size() \
           && lambda->paramIdsPtr->size() == lambda->paramIndexes.size());    
  }
}

string LatentCrfModel::GetThetaFilename(int iteration) {
  stringstream thetaParamsFilename;
  thetaParamsFilename << outputPrefix << "." << iteration << ".theta";
  return thetaParamsFilename.str();
}

string LatentCrfModel::GetLambdaFilename(int iteration, bool humane) {
  stringstream lambdaParamsFilename;
  lambdaParamsFilename << outputPrefix << "." << iteration << ".lambda";
  if(humane) {
    lambdaParamsFilename << ".humane";
  }
  return lambdaParamsFilename.str();
}

int64_t LatentCrfModel::GetContextOfTheta(unsigned sentId, int y) {
  cerr << "int64_t LatentCrfModel::GetContextOfTheta(unsigned sentId, int y) not implemented" << endl;
  assert(false);
}

// returns -log p(z|x)
double LatentCrfModel::UpdateThetaMleForSent(const unsigned sentId, 
					     MultinomialParams::ConditionalMultinomialParam<int64_t> &mle, 
					     boost::unordered_map<int64_t, double> &mleMarginals) {
  assert(sentId < examplesCount);
  // build the FSTs
  fst::VectorFst<FstUtils::LogArc> thetaLambdaFst;
  fst::VectorFst<FstUtils::LogArc> lambdaFst;
  std::vector<FstUtils::LogWeight> thetaLambdaAlphas, lambdaAlphas, thetaLambdaBetas, lambdaBetas;
  BuildThetaLambdaFst(sentId, GetReconstructedObservableSequence(sentId), thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas);
  BuildLambdaFst(sentId, lambdaFst, lambdaAlphas, lambdaBetas, NULL, NULL);
  // compute the B matrix for this sentence
  boost::unordered_map< int64_t, boost::unordered_map< int64_t, LogVal<double> > > B;
  B.clear();
  ComputeB(sentId, this->GetReconstructedObservableSequence(sentId), thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas, B);
  // compute the C value for this sentence
  double nLogC = ComputeNLogC(thetaLambdaFst, thetaLambdaBetas);
  double nLogZ = ComputeNLogZ_lambda(lambdaFst, lambdaBetas);
  double nLogP_ZGivenX = nLogC - nLogZ;
  // update mle for each z^*|y^* fired
  for(auto yIter = B.begin(); yIter != B.end(); yIter++) {
    int context = GetContextOfTheta(sentId, yIter->first);
    for(auto zIter = yIter->second.begin(); zIter != yIter->second.end(); zIter++) {
      int64_t z_ = zIter->first;
      double nLogb = -log<double>(zIter->second);
      assert(zIter->second.s_ == false); //  all B values are supposed to be positive
      double bOverC = MultinomialParams::nExp(nLogb - nLogC);
      assert(bOverC > -0.001);

      if(learningInfo.useEarlyStopping && sentId % 10 == 0) {
        bOverC = 0.0;
      }

      mle[context][z_] += bOverC;
      mleMarginals[context] += bOverC;
    }
  }
  return nLogP_ZGivenX;
}

// returns -log p(z|x)
// TODO: we don't need the lambdaFst. the return value of this function is just used for debugging.
double LatentCrfModel::UpdateThetaMleForSent(const unsigned sentId, 
					     MultinomialParams::ConditionalMultinomialParam<pair<int64_t,int64_t> > &mle, 
					     boost::unordered_map< pair<int64_t, int64_t> , double> &mleMarginals) {
  assert(sentId < examplesCount);
  // build the FSTs
  fst::VectorFst<FstUtils::LogArc> thetaLambdaFst;
  fst::VectorFst<FstUtils::LogArc> lambdaFst;
  std::vector<FstUtils::LogWeight> thetaLambdaAlphas, lambdaAlphas, thetaLambdaBetas, lambdaBetas;
  BuildThetaLambdaFst(sentId, GetReconstructedObservableSequence(sentId), thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas);
  BuildLambdaFst(sentId, lambdaFst, lambdaAlphas, lambdaBetas);
  // compute the B matrix for this sentence
  boost::unordered_map< pair<int64_t, int64_t>, boost::unordered_map< int64_t, LogVal<double> > > B;
  B.clear();
  ComputeB(sentId, this->GetReconstructedObservableSequence(sentId), thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas, B);
  // compute the C value for this sentence
  double nLogC = ComputeNLogC(thetaLambdaFst, thetaLambdaBetas);
  //  cerr << "C = " << MultinomialParams::nExp(nLogC) << endl;
  double nLogZ = ComputeNLogZ_lambda(lambdaFst, lambdaBetas);
  double nLogP_ZGivenX = nLogC - nLogZ;
  //cerr << "nloglikelihood += " << nLogC << endl;
  // update mle for each z^*|y^* fired
  for(auto yIter = B.begin(); yIter != B.end(); yIter++) {
    const pair<int64_t, int64_t> &y_ = yIter->first;
    for(auto zIter = yIter->second.begin(); zIter != yIter->second.end(); zIter++) {
      int64_t z_ = zIter->first;
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




