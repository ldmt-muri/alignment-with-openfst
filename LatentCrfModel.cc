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

// initialize model weights to zeros
LatentCrfModel::LatentCrfModel(const string &textFilename, const string &outputPrefix, LearningInfo &learningInfo) : 
  lambda(*learningInfo.srcVocabDecoder),
  vocabEncoder(textFilename) {

  // set member variables
  this->textFilename = textFilename;
  this->outputPrefix = outputPrefix;
  this->learningInfo = learningInfo;

  // set constants
  this->START_OF_SENTENCE_Y_VALUE = 2;

  // POS tag yDomain
  this->yDomain.insert(START_OF_SENTENCE_Y_VALUE); // the conceptual yValue of word at position -1 in a sentence
  this->yDomain.insert(3); // noun
  this->yDomain.insert(4); // verb
  /*  this->yDomain.insert(5); // adjective
  this->yDomain.insert(6); // adverb
  this->yDomain.insert(7); // pronoun
  this->yDomain.insert(8); // determiner/article
  this->yDomain.insert(9); // preposition/postposition
  this->yDomain.insert(10); // numerals
  this->yDomain.insert(11); // conjunctions
  this->yDomain.insert(12); // particles
  this->yDomain.insert(13); // punctuation marks
  this->yDomain.insert(14); // others (e.g. abbreviations, foreign words ...etc)
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
  // features 51-70 are reserved for latentCrf model
  for(int i = 51; i < 70; i++) {
    enabledFeatureTypes.push_back(true);
  }

  // initialize the theta params to unnormalized uniform
  nLogTheta.clear();
  for(set<int>::const_iterator yDomainIter = yDomain.begin(); yDomainIter != yDomain.end(); yDomainIter++) {
    for(set<int>::const_iterator zDomainIter = xDomain.begin(); zDomainIter != xDomain.end(); zDomainIter++) {
      nLogTheta[*yDomainIter][*zDomainIter] = 1;
    }
  }
  // then normalize
  MultinomialParams::NormalizeParams(nLogTheta);

  // lambdas are initialized to all zeros
  assert(lambda.GetParamsCount() == 0);
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
	// compute h(y_i, y_{i-1}, x, i)
	map<string, double> h;
	lambda.FireFeatures(yI, yIM1, x, i, enabledFeatureTypes, h);
	// compute the weight of this transition:
	// \lambda h(y_i, y_{i-1}, x, i), and multiply by -1 to be consistent with the -log probability representation
	double nLambdaH = -1.0 * lambda.DotProduct(h);
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
}

// assumptions: 
// - fst is populated using BuildLambdaFst()
// - FXZk is cleared
void LatentCrfModel::ComputeF(const vector<int> &x,
			      const VectorFst<LogArc> &fst,
			      const vector<fst::LogWeight> &alphas, const vector<fst::LogWeight> &betas,
			      map<string, double> &FXZk) {
  
  assert(FXZk.size() == 0);
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
	map<string, double> h;
	lambda.FireFeatures(yI, yIM1, x, i, enabledFeatureTypes, h);
	for(map<string, double>::const_iterator h_k = h.begin(); h_k != h.end(); h_k++) {

	  //	  cerr << "      featureId=" << h_k->first << " value=" << h_k->second << endl;

	  // add the arc's h_k feature value weighted by the marginal weight of passing through this arc
	  if(FXZk.count(h_k->first) == 0) {
	    FXZk[h_k->first] = 0;
	  }
	  //cerr << FXZk[h_k->first];
	  FXZk[h_k->first] += MultinomialParams::nExp(nLogMarginal) * h_k->second;
	  //cerr << " => " << FXZk[h_k->first] << endl;
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
// - fst is populated using BuildThetaLambdaFst()
// - DXZk is cleared
void LatentCrfModel::ComputeD(const vector<int> &x, const vector<int> &z, 
			   const VectorFst<LogArc> &fst,
			   const vector<fst::LogWeight> &alphas, const vector<fst::LogWeight> &betas,
			   map<string, double> &DXZk) {

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
	map<string, double> h;
	lambda.FireFeatures(yI, yIM1, x, i, enabledFeatureTypes, h);
	for(map<string, double>::const_iterator h_k = h.begin(); h_k != h.end(); h_k++) {

	  // add the arc's h_k feature value weighted by the marginal weight of passing through this arc
	  if(DXZk.count(h_k->first) == 0) {
	    DXZk[h_k->first] = 0;
	  }
	  //cerr << DXZk[h_k->first];
	  DXZk[h_k->first] += MultinomialParams::nExp(nLogMarginal) * h_k->second;
	  //cerr << " => " << DXZk[h_k->first] << endl;
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
  
  assert(BXZ.size() == 0);

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
}

// build an FST which path sums to 
// -log \sum_y [ \prod_i \theta_{z_i\mid y_i} e^{\lambda h(y_i, y_{i-1}, x, i)} ]
void LatentCrfModel::BuildThetaLambdaFst(const vector<int> &x, const vector<int> &z, VectorFst<LogArc> &fst, vector<fst::LogWeight> &alphas, vector<fst::LogWeight> &betas) {

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
	// compute h(y_i, y_{i-1}, x, i)
	map<string, double> h;
	lambda.FireFeatures(yI, yIM1, x, i, enabledFeatureTypes, h);

	// prepare -log \theta_{z_i|y_i}
	int zI = z[i];
	double nLogTheta_zI_yI = this->nLogTheta[yI][zI];

	// compute the weight of this transition: \lambda h(y_i, y_{i-1}, x, i), and multiply by -1 to be consistent with the -log probability representatio
	double nLambdaH = -1.0 * lambda.DotProduct(h);
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
    lambda.FireFeatures(y[i], y[i-1], x, i, enabledFeatureTypes, h);
    //  compute \lambda h(y_i, y_{i-1}, x, i) , multiply by -1 to be consistent with the -log probability representation
    double nlambdaH = -1 * lambda.DotProduct(h);
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
    lambda.FireFeatures(y[i], y[i-1], x, i, enabledFeatureTypes, h);
    //  compute \lambda h(y_i, y_{i-1}, x, i)
    double lambdaH = -1 * lambda.DotProduct(h);
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
	// compute h(y_i, y_{i-1}, x, i)
	map<string, double> h;
	lambda.FireFeatures(yI, yIM1, x, i, enabledFeatureTypes, h);
	// \lambda h(...,i)
	double lambdaH = -1.0 * lambda.DotProduct(h);
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
  if(learningInfo.optimizationMethod.algorithm == BLOCK_COORD_DESCENT) {
    BlockCoordinateDescent();
  } else {
    assert(false);
  }
}

// a call back function that computes the gradient and the objective function for the lbfgs minimizer
double LatentCrfModel::EvaluateNLogLikelihoodDerivativeWRTLambda(void *ptrFromSentId,
			     const double *lambdasArray,
			     double *gradient,
			     const int lambdasCount,
			     const double step) {
  LatentCrfModel &model = LatentCrfModel::GetInstance();

  // update the model parameters, temporarily, so that we can compute the derivative at the required values
  model.lambda.UpdateParams(lambdasArray, lambdasCount);

  // debug
  //  cerr << "lbfgs suggests the following lambda parameter weights" << endl;
  //  model.lambda.PrintParams();
  

  // for each sentence in this mini batch, aggregate the nloglikelihood and its derivatives across sentences
  double nlogLikelihood = 0;
  map<string, double> derivativeWRTLambda;
  int fromSentId = *((int*)ptrFromSentId);
  for(int sentId = fromSentId; sentId < min((int)model.data.size(), fromSentId + model.learningInfo.optimizationMethod.miniBatchSize); sentId++) {
    // build the FSTs
    VectorFst<LogArc> thetaLambdaFst, lambdaFst;
    vector<fst::LogWeight> thetaLambdaAlphas, lambdaAlphas, thetaLambdaBetas, lambdaBetas;
    model.BuildThetaLambdaFst(model.data[sentId], model.data[sentId], thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas);
    model.BuildLambdaFst(model.data[sentId], lambdaFst, lambdaAlphas, lambdaBetas);
    // compute the D map for this sentence
    map<string, double> D;
    model.ComputeD(model.data[sentId], model.data[sentId], thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas, D);      
    // compute the C value for this sentence
    double nLogC = model.ComputeNLogC(thetaLambdaFst, thetaLambdaBetas);
    // update the loglikelihood
    nlogLikelihood += nLogC;
    // add D/C to the gradient
    for(map<string, double>::const_iterator dIter = D.begin(); dIter != D.end(); dIter++) {
      double d = dIter->second;
      double nLogd = MultinomialParams::nLog(d);
      double dOverC = MultinomialParams::nExp(nLogd - nLogC);
      derivativeWRTLambda[dIter->first] -= dOverC;
    }
    // compute the F map fro this sentence
    map<string, double> F;
    model.ComputeF(model.data[sentId], lambdaFst, lambdaAlphas, lambdaBetas, F);
    // compute the Z value for this sentence
    double nLogZ = model.ComputeNLogZ_lambda(lambdaFst, lambdaBetas);
    // update the log likelihood
    nlogLikelihood -= nLogZ;
    //      cerr << "nloglikelihood -= " << nLogZ << ", |x| = " << data[sentId].size() << endl;
    // subtract F/Z from the gradient
    for(map<string, double>::const_iterator fIter = F.begin(); fIter != F.end(); fIter++) {
      double f = fIter->second;
      double nLogf = MultinomialParams::nLog(f);
      double fOverZ = MultinomialParams::nExp(nLogf - nLogZ);
      derivativeWRTLambda[fIter->first] += fOverZ;
    }
  }
  // debug
  //  cerr << "nloglikelihood derivative wrt lambdas: " << endl;
  //  LogLinearParams::PrintParams(derivativeWRTLambda);

  // write the gradient in the (hopefully) pre-allocated array 'gradient'
  model.lambda.ConvertFeatureMapToFeatureArray(derivativeWRTLambda, gradient);
  // return the to-be-minimized objective function
  cerr << "Evaluate returning " << nlogLikelihood << ". step is " << step << endl;
  //  cerr << "===================================" << endl;
  //  cerr << "gradient: ";
  //  for(map<string, double>::const_iterator gradientIter = derivativeWRTLambda.begin(); 
  //      gradientIter != derivativeWRTLambda.end(); gradientIter++) {
  //    cerr << gradientIter->first << ":" << gradientIter->second << " ";
  //  }
  //  cerr << endl;
  return nlogLikelihood;
}

int LatentCrfModel::LbfgsProgressReport(void *instance,
				     const lbfgsfloatval_t *x, 
				     const lbfgsfloatval_t *g,
				     const lbfgsfloatval_t fx,
				     const lbfgsfloatval_t xnorm,
				     const lbfgsfloatval_t gnorm,
				     const lbfgsfloatval_t step,
				     int n,
				     int k,
				     int ls) {
  cerr << "lbfgs Iteration " << k << ": nLogLikelihood = " << fx << ",xnorm = " << xnorm << ",gnorm = " << gnorm << ",step = " << step << endl;
  return 0;
}

// make sure all features which may fire on this training data have a corresponding parameter in lambda (member)
void LatentCrfModel::WarmUp() {
  UniformSampler uniform;
  cerr << "warming up..." << endl;
  //  cerr << "lambda.GetParamsCount() = " << lambda.GetParamsCount() << endl;
  for(int sentId = 0; sentId < data.size(); sentId++) {
    //    cerr << "now processing sent# " << sentId << endl;
    // build the FST
    VectorFst<LogArc> lambdaFst;
    vector<fst::LogWeight> lambdaAlphas, lambdaBetas;
    BuildLambdaFst(data[sentId], lambdaFst, lambdaAlphas, lambdaBetas);
    //    cerr << "lambda fst successfully built" << endl;
    // compute the F map fro this sentence
    map<string, double> F;
    ComputeF(data[sentId], lambdaFst, lambdaAlphas, lambdaBetas, F);
    //    cerr << "F successfully computed" << endl;
    // add each feature fired on any sentence to the lambda parameters
    for(map<string, double>::const_iterator fIter = F.begin(); fIter != F.end(); fIter++) {
      lambda.UpdateParam(fIter->first, uniform.Draw() - 0.5);
    }
  }
  cerr << "lambdas initialized to: " << endl;
  lambda.PrintParams();
  cerr << "done" << endl;
}

void LatentCrfModel::BlockCoordinateDescent() {  
  
  // add all features in this data set to lambda.params
  WarmUp();

  do {
    // log likelihood
    double nlogLikelihood = 0;

    // update the thetas by normalizing soft counts (i.e. the closed form solution)
    MultinomialParams::ConditionalMultinomialParam mle;
    for(int sentId = 0; sentId < data.size(); sentId++) {
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
	  mle[y_][z_] += bOverC;
	}
      }
    }
    // debug
    //    cerr << "mle (before normalization):" << endl;
    //    MultinomialParams::PrintParams(mle);
    //    cerr << "=======================================" << endl;
    // now, normalize the MLE estimates
    MultinomialParams::NormalizeParams(mle);
    // update the thetas
    nLogTheta = mle;
    // debug
    cerr << "nlog theta params:" << endl;
    MultinomialParams::PrintParams(nLogTheta);
    cerr << "=======================================" << endl;

    // update the lambdas with mini-batch lbfgs, one full pass over training data
    // needed to call liblbfgs
    double* lambdasArray;
    int lambdasArrayLength;
    // lbfgs configurations
    lbfgs_parameter_t lbfgsParams;
    lbfgs_parameter_init(&lbfgsParams);
    assert(learningInfo.optimizationMethod.subOptMethod != 0);
    lbfgsParams.max_iterations = learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxIterations;
    // for each mini-batch
    cerr << "minibatch size = " << learningInfo.optimizationMethod.subOptMethod->miniBatchSize << endl;
    for(int sentId = 0; sentId < data.size(); sentId += learningInfo.optimizationMethod.subOptMethod->miniBatchSize) {

      // populate lambdasArray and lambasArrayLength
      lambdasArray = lambda.GetParamWeightsArray();
      lambdasArrayLength = lambda.GetParamsCount();
      // call the lbfgs minimizer for this mini-batch
      double optimizedMiniBatchNLogLikelihood = 0;
      int lbfgsStatus = lbfgs(lambdasArrayLength, lambdasArray, &optimizedMiniBatchNLogLikelihood, 
			      EvaluateNLogLikelihoodDerivativeWRTLambda, LbfgsProgressReport, &sentId, &lbfgsParams);

      // debug
      cerr << "lbfgsStatusCode = " << LbfgsUtils::LbfgsStatusIntToString(lbfgsStatus) << " = " << lbfgsStatus << endl;
      if(lbfgsStatus == LBFGSERR_ROUNDING_ERROR) {
	cerr << "rounding error (" << lbfgsStatus << "). it seems like my gradient is buggy." << endl << "retry..." << endl;
	lbfgsStatus = lbfgs(lambdasArrayLength, lambdasArray, &optimizedMiniBatchNLogLikelihood,
			    EvaluateNLogLikelihoodDerivativeWRTLambda, LbfgsProgressReport, &sentId, &lbfgsParams);
	cerr << "lbfgsStatusCode = " << LbfgsUtils::LbfgsStatusIntToString(lbfgsStatus) << " = " << lbfgsStatus << endl;
      }
    
      // update iteration's nloglikelihood
      nlogLikelihood += optimizedMiniBatchNLogLikelihood;
    }
    
    // debug
    cerr << "lambda params:" << endl;
    lambda.PrintParams();
    cerr << "=======================================" << endl;

    // debug
    cerr << "coordinate descent iteration #" << learningInfo.iterationsCount << " nloglikelihood=" << nlogLikelihood << endl;
    
    // update learningInfo
    learningInfo.logLikelihood.push_back(nlogLikelihood);
    learningInfo.iterationsCount++;

    // check convergence
  } while(!learningInfo.IsModelConverged());

  // debug
  lambda.PersistParams(outputPrefix + string(".lambda"));
  MultinomialParams::PersistParams(outputPrefix + string(".theta"), nLogTheta);
  vocabEncoder.PersistVocab(outputPrefix + string(".vocab"));
}
