#include "AutoEncoder.h"

using namespace std;
using namespace fst;
using namespace OptUtils;


// initialize model weights to zeros
AutoEncoder::AutoEncoder(const string &textFilename, const string &outputPrefix, LearningInfo &learningInfo) : lambda(*learningInfo.srcVocabDecoder) {

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
  // zero is reserved for FST epsilon
  assert(this->yDomain.count(0) == 0);

  // words xDomain
  VocabEncoder vocabEncoder(textFilename);
  for(map<int,string>::const_iterator vocabIter = vocabEncoder.intToToken.begin();
      vocabIter != vocabEncoder.intToToken.end();
      vocabIter++) {
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
}

// compute the partition function Z_\lambda(x)
// assumptions:
// - fst and betas are populated using BuildLambdaFst()
double AutoEncoder::ComputeNLogZ_lambda(const VectorFst<LogArc> &fst, const vector<fst::LogWeight> &betas) {
  return betas[fst.Start()].Value();
}

// compute the partition function Z_\lambda(x)
double AutoEncoder::ComputeNLogZ_lambda(const vector<int> &x) {
  VectorFst<LogArc> fst;
  vector<fst::LogWeight> alphas;
  vector<fst::LogWeight> betas;
  BuildLambdaFst(x, fst, alphas, betas);
  return ComputeNLogZ_lambda(fst, betas);
}

// build an FST to compute Z(x) = \sum_y \prod_i \exp \lambda h(y_i, y_{i-1}, x, i)
void AutoEncoder::BuildLambdaFst(const vector<int> &x, VectorFst<LogArc> &fst, vector<fst::LogWeight> &alphas, vector<fst::LogWeight> &betas) {
  // arcs represent a particular choice of y_i at time step i
  // arc weights are -\lambda h(y_i, y_{i-1}, x, i)
  assert(fst.NumStates() == 0);
  int startState = fst.AddState();
  fst.SetStart(startState);
  
  // map values of y_{i-1} and y_i to fst states
  map<int, int> yIM1ToState, yIToState;
  assert(yIM1ToState.size() == 0 && yIToState.size());
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
	map<string, float> h;
	lambda.FireFeatures(yI, yIM1, x, i, enabledFeatureTypes, h);
	// compute the weight of this transition: \lambda h(y_i, y_{i-1}, x, i), and multiply by -1 to be consistent with the -log probability representation
	float lambdaH = -1.0 * lambda.DotProduct(h);
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
	fst.AddArc(fromState, fst::LogArc(yIM1, yI, lambdaH, toState));	
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
void AutoEncoder::ComputeF(const vector<int> &x,
			   const VectorFst<LogArc> &fst,
			   const vector<fst::LogWeight> &alphas, const vector<fst::LogWeight> &betas,
			   map<string, double> &FXZk) {

  // schedule for visiting states such that we know the timestep for each arc
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
      for(ArcIterator< VectorFst<LogArc> > aiter(fst, fromState); !aiter.Done(); aiter.Next()) {
	LogArc arc = aiter.Value();
	int yIM1 = arc.ilabel;
	int yI = arc.olabel;
	float arcWeight = arc.weight.Value();
	int toState = arc.nextstate;

	// compute marginal weight of passing on this arc
	float nLogMarginal = alphas[fromState].Value() + betas[toState].Value() + arcWeight;

	// for each feature that fires on this arc
	map<string, float> h;
	lambda.FireFeatures(yI, yIM1, x, i, enabledFeatureTypes, h);
	for(map<string, float>::const_iterator h_k = h.begin(); h_k != h.end(); h_k++) {

	  // add the arc's h_k feature value weighted by the marginal weight of passing through this arc
	  cerr << FXZk[h_k->first] << endl;
	  FXZk[h_k->first] += MultinomialParams::nExp(nLogMarginal) * h_k->second;
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
void AutoEncoder::ComputeD(const vector<int> &x, const vector<int> &z, 
			   const VectorFst<LogArc> &fst,
			   const vector<fst::LogWeight> &alphas, const vector<fst::LogWeight> &betas,
			   map<string, double> &DXZk) {

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
	float arcWeight = arc.weight.Value();
	int toState = arc.nextstate;

	// compute marginal weight of passing on this arc
	float nLogMarginal = alphas[fromState].Value() + betas[toState].Value() + arcWeight;

	// for each feature that fires on this arc
	map<string, float> h;
	lambda.FireFeatures(yI, yIM1, x, i, enabledFeatureTypes, h);
	for(map<string, float>::const_iterator h_k = h.begin(); h_k != h.end(); h_k++) {

	  // add the arc's h_k feature value weighted by the marginal weight of passing through this arc
	  cerr << DXZk[h_k->first] << endl;
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
}

// assumptions:
// - fst, betas are populated using BuildThetaLambdaFst()
double AutoEncoder::ComputeC(const VectorFst<LogArc> &fst,
			     const vector<fst::LogWeight> &betas) {
  double pathSum = betas[fst.Start()].Value();
  double C = MultinomialParams::nExp(pathSum);
  return C;
}

// compute B(x,z) which can be indexed as: BXZ[y^*][z^*] to give B(x, z, z^*, y^*)
// assumptions: 
// - BXZ is cleared
// - fst, alphas, and betas are populated using BuildThetaLambdaFst
void AutoEncoder::ComputeB(const vector<int> &x, const vector<int> &z, 
			   const VectorFst<LogArc> &fst, 
			   const vector<fst::LogWeight> &alphas, const vector<fst::LogWeight> &betas, 
			   map< int, map< int, double > > &BXZ) {
  // \sum_y [ \prod_i \theta_{z_i\mid y_i} e^{\lambda h(y_i, y_{i-1}, x, i)} ] \sum_i \delta_{y_i=y^*,z_i=z^*}
  
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
	float arcWeight = arc.weight.Value();
	int toState = arc.nextstate;

	// compute marginal weight of passing on this arc
	float nLogMarginal = alphas[fromState].Value() + betas[toState].Value() + arcWeight;

	// update the corresponding B value
	BXZ[yI][zI] += MultinomialParams::nExp(nLogMarginal);
	cerr << MultinomialParams::nExp(nLogMarginal) << " => " << BXZ[yI][zI];

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
void AutoEncoder::BuildThetaLambdaFst(const vector<int> &x, const vector<int> &z, VectorFst<LogArc> &fst, vector<fst::LogWeight> &alphas, vector<fst::LogWeight> &betas) {

  // arcs represent a particular choice of y_i at time step i
  // arc weights are -log \theta_{z_i|y_i} - \lambda h(y_i, y_{i-1}, x, i)
  assert(fst.NumStates() == 0);
  int startState = fst.AddState();
  fst.SetStart(startState);
  
  // map values of y_{i-1} and y_i to fst states
  map<int, int> yIM1ToState, yIToState;
  assert(yIM1ToState.size() == 0 && yIToState.size());
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
	map<string, float> h;
	lambda.FireFeatures(yI, yIM1, x, i, enabledFeatureTypes, h);

	// prepare -log \theta_{z_i|y_i}
	int zI = z[i];
	float nLogTheta_zI_yI = this->nLogTheta[yI][zI];

	// compute the weight of this transition: \lambda h(y_i, y_{i-1}, x, i), and multiply by -1 to be consistent with the -log probability representatio
	float nLambdaH = -1.0 * lambda.DotProduct(h);
	float weight = nLambdaH + nLogTheta_zI_yI;

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
double AutoEncoder::ComputeNLogPrYZGivenX(vector<int>& x, vector<int>& y, vector<int>& z) {
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
    map<string, float> h;
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
double AutoEncoder::ComputeNLogPrYGivenXZ(vector<int> &x, vector<int> &y, vector<int> &z) {
  assert(x.size() == y.size());
  assert(x.size() == z.size());

  double result = 0;

  // multiply the numerator
  for(int i = 0; i < x.size(); i++) {

    // multiply \theta_{z_i|y_i} (which is already stored in its -log value)
    result += nLogTheta[y[i]][z[i]];

    // multiply \exp \lambda h(y_i, y_{i-1}, x, i)
    //  compute h(y_i, y_{i-1}, x, i)
    map<string, float> h;
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
  assert(yIM1ToState.size() == 0 && yIToState.size());
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
	map<string, float> h;
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

void AutoEncoder::Train() {
  BlockCoordinateGradientDescent();
}

void AutoEncoder::BlockCoordinateGradientDescent() {

  do {

    // first, update the thetas
    MultinomialParams::ConditionalMultinomialParam mle;
    map< int, map< int, double > > B;
    B.clear();
    for(int sentId = 0; sentId < data.size(); sentId++) {
      // build the FST
      VectorFst<LogArc> thetaLambdaFst;
      vector<fst::LogWeight> alphas, betas;
      BuildThetaLambdaFst(data[sentId], data[sentId], thetaLambdaFst, alphas, betas);
      // compute the B matrix for this sentence
      ComputeB(this->data[sentId],
	       this->data[sentId], 
	       thetaLambdaFst, 
	       alphas, 
	       betas, 
	       B);
      // compute the C value for this sentence
      double C = ComputeC(thetaLambdaFst, betas);
      assert(C != 0.0);
      // update mle for each z^*|y^*
      for(map< int, map<int, double> >::const_iterator yIter = B.begin(); yIter != B.end(); yIter++) {
	int y_ = yIter->first;
	for(map<int, double>::const_iterator zIter = yIter->second.begin(); zIter != yIter->second.end(); zIter++) {
	  int z_ = zIter->first;
	  double b = zIter->second;
	  mle[y_][z_] += b / C;
	}
      }
    }
    // now, normalize the MLE estimates
    MultinomialParams::NormalizeParams(mle);
    // update the thetas
    nLogTheta = mle;

    // second, update the lambdas
    map<string, float> derivativeWRTLambda;
    for(int sentId = 0; sentId < data.size(); sentId++) {
      // build the FSTs
      VectorFst<LogArc> thetaLambdaFst, lambdaFst;
      vector<fst::LogWeight> thetaLambdaAlphas, lambdaAlphas, thetaLambdaBetas, lambdaBetas;
      BuildThetaLambdaFst(data[sentId], data[sentId], thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas);
      BuildLambdaFst(data[sentId], lambdaFst, lambdaAlphas, lambdaBetas);
      // compute the D map for this sentence
      map<string, double> D;
      ComputeD(data[sentId], data[sentId], thetaLambdaFst, thetaLambdaAlphas, thetaLambdaBetas, D);      
      // compute the C value for this sentence
      double C = ComputeC(thetaLambdaFst, thetaLambdaBetas);
      assert(C != 0.0);
      // add D/C to the gradient
      for(map<string, double>::const_iterator dIter = D.begin(); dIter != D.end(); dIter++) {
	double d = dIter->second;
	derivativeWRTLambda[dIter->first] += d/C;
      }
      // compute the F map fro this sentence
      map<string, double> F;
      ComputeF(data[sentId], thetaLambdaFst, lambdaAlphas, lambdaBetas, F);
      // compute the Z value for this sentence
      double nLogZ = ComputeNLogZ_lambda(lambdaFst, lambdaBetas);
      double Z = MultinomialParams::nExp(nLogZ);
      assert(Z != 0.0);
      // subtract F/Z from the gradient
      for(map<string, double>::const_iterator fIter = F.begin(); fIter != F.end(); fIter++) {
	double f = fIter->second;
	derivativeWRTLambda[fIter->first] -= f/Z;
      }
    }
    // now, that we computed the loglikelihood derivative wrt lambdas, update the lambdas
    // TODO: add an L1 regularizer
    OptUtils::OptMethod optimizationMethod;
    lambda.UpdateParams(derivativeWRTLambda, optimizationMethod);

    // TODO: update learningInfo during training so that the convergence method works appropriately
  } while(learningInfo.IsModelConverged());
}

