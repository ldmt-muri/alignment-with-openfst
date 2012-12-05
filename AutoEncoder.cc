#include "AutoEncoder.h"

using namespace std;
using namespace fst;
using namespace OptUtils;


// initialize model weights to zeros
AutoEncoder::AutoEncoder(const string &textFilename, const string &outputPrefix) {
  // set member variables
  this->textFilename = textFilename;
  this->outputPrefix = outputPrefix;

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
}

// compute the partition function Z_\lambda(x)
double AutoEncoder::ComputeNLogZ_lambda(vector<int>& x) {
  // build an FST to compute Z(x) = \sum_y \prod_i \exp \lambda h(y_i, y_{i-1}, x, i)
  // arcs represent a particular choice of y_i at time step i
  // arc weights are \lambda h(y_i, y_{i-1}, x, i)
  VectorFst<LogArc> fst;
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
	    fst.SetFinal(toState, LogWeight::One);
	  }
	} else {
	  toState = yIToState[yI];
	}
	// now add the arc
	fst.AddArc(fromState, fst::LogArc(yIM1, yI, lambdaH, toState));	
      }
    }
    // now, that all states reached in step i have already been created, yIM1ToState has become irrelevant
    yIM1ToState.assign(yIToState);
  }

  // now compute the path sum, i.e. log Z_\lambda(x)
  vector<float> distancesToFinal;
  ShortestDistance(fst, distancesToFinal, true);

  // we return -log Z_\lambda(x)
  return distancesToFinal[startState];
}

// compute p(y, z | x) = \frac{\prod_i \theta_{z_i|y_i} \exp \lambda h(y_i, y_{i-1}, x, i)}{Z_\lambda(x)}
double ComputeNLogPrYZGivenX(vector<int>& x, vector<int>& y, vector<int>& z) {
  assert(x.size() == y.size());
  assert(x.size() == z.size());

  // initialize prob = 1.0
  double result = 0;

  // divide by Z_\lambda(x)
  result -= ComputeNLogZ_lambda(x);

  for(int i = 0; i < x.size(); i++) {

    // multiply \theta_{z_i|y_i} (which is already stored using in its -log value)
    result += theta[y[i]][z[i]];

    // multiply \exp \lambda h(y_i, y_{i-1}, x, i)
    //  compute h(y_i, y_{i-1}, x, i)
    map<string, float> h;
    lambda.FireFeatures(yI, yIM1, x, i, enabledFeatureTypes, h);
    //  compute \lambda h(y_i, y_{i-1}, x, i) , multiply by -1 to be consistent with the -log probability representation
    double lambdaH = -1 * lambda.DotProduct(h);
    result += lambdaH;
  }

  return result;
}

// copute p(y | x, z) = \frac  {\prod_i \theta_{z_i|y_i} \exp \lambda h(y_i, y_{i-1}, x, i)} 
//                             -------------------------------------------
//                             {\sum_y' \prod_i \theta_{z_i|y'_i} \exp \lambda h(y'_i, y'_{i-1}, x, i)}
double ComputeNLogPrYGivenXZ(vector<int> &x, vector<int> &y, vector<int> &z) {
  assert(x.size() == y.size());
  assert(x.size() == z.size());

  double result = 0;

  // multiply the numerator
  for(int i = 0; i < x.size(); i++) {

    // multiply \theta_{z_i|y_i} (which is already stored in its -log value)
    result += theta[y[i]][z[i]];

    // multiply \exp \lambda h(y_i, y_{i-1}, x, i)
    //  compute h(y_i, y_{i-1}, x, i)
    map<string, float> h;
    lambda.FireFeatures(yI, yIM1, x, i, enabledFeatureTypes, h);
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
      for(set<int>::const_iterator yDomainIter = yDomain.begin();
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
	double weight = lambdaH + theta[yI][z[i]];
	// determine whether to add a new state or reuse an existing state which also represent label y_i and timestep i
	int toState;	
	if(yIToState.count(yI) == 0) {
	  toState = fst.AddState();
	  yIToState[yI] = toState;
	  // is it a final state?
	  if(i == x.size() - 1) {
	    fst.SetFinal(toState, LogWeight::One);
	  }
	} else {
	  toState = yIToState[yI];
	}
	// now add the arc
	fst.AddArc(fromState, fst::LogArc(yIM1, yI, weight, toState));	
      }
    }
    // now, that all states reached in step i have already been created, yIM1ToState has become irrelevant
    yIM1ToState.assign(yIToState);
  }

  //  now compute the path sum, i.e. -\log [ \sum_y' \prod_i \theta_{z_i|y'_i} \exp \lambda h(y'_i, y'_{i-1}, x, i) ]
  vector<float> distancesToFinal;
  ShortestDistance(fst, distancesToFinal, true);

  //  finally, divide by the denominator
  double denominator = distancesToFinal[startState];
  result -= denominator;

  // return p(y | x, z)
  return result;
}
