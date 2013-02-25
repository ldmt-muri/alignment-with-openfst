#ifndef _HMM_MODEL2_H_
#define _HMM_MODEL2_H_

#include <iostream>
#include <fstream>
#include <math.h>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/thread/thread.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/nonblocking.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>
#include <boost/mpi/collectives.hpp>

#include "LearningInfo.h"
#include "StringUtils.h"
#include "FstUtils.h"
#include "IAlignmentModel.h"
#include "IAlignmentSampler.h"
#include "Samplers.h"
#include "MultinomialParams.h"
#include "UnsupervisedSequenceTaggingModel.h"

using namespace fst;
using namespace std;
using namespace MultinomialParams;

class HmmModel2 : public UnsupervisedSequenceTaggingModel {

  // normalizes the parameters such that \sum_t p(t|s) = 1 \forall s
  void NormalizeFractionalCounts();
  
  // zero all parameters
  void ClearFractionalCounts();

  // builds the lattice of all possible label sequences
  void BuildThetaGammaFst(vector<int> &x, VectorFst<LogArc> &fst);
  
  // builds the lattice of all possible label sequences, also computes potentials
  void BuildThetaGammaFst(unsigned sentId, VectorFst<LogArc> &fst, vector<fst::LogWeight> &alphas, vector<fst::LogWeight> &betas);
  
  // traverse each transition on the fst and accumulate the mle counts of theta and gamma
  void UpdateMle(const unsigned sentId,
		 const VectorFst<LogArc> &fst, 
		 const vector<fst::LogWeight> &alphas, 
		 const vector<fst::LogWeight> &betas, 
		 ConditionalMultinomialParam<int> &thetaMle, 
		 ConditionalMultinomialParam<int> &gammaMle);
 
  void InitParams();
  
 public:
  
  HmmModel2(const string &textFilename, 
	    const string &outputPrefix, 
	    LearningInfo &learningInfo,
	    unsigned numberOfLabels);

  ~HmmModel2() {}

  void PrintParams();
  
  void Train();
  
  void Label(vector<int> &tokens, vector<int> &labels);
  
 private:
  
  // constants
  const int START_OF_SENTENCE_Y_VALUE, FIRST_ALLOWED_LABEL_VALUE;
  
  // configurations
  LearningInfo learningInfo;
  
  // training data
  vector< vector<int> > observations;

  // gaussian sampler
  GaussianSampler gaussianSampler;

  // model parameters theta = emission probabilities, alpha = transition prbailibities
  ConditionalMultinomialParam<int> nlogTheta, nlogGamma;
  
  // output prefix
  string outputPrefix;
  
  // possible values x_i and y_i may take
  set<int> xDomain, yDomain;
  
};

#endif
