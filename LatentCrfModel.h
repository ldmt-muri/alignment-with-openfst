#ifndef _AUTO_ENCODER_H_
#define _AUTO_ENCODER_H_

#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <set>
#include <algorithm>

#include "StringUtils.h"
#include "FstUtils.h"
#include "LbfgsUtils.h"
#include "LogLinearParams.h"
#include "MultinomialParams.h"

using namespace fst;
using namespace std;

// implements the model described at doc/LatentCrfModel.tex
class LatentCrfModel {

  LatentCrfModel(const string &textFilename, 
		 const string &outputPrefix, 
		 LearningInfo &learningInfo);
  
  static LatentCrfModel *instance;

  // optimize the likelihood with block coordinate descent
  void BlockCoordinateDescent();

  // make sure all lambda features which may fire on this training data are added to lambda.params
  void WarmUp();

  // lbfgs call back function to compute the negative loglikelihood and its derivatives with respect to lambdas
  static double EvaluateNLogLikelihoodDerivativeWRTLambda(void *ptrFromSentId,
							  const double *lambdasArray,
							  double *gradient,
							  const int lambdasCount,
							  const double step);
  
  // lbfgs call back functiont to report optimizaton progress 
  static int LbfgsProgressReport(void *instance,
				 const lbfgsfloatval_t *x, 
				 const lbfgsfloatval_t *g,
				 const lbfgsfloatval_t fx,
				 const lbfgsfloatval_t xnorm,
				 const lbfgsfloatval_t gnorm,
				 const lbfgsfloatval_t step,
				 int n,
				 int k,
				 int ls);
  
  // builds an FST to computes B(x,z)
  void BuildThetaLambdaFst(const vector<int> &x, const vector<int> &z, 
			   VectorFst<LogArc> &fst, vector<fst::LogWeight>& alphas, vector<fst::LogWeight>& betas);

  // build an FST to compute Z(x)
  void BuildLambdaFst(const vector<int> &x, VectorFst<LogArc> &fst, vector<fst::LogWeight> &alphas, vector<fst::LogWeight> &betas);

  // compute the partition function Z_\lambda(x)
  double ComputeNLogZ_lambda(const vector<int> &x); // much slower
  double ComputeNLogZ_lambda(const VectorFst<LogArc> &fst, const vector<fst::LogWeight> &betas); // much faster

  // compute B(x,z) which can be indexed as: BXZ[y^*][z^*] to give B(x, z, z^*, y^*)
  // assumptions: BXZ is cleared
  void ComputeB(const vector<int> &x, const vector<int> &z, 
		const VectorFst<LogArc> &fst, 
		const vector<fst::LogWeight> &alphas, const vector<fst::LogWeight> &betas, 
		map< int, map< int, double > > &BXZ);

  // assumptions: 
  // - fst is populated using BuildLambdaFst()
  // - FXZk is cleared
  void ComputeF(const vector<int> &x, 
		const VectorFst<LogArc> &fst,
		const vector<fst::LogWeight> &alphas, const vector<fst::LogWeight> &betas,
		map<string, double> &FXZk);

  // assumptions: 
  // - fst is populated using BuildThetaLambdaFst()
  // - DXZk is cleared
  void ComputeD(const vector<int> &x, const vector<int> &z,
		const VectorFst<LogArc> &fst,
		const vector<fst::LogWeight> &alphas, const vector<fst::LogWeight> &betas,
		map<string, double> &DXZk);
    
  // assumptions:
  // - fst, betas are populated using BuildThetaLambdaFst()
  double ComputeNLogC(const VectorFst<LogArc> &fst,
		      const vector<fst::LogWeight> &betas);
    
  // compute p(y, z | x) = \frac{\prod_i \theta_{z_i|y_i} \exp \lambda h(y_i, y_{i-1}, x, i)}{Z_\lambda(x)}
  double ComputeNLogPrYZGivenX(vector<int>& x, vector<int>& y, vector<int>& z);

  // copute p(y | x, z) = \frac  {\prod_i \theta_{z_i|y_i} \exp \lambda h(y_i, y_{i-1}, x, i)} 
  //                             -------------------------------------------
  //                             {\sum_y' \prod_i \theta_{z_i|y'_i} \exp \lambda h(y'_i, y'_{i-1}, x, i)}
  double ComputeNLogPrYGivenXZ(vector<int> &x, vector<int> &y, vector<int> &z);
    
  double ComputeCorpusNloglikelihood();

 public:

  static LatentCrfModel& GetInstance();

  static LatentCrfModel& GetInstance(const string &textFilename, 
				     const string &outputPrefix, 
				     LearningInfo &learningInfo);

  
  // train the model
  void Train();

 private:
  VocabEncoder vocabEncoder;
  int START_OF_SENTENCE_Y_VALUE;
  string textFilename, outputPrefix;
  vector<vector<int> > data;
  set<int> xDomain, yDomain;
  LogLinearParams lambda;
  MultinomialParams::ConditionalMultinomialParam nLogTheta;
  LearningInfo learningInfo;
  // vectors specifiying which feature types to use (initialized in the constructor)
  std::vector<bool> enabledFeatureTypes;
};

#endif
