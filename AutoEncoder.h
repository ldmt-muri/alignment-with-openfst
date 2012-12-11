#ifndef _AUTO_ENCODER_H_
#define _AUTO_ENCODER_H_

#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <set>

#include "StringUtils.h"
#include "FstUtils.h"
#include "LogLinearParams.h"
#include "MultinomialParams.h"

using namespace fst;
using namespace std;

class AutoEncoder {

 public:

  AutoEncoder(const string &textFilename, 
	      const string &outputPrefix, 
	      LearningInfo &learningInfo);

  // compute the partition function Z_\lambda(x)
  double ComputeNLogZ_lambda(const vector<int> &x); // much slower
  double ComputeNLogZ_lambda(const VectorFst<LogArc> &fst, const vector<fst::LogWeight> &betas); // much faster

  // build an FST to compute \sum_y \prod_i \exp \lambda h(y_i, y_{i-1}, x, i)
  void BuildLambdaFst(const vector<int> &x, VectorFst<LogArc> &fst, vector<fst::LogWeight> &alphas, vector<fst::LogWeight> &betas);

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
  double ComputeC(const VectorFst<LogArc> &fst,
		  const vector<fst::LogWeight> &betas);
    
  // compute B(x,z) which can be indexed as: BXZ[y^*][z^*] to give B(x, z, z^*, y^*)
  // assumptions: 
  // - BXZ is cleared
  // - fst, alphas, and betas are populated using BuildThetaLambdaFst
  void BuildThetaLambdaFst(const vector<int> &x, const vector<int> &z, VectorFst<LogArc> &fst, vector<fst::LogWeight>& alphas, vector<fst::LogWeight>& betas);

  // compute p(y, z | x) = \frac{\prod_i \theta_{z_i|y_i} \exp \lambda h(y_i, y_{i-1}, x, i)}{Z_\lambda(x)}
  double ComputeNLogPrYZGivenX(vector<int>& x, vector<int>& y, vector<int>& z);

  // copute p(y | x, z) = \frac  {\prod_i \theta_{z_i|y_i} \exp \lambda h(y_i, y_{i-1}, x, i)} 
  //                             -------------------------------------------
  //                             {\sum_y' \prod_i \theta_{z_i|y'_i} \exp \lambda h(y'_i, y'_{i-1}, x, i)}
  double ComputeNLogPrYGivenXZ(vector<int> &x, vector<int> &y, vector<int> &z);
    
  // train the model
  void Train();

  // block coordinate gradient descent
  void BlockCoordinateGradientDescent();

 private:
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
