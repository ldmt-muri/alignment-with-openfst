#ifndef _AUTO_ENCODER_H_
#define _AUTO_ENCODER_H_

#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <set>
#include <algorithm>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/nonblocking.hpp>
#include <boost/thread/thread.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/exception/all.hpp>
#include <boost/exception/diagnostic_information.hpp> 
#include <boost/exception_ptr.hpp> 

#include "cdec-utils/logval.h"
#include "cdec-utils/semiring.h"
#define HAVE_BOOST_ARCHIVE_TEXT_OARCHIVE_HPP 1
#include "cdec-utils/fast_sparse_vector.h"

#include "anneal/Cpp/simann.hpp"

#include "ClustersComparer.h"
#include "StringUtils.h"
#include "FstUtils.h"
#include "LbfgsUtils.h"

#include "LogLinearParams.h"
#include "MultinomialParams.h"

namespace mpi = boost::mpi;

// implements the model described at doc/LatentCrfModel.tex
class LatentCrfModel {

  LatentCrfModel(const std::string &textFilename, 
		 const std::string &outputPrefix, 
		 LearningInfo &learningInfo,
		 unsigned numberOfLabels,
		 unsigned firstLabelId);
  
  ~LatentCrfModel();

  static LatentCrfModel *instance;

  void AddEnglishClosedVocab();

  // optimize the likelihood with block coordinate descent
  void BlockCoordinateDescent();

  // normalize soft counts with identical content to sum to one
  template <typename ContextType>
    void NormalizeThetaMle(MultinomialParams::ConditionalMultinomialParam<ContextType> &mle, 
			   std::map<ContextType, double> &mleMarginals) {
    // fix theta mle estimates
    for(typename std::map<ContextType, MultinomialParams::MultinomialParam >::const_iterator yIter = mle.params.begin(); yIter != mle.params.end(); yIter++) {
      ContextType y_ = yIter->first;
      double unnormalizedMarginalProbz_giveny_ = 0.0;
      // verify that \sum_z* mle[y*][z*] = mleMarginals[y*]
      for(MultinomialParams::MultinomialParam::const_iterator zIter = yIter->second.begin(); zIter != yIter->second.end(); zIter++) {
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
      for(MultinomialParams::MultinomialParam::const_iterator zIter = yIter->second.begin(); zIter != yIter->second.end(); zIter++) {
	int z_ = zIter->first;
	double normalizedProbz_giveny_ = zIter->second / mleMarginals[y_];
	mle[y_][z_] = normalizedProbz_giveny_;
	// take the nlog
	mle[y_][z_] = MultinomialParams::nLog(mle[y_][z_]);
      }
    }
  }
  
  // make sure all lambda features which may fire on this training data are added to lambda.params
  void WarmUp();

  // call back function for simulated annealing
  static float EvaluateNLogLikelihood(float *lambdasArray);

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

  // adds up the values in v1 and v2 and returns the summation vector
  static FastSparseVector<double> AccumulateDerivatives(const FastSparseVector<double> &v1, const FastSparseVector<double> &v2);

  // builds an FST to computes B(x,z)
  void BuildThetaLambdaFst(const std::vector<int> &x, const std::vector<int> &z, 
			   fst::VectorFst<FstUtils::LogArc> &fst, std::vector<FstUtils::LogWeight>& alphas, std::vector<FstUtils::LogWeight>& betas);

  // build an FST to compute Z(x)
  void BuildLambdaFst(const std::vector<int> &x, fst::VectorFst<FstUtils::LogArc> &fst);

  // build an FST to compute Z(x). also computes potentials
  void BuildLambdaFst(const std::vector<int> &x, fst::VectorFst<FstUtils::LogArc> &fst, std::vector<FstUtils::LogWeight> &alphas, std::vector<FstUtils::LogWeight> &betas);

  // compute the partition function Z_\lambda(x)
  double ComputeNLogZ_lambda(const std::vector<int> &x); // much slower
  double ComputeNLogZ_lambda(const fst::VectorFst<FstUtils::LogArc> &fst, const std::vector<FstUtils::LogWeight> &betas); // much faster

  // compute B(x,z) which can be indexed as: BXZ[y^*][z^*] to give B(x, z, z^*, y^*)
  // assumptions: BXZ is cleared
  void ComputeB(const std::vector<int> &x, const std::vector<int> &z, 
		const fst::VectorFst<FstUtils::LogArc> &fst, 
		const std::vector<FstUtils::LogWeight> &alphas, const std::vector<FstUtils::LogWeight> &betas, 
		std::map< int, std::map< int, LogVal<double> > > &BXZ);

  // compute B(x,z) which can be indexed as: BXZ[y^*][z^*] to give B(x, z, z^*, y^*)
  // assumptions: BXZ is cleared
  void ComputeB(const std::vector<int> &x, const std::vector<int> &z, 
		const fst::VectorFst<FstUtils::LogArc> &fst, 
		const std::vector<FstUtils::LogWeight> &alphas, const std::vector<FstUtils::LogWeight> &betas, 
		std::map< std::pair<int, int>, std::map< int, LogVal<double> > > &BXZ);

  // assumptions: 
  // - fst is populated using BuildLambdaFst()
  // - FXZk is cleared
  void ComputeF(const std::vector<int> &x, 
		const fst::VectorFst<FstUtils::LogArc> &fst,
		const std::vector<FstUtils::LogWeight> &alphas, const std::vector<FstUtils::LogWeight> &betas,
		FastSparseVector<LogVal<double> > &FXZk);

  // assumptions: 
  // - fst is populated using BuildThetaLambdaFst()
  // - DXZk is cleared
  void ComputeD(const std::vector<int> &x, const std::vector<int> &z,
		const fst::VectorFst<FstUtils::LogArc> &fst,
		const std::vector<FstUtils::LogWeight> &alphas, const std::vector<FstUtils::LogWeight> &betas,
		std::map<std::string, double> &DXZk);

  // assumptions: 
  // - fst is populated using BuildThetaLambdaFst()
  // - DXZk is cleared
  void ComputeD(const std::vector<int> &x, const std::vector<int> &z, 
		const fst::VectorFst<FstUtils::LogArc> &fst,
		const std::vector<FstUtils::LogWeight> &alphas, const std::vector<FstUtils::LogWeight> &betas,
		FastSparseVector<LogVal<double> > &DXZk);
    
  // assumptions:
  // - fst, betas are populated using BuildThetaLambdaFst()
  double ComputeNLogC(const fst::VectorFst<FstUtils::LogArc> &fst,
		      const std::vector<FstUtils::LogWeight> &betas);
    
  // compute p(y, z | x) = \frac{\prod_i \theta_{z_i|y_i} \exp \lambda h(y_i, y_{i-1}, x, i)}{Z_\lambda(x)}
  double ComputeNLogPrYZGivenX(std::vector<int>& x, std::vector<int>& y, std::vector<int>& z);

  // copute p(y | x, z) = \frac  {\prod_i \theta_{z_i|y_i} \exp \lambda h(y_i, y_{i-1}, x, i)} 
  //                             -------------------------------------------
  //                             {\sum_y' \prod_i \theta_{z_i|y'_i} \exp \lambda h(y'_i, y'_{i-1}, x, i)}
  double ComputeNLogPrYGivenXZ(std::vector<int> &x, std::vector<int> &y, std::vector<int> &z);
    
  double ComputeCorpusNloglikelihood();

  // configure lbfgs parameters according to the LearningInfo member of the model
  lbfgs_parameter_t SetLbfgsConfig();

  // fire features in this sentence
  void FireFeatures(const std::vector<int> &x,
		    const fst::VectorFst<FstUtils::LogArc> &fst,
		    FastSparseVector<double> &h);

  // add constrained features with hand-crafted weights
  void AddConstrainedFeatures();

  void ReduceMleAndMarginals(MultinomialParams::ConditionalMultinomialParam<int> mleGivenOneLabel, 
			     MultinomialParams::ConditionalMultinomialParam< std::pair<int, int> > mleGivenTwoLabels,
			     std::map<int, double> mleMarginalsGivenOneLabel,
			     std::map<std::pair<int, int>, double> mleMarginalsGivenTwoLabels);
    

 public:

  static LatentCrfModel& GetInstance();

  static LatentCrfModel& GetInstance(const std::string &textFilename, 
				     const std::string &outputPrefix, 
				     LearningInfo &learningInfo, 
				     unsigned NUMBER_OF_LABELS, 
				     unsigned FIRST_LABEL_ID);

  // aggregates sets for the mpi reduce operation
  static std::set<std::string> AggregateSets(const std::set<std::string> &v1, const std::set<std::string> &v2);
  
  // aggregates vectors for the mpi reduce operation
  static std::vector<double> AggregateVectors(const std::vector<double> &v1, const std::vector<double> &v2) {
    assert(v1.size() == v2.size());
    std::vector<double> vTotal(v1.size());
    for(unsigned i = 0; i < v1.size(); i++) {
      vTotal[i] = v1[i] + v2[i];
    }
    return vTotal;
  }  
  
  // train the model
  void Train();

  // given an observation sequence x (i.e. tokens), find the most likely label sequence y (i.e. labels)
  void Label(std::vector<int> &tokens, std::vector<int> &labels);
  void Label(std::vector<std::string> &tokens, std::vector<int> &labels);
  void Label(std::vector<std::vector<int> > &tokens, std::vector<std::vector<int> > &lables);
  void Label(std::vector<std::vector<std::string> > &tokens, std::vector<std::vector<int> > &labels);
  void Label(std::string &inputFilename, std::string &outputFilename);

  // analyze
  void Analyze(std::string &inputFilename, std::string &outputFilename);

  // evaluate
  double ComputeVariationOfInformation(std::string &labelsFilename, std::string &goldLabelsFilename);
  double ComputeManyToOne(std::string &aLabelsFilename, std::string &bLabelsFilename);

  // collect soft counts from this sentence
  void NormalizeThetaMleAndUpdateTheta(MultinomialParams::ConditionalMultinomialParam<int> &mleGivenOneLabel, 
				       std::map<int, double> &mleMarginalsGivenOneLabel,
				       MultinomialParams::ConditionalMultinomialParam< std::pair<int, int> > &mleGivenTwoLabels, 
				       std::map< std::pair<int, int>, double> &mleMarginalsGivenTwoLabels);
  
  // broadcasts the essential member variables in LogLinearParam
  void BroadcastLambdas() {
    lambda->Broadcast(*learningInfo.mpiWorld, 0);
  }
    
  void BroadcastTheta();

  template <typename ContextType>
  void UpdateThetaMleForSent(const unsigned sentId, 
			     MultinomialParams::ConditionalMultinomialParam<ContextType> &mle, 
			     std::map<ContextType, double> &mleMarginals) {
    if(learningInfo.debugLevel >= DebugLevel::SENTENCE) {
      std::cerr << "sentId = " << sentId << endl;
    }
    assert(sentId < data.size());
    // build the FST
    fst::VectorFst<FstUtils::LogArc> thetaLambdaFst;
    std::vector<FstUtils::LogWeight> alphas, betas;
    BuildThetaLambdaFst(data[sentId], data[sentId], thetaLambdaFst, alphas, betas);
    // compute the B matrix for this sentence
    std::map< ContextType, std::map< int, LogVal<double> > > B;
    B.clear();
    ComputeB(this->data[sentId], this->data[sentId], thetaLambdaFst, alphas, betas, B);
    // compute the C value for this sentence
    double nLogC = ComputeNLogC(thetaLambdaFst, betas);
    //cerr << "nloglikelihood += " << nLogC << endl;
    // update mle for each z^*|y^* fired
    for(typename std::map< ContextType, std::map<int, LogVal<double> > >::const_iterator yIter = B.begin(); yIter != B.end(); yIter++) {
      const ContextType &y_ = yIter->first;
      for(std::map<int, LogVal<double> >::const_iterator zIter = yIter->second.begin(); zIter != yIter->second.end(); zIter++) {
	int z_ = zIter->first;
	double nLogb = zIter->second.s_? zIter->second.v_ : -zIter->second.v_;
	double bOverC = MultinomialParams::nExp(nLogb - nLogC);
	mle[y_][z_] += bOverC;
	mleMarginals[y_] += bOverC;
      }
    }
  }
  void UpdateThetaMleForSent(const unsigned sentId, 
			     MultinomialParams::ConditionalMultinomialParam<int> &mleGivenOneLabel, 
			     std::map<int, double> &mleMarginalsGivenOneLabel,
			     MultinomialParams::ConditionalMultinomialParam< std::pair<int, int> > &mleGivenTwoLabels, 
			     std::map< std::pair<int, int>, double> &mleMarginalsGivenTwoLabels);

  void InitTheta();

  double GetNLogTheta(int yim1, int yi, int zi);

  void PersistTheta(std::string thetaParamsFilename);

  void SupervisedTrain(std::string goldLabelsFilename);

  static double EvaluateNLogLikelihoodYGivenXDerivativeWRTLambda(void *uselessPtr,
								 const double *lambdasArray,
								 double *gradient,
								 const int lambdasCount,
								 const double step);

 public:
  std::vector<std::vector<int> > data, labels;
  LearningInfo learningInfo;
  LogLinearParams *lambda;
  MultinomialParams::ConditionalMultinomialParam<int> nLogThetaGivenOneLabel;
  MultinomialParams::ConditionalMultinomialParam< std::pair<int, int> > nLogThetaGivenTwoLabels;

 private:
  VocabEncoder vocabEncoder;
  int START_OF_SENTENCE_Y_VALUE, END_OF_SENTENCE_Y_VALUE, FIRST_ALLOWED_LABEL_VALUE;
  std::string textFilename, outputPrefix;
  std::set<int> xDomain, yDomain;
  // vectors specifiying which feature types to use (initialized in the constructor)
  std::vector<bool> enabledFeatureTypes;
  unsigned countOfConstrainedLambdaParameters;
  double REWARD_FOR_CONSTRAINED_FEATURES, PENALTY_FOR_CONSTRAINED_FEATURES;
  GaussianSampler gaussianSampler;
  SimAnneal simulatedAnnealer;
};

#endif
