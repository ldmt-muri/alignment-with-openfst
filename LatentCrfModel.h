#ifndef _LATENT_CRF_MODEL_H_
#define _LATENT_CRF_MODEL_H_

#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <set>
#include <algorithm>

#include "mpi.h"

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
#include <boost/function.hpp>
#include <boost/bind/protect.hpp>

#define HAVE_BOOST_ARCHIVE_TEXT_OARCHIVE_HPP 1
//#define HAVE_CMPH 1
#include "cdec-utils/logval.h"
#include "cdec-utils/semiring.h"
#include "cdec-utils/fast_sparse_vector.h"

#include "anneal/Cpp/simann.hpp"

#include "ClustersComparer.h"
#include "StringUtils.h"
#include "FstUtils.h"
#include "LbfgsUtils.h"
#include "Functors.h"

#include "LogLinearParams.h"
#include "MultinomialParams.h"
#include "UnsupervisedSequenceTaggingModel.h"

namespace mpi = boost::mpi;

// implements the model described at doc/LatentCrfModel.tex
class LatentCrfModel : public UnsupervisedSequenceTaggingModel {

  // template and inline member functions
#include "LatentCrfModel-inl.h"

  // optimize the likelihood with block coordinate descent
  void BlockCoordinateDescent();

  // call back function for simulated annealing
  static float EvaluateNLogLikelihood(float *lambdasArray);

  // adds up the values in v1 and v2 and returns the summation vector
  static FastSparseVector<double> AccumulateDerivatives(const FastSparseVector<double> &v1, const FastSparseVector<double> &v2);

  // compute the partition function Z_\lambda(x)
  double ComputeNLogZ_lambda(unsigned sentId); // much slower
  double ComputeNLogZ_lambda(const fst::VectorFst<FstUtils::LogArc> &fst, const std::vector<FstUtils::LogWeight> &betas); // much faster

  // compute B(x,z) which can be indexed as: BXZ[y^*][z^*] to give B(x, z, z^*, y^*)
  // assumptions: BXZ is cleared
  void ComputeB(unsigned sentId, const std::vector<int> &z, 
		const fst::VectorFst<FstUtils::LogArc> &fst, 
		const std::vector<FstUtils::LogWeight> &alphas, const std::vector<FstUtils::LogWeight> &betas, 
		std::map< int, std::map< int, LogVal<double> > > &BXZ);

  // compute B(x,z) which can be indexed as: BXZ[y^*][z^*] to give B(x, z, z^*, y^*)
  // assumptions: BXZ is cleared
  void ComputeB(unsigned sentId, const std::vector<int> &z, 
		const fst::VectorFst<FstUtils::LogArc> &fst, 
		const std::vector<FstUtils::LogWeight> &alphas, const std::vector<FstUtils::LogWeight> &betas, 
		std::map< std::pair<int, int>, std::map< int, LogVal<double> > > &BXZ);

  // assumptions:
  // - fst, betas are populated using BuildThetaLambdaFst()
  double ComputeNLogC(const fst::VectorFst<FstUtils::LogArc> &fst,
		      const std::vector<FstUtils::LogWeight> &betas);
    
  // compute p(y, z | x) = \frac{\prod_i \theta_{z_i|y_i} \exp \lambda h(y_i, y_{i-1}, x, i)}{Z_\lambda(x)}
  double ComputeNLogPrYZGivenX(unsigned sentId, const std::vector<int>& y, const std::vector<int>& z);

  // copute p(y | x, z) = \frac  {\prod_i \theta_{z_i|y_i} \exp \lambda h(y_i, y_{i-1}, x, i)} 
  //                             -------------------------------------------
  //                             {\sum_y' \prod_i \theta_{z_i|y'_i} \exp \lambda h(y'_i, y'_{i-1}, x, i)}
  double ComputeNLogPrYGivenXZ(unsigned sentId, const std::vector<int> &y, const std::vector<int> &z);
    
  double ComputeCorpusNloglikelihood();

  // configure lbfgs parameters according to the LearningInfo member of the model
  lbfgs_parameter_t SetLbfgsConfig();

  // add constrained features with hand-crafted weights
  void AddConstrainedFeatures();

  void ReduceMleAndMarginals(MultinomialParams::ConditionalMultinomialParam<int> mleGivenOneLabel, 
			     MultinomialParams::ConditionalMultinomialParam< std::pair<int, int> > mleGivenTwoLabels,
			     std::map<int, double> mleMarginalsGivenOneLabel,
			     std::map<std::pair<int, int>, double> mleMarginalsGivenTwoLabels);
    
  double ComputeNLoglikelihoodZGivenXAndGradient(vector<double> &gradient);

  // lbfgs call back function to compute the negative loglikelihood and its derivatives with respect to lambdas
  static double LbfgsCallbackEvalYGivenXLambdaGradient(void *ptrFromSentId,
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
  void BuildThetaLambdaFst(unsigned sentId, const std::vector<int> &z, 
			   fst::VectorFst<FstUtils::LogArc> &fst, std::vector<FstUtils::LogWeight>& alphas, std::vector<FstUtils::LogWeight>& betas);

  // build an FST to compute Z(x)
  void BuildLambdaFst(unsigned sentId, fst::VectorFst<FstUtils::LogArc> &fst);

  // build an FST to compute Z(x). also computes potentials
  void BuildLambdaFst(unsigned sentId, fst::VectorFst<FstUtils::LogArc> &fst, std::vector<FstUtils::LogWeight> &alphas, std::vector<FstUtils::LogWeight> &betas);

  // assumptions: 
  // - fst is populated using BuildLambdaFst()
  // - FXZk is cleared
  void ComputeF(unsigned sentId, 
		const fst::VectorFst<FstUtils::LogArc> &fst,
		const std::vector<FstUtils::LogWeight> &alphas, const std::vector<FstUtils::LogWeight> &betas,
		FastSparseVector<LogVal<double> > &FXZk);

  // assumptions: 
  // - fst is populated using BuildThetaLambdaFst()
  // - DXZk is cleared
  void ComputeD(unsigned sentId, const std::vector<int> &z,
		const fst::VectorFst<FstUtils::LogArc> &fst,
		const std::vector<FstUtils::LogWeight> &alphas, const std::vector<FstUtils::LogWeight> &betas,
		std::map<std::string, double> &DXZk);

  // assumptions: 
  // - fst is populated using BuildThetaLambdaFst()
  // - DXZk is cleared
  void ComputeD(unsigned sentId, const std::vector<int> &z, 
		const fst::VectorFst<FstUtils::LogArc> &fst,
		const std::vector<FstUtils::LogWeight> &alphas, const std::vector<FstUtils::LogWeight> &betas,
		FastSparseVector<LogVal<double> > &DXZk);
    
 private:
  LatentCrfModel(const std::string &textFilename, 
		 const std::string &outputPrefix, 
		 LearningInfo &learningInfo,
		 unsigned numberOfLabels,
		 unsigned firstLabelId);
  
  ~LatentCrfModel();

  static LatentCrfModel *instance;

  void AddEnglishClosedVocab();

  // make sure all lambda features which may fire on this training data are added to lambda.params
  void InitLambda();

  // fire features in this sentence
  void FireFeatures(unsigned sentId,
		    const fst::VectorFst<FstUtils::LogArc> &fst,
		    FastSparseVector<double> &h);

 public:

  // given an observation sequence x (i.e. tokens), find the most likely label sequence y (i.e. labels)
  void Label(std::vector<int> &tokens, std::vector<int> &labels);
  void Label(std::vector<std::string> &tokens, std::vector<int> &labels);
  void Label(std::vector<std::vector<int> > &tokens, std::vector<std::vector<int> > &lables);
  void Label(std::vector<std::vector<std::string> > &tokens, std::vector<std::vector<int> > &labels);
  void Label(std::string &inputFilename, std::string &outputFilename);

  static LatentCrfModel& GetInstance(const std::string &textFilename, 
				     const std::string &outputPrefix, 
				     LearningInfo &learningInfo, 
				     unsigned NUMBER_OF_LABELS, 
				     unsigned FIRST_LABEL_ID);

 public:

  static LatentCrfModel& GetInstance();

  static FastSparseVector<double> AggregateSparseVectors(const FastSparseVector<double> &v1, 
							 const FastSparseVector<double> &v2);

  static set<string> AggregateSets(const set<string> &v1, const set<string> &v2);

  // train the model
  void Train();

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
  void BroadcastLambdas(unsigned rankId);
    
  void BroadcastTheta(unsigned rankId);

  void UpdateThetaMleForSent(const unsigned sentId, 
			     MultinomialParams::ConditionalMultinomialParam<int> &mleGivenOneLabel, 
			     std::map<int, double> &mleMarginalsGivenOneLabel,
			     MultinomialParams::ConditionalMultinomialParam< std::pair<int, int> > &mleGivenTwoLabels, 
			     std::map< std::pair<int, int>, double> &mleMarginalsGivenTwoLabels);

  void InitTheta();

  double GetNLogTheta(int yim1, int yi, int zi);

  void PersistTheta(std::string thetaParamsFilename);

  void SupervisedTrain(std::string goldLabelsFilename);

  static double LbfgsCallbackEvalZGivenXLambdaGradient (void *uselessPtr,
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

/*
class LatentCrfPosTagger : public LatentCrfModel {

 private:
  LatentCrfPosTagger(const std::string &textFilename, 
		     const std::string &outputPrefix, 
		     LearningInfo &learningInfo,
		     unsigned numberOfLabels,
		     unsigned firstLabelId);
  
  ~LatentCrfPosTagger();

  static LatentCrfPosTagger *instance;

  void AddEnglishClosedVocab();

  // make sure all lambda features which may fire on this training data are added to lambda.params
  void WarmUp();

  // fire features in this sentence
  void FireFeatures(const std::vector<int> &x,
		    const fst::VectorFst<FstUtils::LogArc> &fst,
		    FastSparseVector<double> &h);

 public:

  static LatentCrfModel& GetInstance(const std::string &textFilename, 
				     const std::string &outputPrefix, 
				     LearningInfo &learningInfo, 
				     unsigned NUMBER_OF_LABELS, 
				     unsigned FIRST_LABEL_ID);

};
*/

#endif
