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

 public: 

  enum Task {POS_TAGGING, WORD_ALIGNMENT};
  
  // STATIC METHODS
  /////////////////

  static LatentCrfModel& GetInstance();

  // call back function for simulated annealing
  static float EvaluateNll(float *lambdasArray);

  // evaluate the \sum_<x,z>  -log(z|x) , plus L2(\lambda) when the model is configured to use it
  double EvaluateNll();

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

  static double LbfgsCallbackEvalZGivenXLambdaGradient (void *uselessPtr,
							const double *lambdasArray,
							double *gradient,
							const int lambdasCount,
							const double step);

  // HIGHLEVEL TRULY PUBLIC OPERATION
  ////////////////////////////////////

  // train the model
  void Train();

  void SupervisedTrain(std::string goldLabelsFilename);

  void BlockCoordinateDescent();

  // analyze
  void Analyze(std::string &inputFilename, std::string &outputFilename);

  // evaluate
  double ComputeVariationOfInformation(std::string &labelsFilename, std::string &goldLabelsFilename);
  double ComputeManyToOne(std::string &aLabelsFilename, std::string &bLabelsFilename);

  void PersistTheta(std::string thetaParamsFilename);

  // LABEL new examples
  ///////////////

  // given an observation sequence x (i.e. tokens), find the most likely label sequence y (i.e. labels)
  void Label(std::vector<std::string> &tokens, std::vector<int> &labels);
  void Label(std::vector<std::vector<int> > &tokens, std::vector<std::vector<int> > &lables);
  void Label(std::vector<std::vector<std::string> > &tokens, std::vector<std::vector<int> > &labels);
  void Label(std::string &inputFilename, std::string &outputFilename);
  virtual void Label(std::vector<int> &tokens, std::vector<int> &labels) = 0;

  // CONVENIENCE MPI OPERATIONS
  /////////////////////////////

  void ReduceMleAndMarginals(MultinomialParams::ConditionalMultinomialParam<int> mleGivenOneLabel, 
			     MultinomialParams::ConditionalMultinomialParam< std::pair<int, int> > mleGivenTwoLabels,
			     std::map<int, double> mleMarginalsGivenOneLabel,
			     std::map<std::pair<int, int>, double> mleMarginalsGivenTwoLabels);
  
  // broadcasts the essential member variables in LogLinearParam
  void BroadcastLambdas(unsigned rankId);
    
  void BroadcastTheta(unsigned rankId);

  // SETUP
  ////////
  
  // creates a list of vocab IDs of closed vocab words
  void AddEnglishClosedVocab();

  // configure lbfgs parameters according to the LearningInfo member of the model
  lbfgs_parameter_t SetLbfgsConfig();

  // (MINI)BATCH LEVEL

  void NormalizeThetaMleAndUpdateTheta(MultinomialParams::ConditionalMultinomialParam<int> &mleGivenOneLabel, 
				       std::map<int, double> &mleMarginalsGivenOneLabel,
				       MultinomialParams::ConditionalMultinomialParam< std::pair<int, int> > &mleGivenTwoLabels, 
				       std::map< std::pair<int, int>, double> &mleMarginalsGivenTwoLabels);
  
  // make sure all lambda features which may fire on this training data are added to lambda.params
  void InitLambda();

  virtual void InitTheta() = 0;

  // add constrained features with hand-crafted weights
  virtual void AddConstrainedFeatures() = 0;

  // SUBSENT LEVEL
  ////////////////

  // fire features in this sentence
  void FireFeatures(unsigned sentId,
		    const fst::VectorFst<FstUtils::LogArc> &fst,
		    FastSparseVector<double> &h);

  void FireFeatures(int yI, int yIM1, unsigned sentId, int i, 
		    const std::vector<bool> &enabledFeatureTypes, 
		    FastSparseVector<double> &activeFeatures);

  double GetNLogTheta(int yim1, int yi, int zi, unsigned exampleId);
  double GetNLogTheta(const std::pair<int,int> context, int event);
  double GetNLogTheta(int context, int event);

  virtual std::vector<int>& GetObservableSequence(int exampleId) = 0;

  virtual std::vector<int>& GetObservableContext(int exampleId) = 0;

  // SENT LEVEL
  ///////////

  // adds l2 reguarlization term (for lambdas) to both the objective and the gradient
  double AddL2Term(const std::vector<double> &unregularizedGradient, double *regularizedGradient, double unregularizedObjective);

  // adds l2 reguarlization term (for lambdas) to the objective
  double AddL2Term(double unregularizedObjective);

  // prepare the model before processing an example
  virtual void PrepareExample(unsigned exampleId) = 0;

  // collect soft counts from this sentence
  double UpdateThetaMleForSent(const unsigned sentId, 
			     MultinomialParams::ConditionalMultinomialParam<int> &mleGivenOneLabel, 
			     std::map<int, double> &mleMarginalsGivenOneLabel,
			     MultinomialParams::ConditionalMultinomialParam< std::pair<int, int> > &mleGivenTwoLabels, 
			     std::map< std::pair<int, int>, double> &mleMarginalsGivenTwoLabels);

  // builds an FST to computes B(x,z)
  void BuildThetaLambdaFst(unsigned sentId, const std::vector<int> &z, 
			   fst::VectorFst<FstUtils::LogArc> &fst, std::vector<FstUtils::LogWeight>& alphas, std::vector<FstUtils::LogWeight>& betas);

  // build an FST to compute Z(x)
  void BuildLambdaFst(unsigned sentId, fst::VectorFst<FstUtils::LogArc> &fst);

  // build an FST to compute Z(x). also computes potentials
  void BuildLambdaFst(unsigned sentId, fst::VectorFst<FstUtils::LogArc> &fst, std::vector<FstUtils::LogWeight> &alphas, std::vector<FstUtils::LogWeight> &betas);

  // iterates over training examples, accumulates p(z|x) according to the current model and also accumulates its derivative w.r.t lambda
  double ComputeNllZGivenXAndLambdaGradient(vector<double> &gradient);

  // compute the partition function Z_\lambda(x)
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
		FastSparseVector<LogVal<double> > &DXZk);

 protected:
  LatentCrfModel(const std::string &textFilename, 
		 const std::string &outputPrefix, 
		 LearningInfo &learningInfo,
		 unsigned firstLabelId,
		 Task modelTask);
  
  ~LatentCrfModel();

 public:
  std::vector<std::vector<int> > labels;
  LearningInfo learningInfo;
  LogLinearParams *lambda;
  MultinomialParams::ConditionalMultinomialParam<int> nLogThetaGivenOneLabel;
  MultinomialParams::ConditionalMultinomialParam< std::pair<int, int> > nLogThetaGivenTwoLabels;
  static int START_OF_SENTENCE_Y_VALUE;
  static unsigned NULL_POSITION;
  int END_OF_SENTENCE_Y_VALUE, FIRST_ALLOWED_LABEL_VALUE;
  unsigned examplesCount;
  std::string textFilename, outputPrefix;
  
 protected:
  static LatentCrfModel *instance;
  std::set<int> zDomain, yDomain;
  // vectors specifiying which feature types to use (initialized in the constructor)
  std::vector<bool> enabledFeatureTypes;
  unsigned countOfConstrainedLambdaParameters;
  double REWARD_FOR_CONSTRAINED_FEATURES, PENALTY_FOR_CONSTRAINED_FEATURES;
  GaussianSampler gaussianSampler;
  SimAnneal simulatedAnnealer;
  // during training time, and by default, this should be set to false. 
  // When we use the trained model to predict the labels, we set it to true
  bool testingMode;
  Task task;
};



#endif
