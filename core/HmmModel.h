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
#include "../wammar-utils/StringUtils.h"
#include "../wammar-utils/FstUtils.h"
#include "../alignment/IAlignmentModel.h"
#include "../alignment/IAlignmentSampler.h"
#include "../wammar-utils/Samplers.h"
#include "MultinomialParams.h"
#include "UnsupervisedSequenceTaggingModel.h"
#include "LatentCrfModel.h"

class HmmModel2 : public UnsupervisedSequenceTaggingModel {

  // normalizes the parameters such that \sum_t p(t|s) = 1 \forall s
  void NormalizeFractionalCounts();
  
  // zero all parameters
  void ClearFractionalCounts();

  // builds the lattice of all possible label sequences
  void BuildThetaGammaFst(vector<int64_t> &x, fst::VectorFst<FstUtils::LogArc> &fst, unsigned sentId);
  
  // builds the lattice of all possible label sequences, also computes potentials
  void BuildThetaGammaFst(unsigned sentId, fst::VectorFst<FstUtils::LogArc> &fst, vector<FstUtils::LogWeight> &alphas, vector<FstUtils::LogWeight> &betas);
  
  // traverse each transition on the fst and accumulate the mle counts of theta and gamma
  void UpdateMle(const unsigned sentId,
		 const fst::VectorFst<FstUtils::LogArc> &fst, 
		 const vector<FstUtils::LogWeight> &alphas, 
		 const vector<FstUtils::LogWeight> &betas, 
		 MultinomialParams::ConditionalMultinomialParam<int64_t> &thetaMle, 
		 MultinomialParams::ConditionalMultinomialParam<int64_t> &gammaMle);
 
  void InitParams();
  
  double getGaussianPDF(int64_t yi, const Eigen::VectorNeural& zi);
  const vector<Eigen::VectorNeural>& GetNeuralSequence(int exampleId);
    void UpdateMle(const unsigned sentId,
		 const fst::VectorFst<FstUtils::LogArc> &fst, 
		 const vector<FstUtils::LogWeight> &alphas, 
		 const vector<FstUtils::LogWeight> &betas, 
        boost::unordered_map< int64_t, std::vector<Eigen::VectorNeural> > &meanPerLabel,
        boost::unordered_map< int64_t, std::vector<LogVal<double >>> &nNormalizingConstant, 
		 MultinomialParams::ConditionalMultinomialParam<int64_t> &gammaMle);
  void NormalizeMleMeanAndUpdateMean(boost::unordered_map< int64_t, std::vector<Eigen::VectorNeural> >& means,
        boost::unordered_map< int64_t, std::vector<LogVal<double>>>& nNormalizingConstant);
      void Label(vector<vector<string> > &tokens, vector<vector<int> > &labels);
      void Label(vector<int64_t> &tokens, vector<int> &labels);
 public:
  
  HmmModel2(const string &textFilename, 
	    const string &outputPrefix, 
	    LearningInfo &learningInfo,
	    unsigned numberOfLabels,
	    unsigned firstLabelId);

  void PersistParams(string &filename);
  
  void Train();
  
  // using UnsupervisedSequenceTaggingModel::Label;
void Label(vector<int64_t> &tokens, vector<int> &labels, unsigned sentId);
    void Label(string &inputFilename, string &outputFilename) override;
  
  // configurations
  LearningInfo *learningInfo;
  
 private:
  
  // constants
  const int START_OF_SENTENCE_Y_VALUE, FIRST_ALLOWED_LABEL_VALUE;
  
  // training data
  vector< vector<int64_t> > observations;


  
  // gaussian sampler
  GaussianSampler gaussianSampler;

  // output prefix
  string outputPrefix;
  
  // possible values x_i and y_i may take
  set<int64_t> xDomain;
  set<int> yDomain;

 public:
  // model parameters theta = emission probabilities, alpha = transition prbailibities
  MultinomialParams::ConditionalMultinomialParam<int64_t> nlogTheta, nlogGamma;
  
  // neural representation
  std::vector<std::vector<Eigen::VectorNeural>> neuralRep;
  boost::unordered_map<int64_t, Eigen::VectorNeural> neuralMean;
  boost::unordered_map<int64_t, Eigen::MatrixNeural> neuralVar;
    
};

#endif
