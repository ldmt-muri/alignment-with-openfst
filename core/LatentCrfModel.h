#ifndef _LATENT_CRF_MODEL_H_
#define _LATENT_CRF_MODEL_H_

#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <set>
#include <algorithm>
#include <ctime>
#include <vector>

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
#include <boost/unordered_map.hpp>

#ifndef EIGEN_CONFIG_H_
#define EIGEN_CONFIG_H_

#include <boost/serialization/array.hpp>
/** 
 * FIXME:
 * UGLY!!!!!
 */
#define EIGEN_DENSEBASE_PLUGIN "/usr0/home/chuchenl/git/alignment-with-openfst/core/EigenDenseBaseAddons.h"

#include <Eigen/Core>

#endif // EIGEN_CONFIG_H_

#include <Eigen/Dense>
#include <Eigen/QR>

#define HAVE_BOOST_ARCHIVE_TEXT_OARCHIVE_HPP 1

#include "MultinomialParams.h"

//#define HAVE_CMPH 1
#include "../cdec-utils/logval.h"
#include "../cdec-utils/semiring.h"
#include "../cdec-utils/fast_sparse_vector.h"

//#include "../wammar-utils/ClustersComparer.h"
#include "../wammar-utils/StringUtils.h"
#include "../wammar-utils/FstUtils.h"
#include "../wammar-utils/LbfgsUtils.h"
#include "Functors.h"

#include "LogLinearParams.h"
#include "UnsupervisedSequenceTaggingModel.h"

// Define a matrix of doubles using Eigen.
typedef LogVal<double> LogValD;
namespace Eigen {
  typedef Eigen::Matrix<LogValD, Dynamic, Dynamic> MatrixXlogd;
  typedef Eigen::Matrix<LogValD, Dynamic, 1> VectorXlogd;
  
#ifndef NEURALCONST
#define NEURALCONST 100
#endif
  const int NEURAL_SIZE = NEURALCONST;
  
  typedef Eigen::Matrix<double, NEURAL_SIZE, 1> VectorNeural;
  typedef Eigen::Matrix<double, NEURAL_SIZE, NEURAL_SIZE> MatrixNeural;
  
  const double NONE = 5566;
  const Eigen::VectorNeural NONE_VEC = VectorNeural::Ones(NEURAL_SIZE,1) * NONE;
}

typedef std::mt19937 rng;

namespace boost {
namespace serialization{

template<class Archive, typename T, typename H, typename P, typename A>
void save(Archive &ar,
          const std::tr1::unordered_set<T,H,P,A> &s, const unsigned int) {
    vector<T> vec(s.begin(),s.end());   
    ar<<vec;    
}
template<class Archive, typename T, typename H, typename P, typename A>
void load(Archive &ar,
          std::tr1::unordered_set<T,H,P,A> &s, const unsigned int) {
    vector<T> vec;  
    ar>>vec;   
    std::copy(vec.begin(),vec.end(),    
              std::inserter(s,s.begin()));  
}

template<class Archive, typename T, typename H, typename P, typename A>
void serialize(Archive &ar,
               std::tr1::unordered_set<T,H,P,A> &s, const unsigned int version) {
    boost::serialization::split_free(ar,s,version);
//
//        template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
//        inline void serialize(
//                Archive & ar,
//                Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> & t,
//                const unsigned int file_version
//                ) {
//            for (size_t i = 0; i < t.size(); i++)
//                ar & t.data()[i];
//        }        
    }
        template<class Archive>
        void serialize(Archive &ar,
                LogVal<double> &s, const unsigned int version) {
            ar & s.v_;
        }
}
}

using unordered_set_featureId = std::tr1::unordered_set<FeatureId, FeatureId::FeatureIdHash, FeatureId::FeatureIdEqual>;

struct AggregateSets2 {
  unordered_set_featureId operator()(const unordered_set_featureId &v1, 
                                     const unordered_set_featureId &v2) {
    cerr << "aggregating unordered sets of featureIds |v1| = " << v1.size() << ", |v2| = " << v2.size() << " ...";
    unordered_set_featureId vTotal;
    cerr << ", vTotal.max_load_factor() = " << vTotal.max_load_factor();
    vTotal.rehash( ceil( (v1.size() + v2.size()) / vTotal.max_load_factor()));
    
    for(auto v1Iter = v1.begin(); v1Iter != v1.end(); ++v1Iter) {
      vTotal.insert(*v1Iter);
    }
    for(auto v2Iter = v2.begin(); v2Iter != v2.end(); ++v2Iter) {
      vTotal.insert(*v2Iter);
    }
    cerr << ", |vTotal| = " << vTotal.size() << endl;
    return vTotal;
  }
};


// implements the model described at doc/LatentCrfModel.tex
class LatentCrfModel : public UnsupervisedSequenceTaggingModel {

  // template and inline member functions
#include "LatentCrfModel-inl.h"

 public: 

  enum Task {POS_TAGGING=0, WORD_ALIGNMENT=1, DEPENDENCY_PARSING=2};
  
  // STATIC METHODS
  /////////////////

  static LatentCrfModel& GetInstance();

  // call back function for simulated annealing
  static float EvaluateNll(float *lambdasArray);

  // evaluate the \sum_<x,z>  -log(z|x) , plus L2(\lambda) when the model is configured to use it
  double EvaluateNll();

  // use the method of finite differences to numerically check the gradient computed with dynamic programming
  double CheckGradient(lbfgs_evaluate_t proc_evaluate, vector<int> &testIndexes, double epsilon);

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

  void SupervisedTrain(bool fitLambdas=true, bool fitThetas=true);
  virtual void SupervisedTrainTheta();

  void BlockCoordinateDescent();

  void OptimizeLambdasWithSgd(double& optimizedMiniBatchNll);
  void OptimizeLambdasWithLbfgs(double& optimizedMiniBatchNll, lbfgs_parameter_t& lbfgsParams);
  void OptimizeLambdasWithAdagrad(double& optimizedMiniBatchNll, 
                                  double& miniBatchDevSetNll, 
                                  vector<double>& gradient, 
                                  vector<double>& u, vector<double>& h, 
                                  int& adagradIter);
  void ShuffleElements(vector<int>& elements);
  
  // analyze
  void Analyze(std::string &inputFilename, std::string &outputFilename);

  void PersistTheta(std::string thetaParamsFilename);

  // LABEL new examples
  ///////////////

  // given an observation sequence x (i.e. tokens), find the most likely label sequence y (i.e. labels)
  void Label(std::vector<std::string> &tokens, std::vector<int> &labels);
  void Label(std::vector<std::vector<int64_t> > &tokens, std::vector<std::vector<int> > &lables);
  void Label(std::vector<std::vector<std::string> > &tokens, std::vector<std::vector<int> > &labels);
  virtual void Label(std::string &inputFilename, std::string &outputFilename);
  virtual void Label(std::vector<int64_t> &tokens, std::vector<int> &labels) = 0;

  // CONVENIENCE MPI OPERATIONS
  /////////////////////////////

  void ReduceMleAndMarginals(MultinomialParams::ConditionalMultinomialParam<int64_t> &mleGivenOneLabel, 
			     MultinomialParams::ConditionalMultinomialParam< std::pair<int64_t, int64_t> > &mleGivenTwoLabels,
			     boost::unordered_map<int64_t, double> &mleMarginalsGivenOneLabel,
			     boost::unordered_map<std::pair<int64_t, int64_t>, double> &mleMarginalsGivenTwoLabels);

    void GatherMean(const boost::unordered_map< int64_t,
            std::vector<Eigen::VectorNeural> > &means, boost::unordered_map< int64_t,
            std::vector<LogVal<double>>> &nNormalizingConstant,
            std::vector<boost::unordered_map< int64_t,
            std::vector<Eigen::VectorNeural>>> &allMeans, std::vector<boost::unordered_map< int64_t,
            std::vector<LogVal<double>>>> &allNNormalizingConstant);
  
  void BroadcastTheta(unsigned rankId);
  
  void BroadcastMeans(unsigned rankId);

  // filenames
  string GetLambdaFilename(int iteration, bool humane);
  string GetThetaFilename(int iteration);

  // SETUP
  ////////
  
  // creates a list of vocab IDs of closed vocab words
  void AddEnglishClosedVocab();

  // configure lbfgs parameters according to the LearningInfo member of the model
  lbfgs_parameter_t SetLbfgsConfig(bool);
  void PrintLbfgsConfig(lbfgs_parameter_t &lbfgsParams);

  // (MINI)BATCH LEVEL

  void NormalizeThetaMleAndUpdateTheta(MultinomialParams::ConditionalMultinomialParam<int64_t> &mleGivenOneLabel, 
				       boost::unordered_map<int64_t, double> &mleMarginalsGivenOneLabel,
				       MultinomialParams::ConditionalMultinomialParam< std::pair<int64_t, int64_t> > &mleGivenTwoLabels, 
				       boost::unordered_map< std::pair<int64_t, int64_t>, double> &mleMarginalsGivenTwoLabels);
  

  void NormalizeMleMeanAndUpdateMean( std::vector<boost::unordered_map< int64_t, std::vector<Eigen::VectorNeural>>>& means,
                                      std::vector<boost::unordered_map< int64_t, std::vector<LogVal<double>>>>& nNormalizingConstant);
  
  // make sure all lambda features which may fire on this training data are added to lambda.params
  void InitLambda();

  virtual void InitTheta() = 0;

  // SUBSENT LEVEL
  ////////////////

  // given a sentId and the value of y, find the conditioning context of the relevant multinomial
  // in word alignment, this should return the corresponding src word.
  // in pos tagging, this should return y itself.
  virtual int64_t GetContextOfTheta(unsigned sentId, int y);

  // fire features in this sentence
  virtual void FireFeatures(const unsigned sentId,
                            FastSparseVector<double> &h);

  void FireFeatures(unsigned sentId,
		    const fst::VectorFst<FstUtils::LogArc> &fst,
		    FastSparseVector<double> &h);

  virtual void FireFeatures(int yI, int yIM1, unsigned sentId, int i, 
		    FastSparseVector<double> &activeFeatures) = 0;

  double GetNLogTheta(int yim1, int yi, int64_t zi, unsigned exampleId);
  double GetNLogTheta(const std::pair<int64_t,int64_t> context, int64_t event);
  double GetNLogTheta(int64_t context, int64_t event);

  double getGaussianPDF(int64_t yi, const Eigen::VectorNeural& zi);
  
  
  virtual std::vector<int64_t>& GetObservableSequence(int exampleId) = 0;

  virtual std::vector<int64_t>& GetObservableContext(int exampleId) = 0;

  virtual std::vector<int64_t>& GetReconstructedObservableSequence(int exampleId) = 0;

  virtual const std::vector<Eigen::VectorNeural>& GetNeuralSequence(int exampleId) = 0;

  // SENT LEVEL
  ///////////

  virtual double UpdateThetaMleForSent(const unsigned sentId, 
			       MultinomialParams::ConditionalMultinomialParam<pair<int64_t,int64_t> > &mle, 
			       boost::unordered_map< pair<int64_t, int64_t> , double> &mleMarginals);
    
  virtual double UpdateThetaMleForSent(const unsigned sentId, 
			       MultinomialParams::ConditionalMultinomialParam< int64_t > &mle, 
			       boost::unordered_map< int64_t , double> &mleMarginals);
    
  // adds l2 reguarlization term (for lambdas) to both the objective and the gradient
  double AddL2Term(const std::vector<double> &unregularizedGradient, double *regularizedGradient, double unregularizedObjective, double &gradientL2Norm);

  // adds l2 reguarlization term (for lambdas) to the objective
  double AddL2Term(double unregularizedObjective);

  void AddWeightedL2Term(vector<double> *gradient, double *objective, FastSparseVector<double> &activeFeatures);

  // prepare the model before processing an example
  virtual void PrepareExample(unsigned exampleId) = 0;

  // collect soft counts from this sentence
  double UpdateThetaMleForSent(const unsigned sentId, 
			     MultinomialParams::ConditionalMultinomialParam<int64_t> &mleGivenOneLabel, 
			     boost::unordered_map<int64_t, double> &mleMarginalsGivenOneLabel,
			     MultinomialParams::ConditionalMultinomialParam< std::pair<int64_t, int64_t> > &mleGivenTwoLabels, 
			     boost::unordered_map< std::pair<int64_t, int64_t>, double> &mleMarginalsGivenTwoLabels);

  // builds an FST to computes B(x,z)
  void BuildThetaLambdaFst(unsigned sentId, const std::vector<int64_t> &z, 
			   fst::VectorFst<FstUtils::LogArc> &fst, std::vector<FstUtils::LogWeight>& alphas, 
         std::vector<FstUtils::LogWeight>& betas);

  // builds an FST to computes B(x,z) for neural rep
  void BuildThetaLambdaFst(unsigned sentId, const std::vector<Eigen::VectorNeural> &z, 
			   fst::VectorFst<FstUtils::LogArc> &fst, std::vector<FstUtils::LogWeight>& alphas, 
         std::vector<FstUtils::LogWeight>& betas);
  
  // build an FST to compute Z(x)
  void BuildLambdaFst(unsigned sentId, fst::VectorFst<FstUtils::LogArc> &fst);

  // build an FST to compute Z(x). also computes potentials
  void BuildLambdaFst(unsigned sentId, fst::VectorFst<FstUtils::LogArc> &fst, std::vector<FstUtils::LogWeight> &alphas, std::vector<FstUtils::LogWeight> &betas);

  // iterates over training examples, accumulates p(z|x) according to the current model and also accumulates its derivative w.r.t lambda
  virtual double ComputeNllZGivenXAndLambdaGradient(vector<double> &gradient, int fromSentId, int toSentId, double *devSetNll);
  virtual double ComputeNllYGivenXAndLambdaGradient(vector<double> &gradient, int fromSentId, int toSentId);

  virtual bool ComputeNllZGivenXAndLambdaGradientPerSentence(bool ignoreThetaTerms, 
                                                             int sentId,
                                                             double& sentNll,
                                                             FastSparseVector<double>& sentNllGradient);
  

  // compute the partition function Z_\lambda(x)
  double ComputeNLogZ_lambda(const fst::VectorFst<FstUtils::LogArc> &fst, const std::vector<FstUtils::LogWeight> &betas); // much faster

  // compute B(x,z) which can be indexed as: BXZ[y^*][z^*] to give B(x, z, z^*, y^*)
  // assumptions: BXZ is cleared
  void ComputeB(unsigned sentId, const std::vector<int64_t> &z, 
		const fst::VectorFst<FstUtils::LogArc> &fst, 
		const std::vector<FstUtils::LogWeight> &alphas, const std::vector<FstUtils::LogWeight> &betas, 
		boost::unordered_map< int64_t, boost::unordered_map< int64_t, LogVal<double> > > &BXZ);

  // compute B(x,z) which can be indexed as: BXZ[y^*][z^*] to give B(x, z, z^*, y^*)
  // assumptions: BXZ is cleared
  void ComputeB(unsigned sentId, const std::vector<int64_t> &z, 
		const fst::VectorFst<FstUtils::LogArc> &fst, 
		const std::vector<FstUtils::LogWeight> &alphas, const std::vector<FstUtils::LogWeight> &betas, 
		boost::unordered_map< std::pair<int64_t, int64_t>, boost::unordered_map< int64_t, LogVal<double> > > &BXZ);

    // FIXME
    // computes expected mean per label
    void ComputeExpectedMean(unsigned sentId,
            const vector<Eigen::VectorNeural>&z,
            const fst::VectorFst<FstUtils::LogArc> &fst,
            const vector<FstUtils::LogWeight> &alphas,
            const vector<FstUtils::LogWeight> &betas,
            boost::unordered_map< int64_t, std::vector<Eigen::VectorNeural> > &meanPerLabel,
            boost::unordered_map< int64_t, std::vector<LogVal<double >>> &nNormalizingConstant, double nLogC);

  // assumptions:
  // - fst, betas are populated using BuildThetaLambdaFst()
  double ComputeNLogC(const fst::VectorFst<FstUtils::LogArc> &fst,
		      const std::vector<FstUtils::LogWeight> &betas);

  // assumptions: 
  // - fst is populated using BuildLambdaFst()
  // - FXZk is cleared
  void ComputeFOverZ(unsigned sentId, 
		const fst::VectorFst<FstUtils::LogArc> &fst,
		const std::vector<FstUtils::LogWeight> &alphas, const std::vector<FstUtils::LogWeight> &betas,
		FastSparseVector<double> &FXZk);

  // assumptions: 
  // - fst is populated using BuildThetaLambdaFst()
  // - DXZk is cleared
  void ComputeDOverC(unsigned sentId, const std::vector<int64_t> &z, 
		const fst::VectorFst<FstUtils::LogArc> &fst,
		const std::vector<FstUtils::LogWeight> &alphas, const std::vector<FstUtils::LogWeight> &betas,
		FastSparseVector<double> &DOverCk);

 protected:
  LatentCrfModel(const std::string &textFilename, 
		 const std::string &outputPrefix, 
		 LearningInfo &learningInfo,
		 unsigned firstLabelId,
		 Task modelTask);
  
  ~LatentCrfModel();

  // this should be done by master only
  void EncodeTgtWordClasses();
  
  // this should be done by all processes
  void LoadTgtWordClasses(std::vector<std::vector<int64_t> > &tgtSents);

  // convert tgt tokens to a word class sequence (if provided)
  vector<int64_t> GetTgtWordClassSequence(vector<int64_t> &x_t);
  
    
  // read each line in the text file, encodes each sentence into vector<VectorNeural> and appends it into 'data'
  // assumptions: data is empty
  void readNeuralRep(const std::string &textFilename, std::vector<std::vector<Eigen::VectorNeural>> &data) {
    assert(data.size() == 0);
    
    // open data file
    std::ifstream textFile(textFilename.c_str(), std::ios::in);
    
    // for each line
    std::string line;
    int64_t lineNumber = -1;
    while(getline(textFile, line)) {
      
      // skip empty lines
      if(line.size() == 0) {
        continue;
      }
      lineNumber++;
      
      // split tokens
      std::vector<string> splits;
      StringUtils::SplitString(line, ' ', splits);
      
      // encode tokens
      data.resize(lineNumber+1);
      
      data[lineNumber].resize(splits.size());
      
      for(auto i = data[lineNumber].begin(); i != data[lineNumber].end(); i++) {
          auto& word = *i;
          word.setZero(Eigen::NEURAL_SIZE, 1);
          auto word_idx = i - data[lineNumber].begin();
          if(splits[word_idx] == "NONE") {
              word = Eigen::NONE_VEC;
              continue;
          }
          std::vector<string> dims;
          StringUtils::SplitString(splits[word_idx], ',', dims);
          assert(dims.size() == Eigen::NEURAL_SIZE);
          for(auto j=dims.begin(); j != dims.end(); j++) {
              auto dim_idx = j - dims.begin();
              double v = std::stod(*j);
              word(dim_idx) = v;
          }
      }
    }
  }

 public:
  std::vector<std::vector<int64_t> > labels;
  LearningInfo learningInfo;
  LogLinearParams *lambda;
  MultinomialParams::ConditionalMultinomialParam<int64_t> nLogThetaGivenOneLabel;
  MultinomialParams::ConditionalMultinomialParam< std::pair<int64_t, int64_t> > nLogThetaGivenTwoLabels;
  static int START_OF_SENTENCE_Y_VALUE;
  static unsigned NULL_POSITION;
  int END_OF_SENTENCE_Y_VALUE, FIRST_ALLOWED_LABEL_VALUE;
  unsigned examplesCount;
  std::string textFilename, outputPrefix;
 
 public:
  std::vector< std::vector<int64_t> > classTgtSents, testClassTgtSents;
  boost::unordered_map<int64_t, int64_t> tgtWordToClass;
  static LatentCrfModel *instance;
  std::vector<int> yDomain;
  GaussianSampler gaussianSampler;
  // during training time, and by default, this should be set to false. 
  // When we use the trained model to predict the labels, we set it to true
  bool testingMode;
  Task task;
  // this is only set during training while optimizing loglinear parameters
  bool optimizingLambda;
  
  // gold label sequences used for supervised and semi-supervised training only. 
  // in unsupervised training, this should be empty.
  std::vector<std::vector<int> > goldLabelSequences;

  // tagging dictionary
  // this maps a word to the possible word classes
  std::tr1::unordered_map<int64_t, std::tr1::unordered_set<int> > tagDict;
  // this maps the vocab id of a POS tag (e.g. "NOUN") to the word class id used internally to represent it
  std::tr1::unordered_map<int64_t, int> posTagVocabIdToClassId;
  
  boost::unordered_map<int64_t, Eigen::VectorNeural> neuralMean;
  boost::unordered_map<int64_t, Eigen::MatrixNeural> neuralVar;
  
  std::vector<boost::unordered_map<int64_t, Eigen::VectorNeural>> historyNeuralMean;
  
  void clearVarCache() {
      varInverse.clear();
      varDet.clear();
  }
  
  const Eigen::MatrixNeural& getVarInverse(int64_t key) {
      auto iter = varInverse.find(key);
      if(iter==varInverse.end()) {
          varInverse[key]=neuralVar[key].inverse();
          return varInverse[key];
      } else {
          return iter->second;
      }
  }
    
  const double getVarDet(int64_t key) {
      auto iter = varDet.find(key);
      if(iter==varDet.end()) {
          Eigen::FullPivHouseholderQR<Eigen::MatrixNeural> qr = Eigen::FullPivHouseholderQR<Eigen::MatrixNeural>(neuralVar[key]);
          const auto det = qr.logAbsDeterminant();
          varDet[key]=det;
          return det;
      } else {
          return iter->second;
      }
  }
  
private:
    boost::unordered_map<int64_t, Eigen::MatrixNeural> varInverse;
    boost::unordered_map<int64_t, double> varDet;
  


  // random generator
  rng random_generator; 
};

#endif
