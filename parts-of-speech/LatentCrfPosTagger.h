#ifndef _LATENT_CRF_POS_TAGGER_H_
#define _LATENT_CRF_POS_TAGGER_H_

#include "../core/LatentCrfModel.h"

class LatentCrfPosTagger : public LatentCrfModel {

 protected:
  LatentCrfPosTagger(const std::string &textFilename, 
                     const std::string &outputPrefix, 
                     LearningInfo &learningInfo,
                     unsigned numberOfLabels,
                     unsigned firstLabelId,
                     const string &wordPairFeaturesFilename,
                     const string &initLambdaFilename,
                     const string &initThetaFilename);
  
  ~LatentCrfPosTagger();

  void InitTheta();

  void SetTestExample(std::vector<int64_t> &tokens);

  void PrepareExample(unsigned exampleId) { /* do nothing */ }

  std::vector<int64_t>& GetObservableContext(int exampleId) { /* do nothing */ return empty; }

  std::vector<int64_t>& GetReconstructedObservableSequence(int exampleId) override;

  int64_t GetContextOfTheta(unsigned sentId, int y) { return y; }

 public:

  static LatentCrfModel* GetInstance();

  static LatentCrfModel* GetInstance(const std::string &textFilename, 
                                     const std::string &outputPrefix, 
                                     LearningInfo &learningInfo, 
                                     unsigned NUMBER_OF_LABELS, 
                                     unsigned FIRST_LABEL_ID,
                                     const string &wordPairFeaturesFilename,
                                     const string &initLambdaFilename,
                                     const string &initThetaFilename);

  std::vector<int64_t>& GetObservableSequence(int exampleId);
  
  const std::vector<Eigen::VectorNeural>& GetNeuralSequence(int exampleId) override;
  
  const std::vector<Eigen::VectorNeural>& GetRandomNeural(double mean);

  // add constrained features with hand-crafted weights
  void AddConstrainedFeatures();

  using UnsupervisedSequenceTaggingModel::Label;
  void Label(std::vector<int64_t> &tokens, std::vector<int> &labels);

  void Label(const string &labelsFilename);
  
  void Label(string &inputFilename, string &outputFilename, bool parallelize) override;

  double ComputeNllYGivenXAndLambdaGradient(vector<double> &derivativeWRTLambda, int fromSentId, int toSentId) override;
  
  void FireFeatures(int yI, int yIM1, unsigned sentId, int i, 
				  FastSparseVector<double> &activeFeatures) override;

  
 public:
  std::vector<std::vector<int64_t> > data, testData;
  
  std::vector<std::vector<Eigen::VectorNeural>> neuralRep;

  std::vector<int64_t> empty;

  std::map<int, string> labelIntToString;
  std::map<string, int> labelStringToInt;

};

#endif
