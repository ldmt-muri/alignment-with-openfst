#ifndef _LATENT_CRF_POS_TAGGER_H_
#define _LATENT_CRF_POS_TAGGER_H_

#include "LatentCrfModel.h"

class LatentCrfPosTagger : public LatentCrfModel {

 protected:
  LatentCrfPosTagger(const std::string &textFilename, 
		     const std::string &outputPrefix, 
		     LearningInfo &learningInfo,
		     unsigned numberOfLabels,
		     unsigned firstLabelId);
  
  ~LatentCrfPosTagger();

  void InitTheta();

  void SetTestExample(std::vector<int64_t> &tokens);

  void PrepareExample(unsigned exampleId) { /* do nothing */ }

  std::vector<int64_t>& GetObservableContext(int exampleId) { /* do nothing */ return empty; }

  std::vector<int64_t>& GetReconstructedObservableSequence(int exampleId);

  int64_t GetContextOfTheta(unsigned sentId, int y) { return y; }

 public:

  static LatentCrfModel* GetInstance();

  static LatentCrfModel* GetInstance(const std::string &textFilename, 
				     const std::string &outputPrefix, 
				     LearningInfo &learningInfo, 
				     unsigned NUMBER_OF_LABELS, 
				     unsigned FIRST_LABEL_ID);

  std::vector<int64_t>& GetObservableSequence(int exampleId);

  // add constrained features with hand-crafted weights
  void AddConstrainedFeatures();

  using UnsupervisedSequenceTaggingModel::Label;
  void Label(std::vector<int64_t> &tokens, std::vector<int> &labels);

 public:
  std::vector<std::vector<int64_t> > data, testData;

  std::vector<int64_t> empty;

};

#endif
