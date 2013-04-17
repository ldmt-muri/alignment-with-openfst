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

  void SetTestExample(std::vector<int> &tokens);

  void PrepareExample(unsigned exampleId) { /* do nothing */ }

  std::vector<int>& GetObservableContext(int exampleId) { /* do nothing */ return empty; }

  int GetContextOfTheta(unsigned sentId, int y) { return y; }

 public:

  static LatentCrfModel* GetInstance();

  static LatentCrfModel* GetInstance(const std::string &textFilename, 
				     const std::string &outputPrefix, 
				     LearningInfo &learningInfo, 
				     unsigned NUMBER_OF_LABELS, 
				     unsigned FIRST_LABEL_ID);

  std::vector<int>& GetObservableSequence(int exampleId);

  // add constrained features with hand-crafted weights
  void AddConstrainedFeatures();

  using UnsupervisedSequenceTaggingModel::Label;
  void Label(std::vector<int> &tokens, std::vector<int> &labels);

 public:
  std::vector<std::vector<int> > data, testData;

  std::vector<int> empty;
};

#endif
