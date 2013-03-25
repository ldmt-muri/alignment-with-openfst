#ifndef _LATENT_CRF_ALIGNER_H_
#define _LATENT_CRF_ALIGNER_H_

#include "LatentCrfModel.h"

class LatentCrfWordAligner : public LatentCrfModel {

 protected:
  LatentCrfAligner(const std::string &textFilename, 
		       const std::string &outputPrefix, 
		       LearningInfo &learningInfo,
		       unsigned numberOfLabels,
		       unsigned firstLabelId);
  
  ~LatentCrfAligner();

  std::vector<int> GetObservableSequence(int exampleId);

  void InitTheta();

 public:

  static LatentCrfAligner& GetInstance();

  static LatentCrfAligner& GetInstance(const std::string &textFilename, 
				       const std::string &outputPrefix, 
				       LearningInfo &learningInfo, 
				       unsigned NUMBER_OF_LABELS, 
				       unsigned FIRST_LABEL_ID);
 private:
  // the value of y_i that indicates an alignment to the NULL source token, and to the first token in the source sentence, respectively.
  unsigned NULL_POS, FIRST_SRC_POS;

  // vocabulary of src language
  std::set<int> x_sDomain;

  // data
  std::vector< std::vector<int> > srcSents, tgtSents;
};

#endif
