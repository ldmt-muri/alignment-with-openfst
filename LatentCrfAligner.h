#ifndef _LATENT_CRF_ALIGNER_H_
#define _LATENT_CRF_ALIGNER_H_

#include "LatentCrfModel.h"

class LatentCrfAligner : public LatentCrfModel {

 protected:
  LatentCrfAligner(const std::string &textFilename, 
		   const std::string &outputPrefix, 
		   LearningInfo &learningInfo,
		   unsigned firstLabelId,
		   const std::string &initialLambdaParamsFilename, 
		   const std::string &initialThetaParamsFilename,
		   const std::string &wordPairFeaturesFilename);

  ~LatentCrfAligner();

  std::vector<int>& GetObservableSequence(int exampleId);

  std::vector<int>& GetObservableContext(int exampleId);

  void InitTheta();

  void PrepareExample(unsigned exampleId);

  int GetContextOfTheta(unsigned sentId, int y);

 public:

  static LatentCrfModel* GetInstance();

  static LatentCrfModel* GetInstance(const std::string &textFilename, 
				     const std::string &outputPrefix, 
				     LearningInfo &learningInfo, 
				     unsigned FIRST_LABEL_ID,
				     const std::string &initialLambdaParamsFilename, 
				     const std::string &initialThetaParamsFilename,
				     const std::string &wordPairFeaturesFilename);

  void Label(std::vector<int> &tokens, std::vector<int> &labels) { assert(false); /* cannot label without context */ }

  void Label(std::vector<int> &tokens, std::vector<int> &context, std::vector<int> &labels);

  void Label(const string &labelsFilename);

  void AddConstrainedFeatures();

  void SetTestExample(std::vector<int> &x_t, std::vector<int> &x_s);

 private:
  // vocabulary of src language
  std::set<int> x_sDomain;

  // data
  std::vector< std::vector<int> > srcSents, tgtSents, testSrcSents, testTgtSents;

  // null token
  static int NULL_TOKEN;

 public:
  // the value of y_i that indicates an alignment to the NULL source token, and to the first token in the source sentence, respectively.
  static unsigned FIRST_SRC_POSITION;
  // the string representing the null token at the source sentence which produce "spurious" tgt words.
  static string NULL_TOKEN_STR;
};

#endif
