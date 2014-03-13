#ifndef _LATENT_CRF_PARSER_H_
#define _LATENT_CRF_PARSER_H_

#include <fstream>
#include "LatentCrfParser.h"

class LatentCrfParser : public LatentCrfModel {

 protected:
  LatentCrfParser(const std::string &textFilename, 
		  const std::string &outputPrefix, 
		  LearningInfo &learningInfo,
		  const std::string &initialLambdaParamsFilename, 
		  const std::string &initialThetaParamsFilename,
		  const std::string &wordPairFeaturesFilename);
  
  ~LatentCrfParser();

  std::vector<int64_t>& GetObservableSequence(int exampleId);

  std::vector<int64_t>& GetObservableContext(int exampleId);

  std::vector<int64_t>& GetReconstructedObservableSequence(int exampleId);

  void InitTheta();

  void PrepareExample(unsigned exampleId);

  int64_t GetContextOfTheta(unsigned sentId, int y);

 public:

  static LatentCrfModel* GetInstance();

  static LatentCrfModel* GetInstance(const std::string &textFilename, 
				     const std::string &outputPrefix, 
				     LearningInfo &learningInfo, 
				     const std::string &initialLambdaParamsFilename, 
				     const std::string &initialThetaParamsFilename,
				     const std::string &wordPairFeaturesFilename);

 TODO: CONTINUE ADAPTING CRF ALIGNER TO CRF PARSER HERE
  void Label(std::vector<int64_t> &tokens, std::vector<int> &labels) { assert(false); /* cannot label without context */ }

  void Label(std::vector<int64_t> &tokens, std::vector<int64_t> &context, std::vector<int> &labels);

  void Label(const string &labelsFilename);

  void SetTestExample(std::vector<int64_t> &x_t, std::vector<int64_t> &x_s);

 private:
  // vocabulary of src language
  //std::set<int64_t> x_sDomain;


  // data
  std::vector< std::vector<int64_t> > srcSents, tgtSents, testSrcSents, testTgtSents;

  // null token
  static int64_t NULL_TOKEN;

 public:
  // the value of y_i that indicates an alignment to the NULL source token, and to the first token in the source sentence, respectively.
  static unsigned FIRST_SRC_POSITION;
  // the string representing the null token at the source sentence which produce "spurious" tgt words.
  static string NULL_TOKEN_STR;

};

#endif
