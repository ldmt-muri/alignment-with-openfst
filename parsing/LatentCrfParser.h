#ifndef _LATENT_CRF_PARSER_H_
#define _LATENT_CRF_PARSER_H_

#include <fstream>
#include "../core/LatentCrfModel.h"

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

  void Label(std::vector<int64_t> &tokens, std::vector<int> &labels);

  void Label(std::vector<int64_t> &tokens, std::vector<int64_t> &context, std::vector<int> &labels);

  void Label(const string &labelsFilename);

  void SetTestExample(std::vector<int64_t> &sent);

 private:
  // vocabulary of src language
  //std::set<int64_t> x_sDomain;


  // data
  std::vector< std::vector<int64_t> > sents, contexts, testSents, testContexts;

 public:
  // the value of y_i that indicates the sentence HEAD 
  static int64_t HEAD_ID;
  // the string representing the null token at the source sentence which produce "spurious" tgt words.
  static string HEAD_STR;
  static int HEAD_POSITION;
};

#endif
