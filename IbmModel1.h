#ifndef _IBM_MODEL_1_H_
#define _IBM_MODEL_1_H_

#include <iostream>
#include <fstream>
#include <math.h>

#include "LearningInfo.h"
#include "StringUtils.h"
#include "FstUtils.h"
#include "IAlignmentModel.h"
#include "MultinomialParams.h"

using namespace fst;
using namespace std;

class IbmModel1 : public IAlignmentModel {

  // normalizes the parameters such that \sum_t p(t|s) = 1 \forall s
  void NormalizeParams();
  
  // creates an fst for each target sentence
  void CreateTgtFsts(vector< VectorFst< FstUtils::LogArc > >& targetFsts);

  void CreateGrammarFst();

  void CreatePerSentGrammarFsts(vector< VectorFst< FstUtils::LogArc > >& perSentGrammarFsts);
  
  // zero all parameters
  void ClearParams();
  
  void LearnParameters(vector< VectorFst< FstUtils::LogArc > >& tgtFsts);
  
 public:

  // bitextFilename is formatted as:
  // source sentence ||| target sentence
  IbmModel1(const string& bitextFilename, 
	    const string& outputFilenamePrefix, 
	    const LearningInfo& learningInfo,
	    const string &NULL_SRC_TOKEN,
	    const VocabEncoder &vocabEncoder);
  
  IbmModel1(const string& bitextFilename, 
	    const string& outputFilenamePrefix, 
	    const LearningInfo& learningInfe);
  
  void CoreConstructor(const string& bitextFilename, 
		       const string& outputFilenamePrefix, 
		       const LearningInfo& learningInfo,
		       const string &NULL_SRC_TOKEN);

  virtual void PrintParams();

  virtual void PersistParams(const string& outputFilename);
  
  // finds out what are the parameters needed by reading hte corpus, and assigning initial weights based on the number of co-occurences
  virtual void InitParams();

  virtual void Train();

  virtual void Align();
  void Align(const string &alignmentsFilename);

  virtual void AlignTestSet(const string &srcTestSetFilename, const string &tgtTestSetFilename, const string &outputAlignmentsFilename);

 private:
  string bitextFilename, outputPrefix;
  VectorFst<FstUtils::LogArc> grammarFst;
  LearningInfo learningInfo;
  vector< vector<int> > srcSents, tgtSents;

 public:  
  // nlog prob(tgt word|src word)
  MultinomialParams::ConditionalMultinomialParam<int> params;
  int NULL_SRC_TOKEN_ID;
  VocabEncoder vocabEncoder;

};

#endif
