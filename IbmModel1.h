#ifndef _IBM_MODEL_1_H_
#define _IBM_MODEL_1_H_

#include <iostream>
#include <fstream>
#include <math.h>

#include "LearningInfo.h"
#include "StringUtils.h"
#include "FstUtils.h"
#include "IAlignmentModel.h"

using namespace fst;
using namespace std;

#define NULL_SRC_TOKEN_ID 1

typedef map<int, map< int, float > >  Model1Param;

class IbmModel1 : public IAlignmentModel.h {

  // normalizes the parameters such that \sum_t p(t|s) = 1 \forall s
  void NormalizeParams();
  
  // creates an fst for each target sentence
  void CreateTgtFsts(vector< VectorFst< LogArc > >& targetFsts);

  void CreateGrammarFst();

  void CreatePerSentGrammarFsts(vector< VectorFst< LogArc > >& perSentGrammarFsts);
  
  // zero all parameters
  void ClearParams();
  
  void LearnParameters(vector< VectorFst< LogArc > >& tgtFsts);
  
 public:

  IbmModel1(const string& srcIntCorpusFilename, const string& tgtIntCorpusFilename, const string& outputFilenamePrefix, const LearningInfo& learningInfo);

  virtual void PrintParams();

  virtual void PersistParams(const string& outputFilename);
  
  // finds out what are the parameters needed by reading hte corpus, and assigning initial weights based on the number of co-occurences
  virtual void InitParams();

  virtual void Train();

  virtual void Align();

  virtual void AlignTestSet(const string &srcTestSetFilename, const string &tgtTestSetFilename, const string &outputAlignmentsFilename);

 private:
  string srcCorpusFilename, tgtCorpusFilename, outputPrefix;
  VectorFst<LogArc> grammarFst;
  LearningInfo learningInfo;

 public:  
  Model1Param params;

};

#endif

