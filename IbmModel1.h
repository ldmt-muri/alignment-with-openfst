#ifndef _IBM_MODEL_1_H_
#define _IBM_MODEL_1_H_

#include <iostream>
#include <fstream>
#include <math.h>

#include "LearningInfo.h"
#include "StringUtils.h"
#include "FstUtils.h"

using namespace fst;
using namespace std;

#define NULL_SRC_TOKEN_ID 1

typedef map<int, map< int, float > >  Model1Param;

class IbmModel1 {

 public:

  // creates an fst for each target sentence
  void CreateTgtFsts(const string& tgtCorpusFilename, vector< VectorFst< LogArc > >& targetFsts);

  // normalizes the parameters such that \sum_t p(t|s) = 1 \forall s
  void NormalizeParams(Model1Param& params);
  
  void PrintParams(const Model1Param params);

  void PersistParams(Model1Param& params, string outputFilename);
  
  // finds out what are the parameters needed by reading hte corpus, and assigning initial weights based on the number of co-occurences
  void InitParams(const string& srcCorpusFilename, const string& tgtCorpusFilename, Model1Param& params, const string& initModelFilename);
  
  void CreateGrammarFst(const Model1Param& params, VectorFst< LogArc >& grammarFst);
  
  bool IsModelConverged(const LearningInfo& learningInfo);

  // zero all parameters
  void ClearParams(Model1Param& params);
  
  void LearnParameters(Model1Param& params, VectorFst< LogArc >& grammarFst, vector< VectorFst< LogArc > >& srcFsts,
		       vector< VectorFst< LogArc > >& tgtFsts, LearningInfo& learningInfo, string outputFilenamePrefix);
  
  // returns a list of acceptors of the source sentences in any order. 
  // Each acceptor has a single state with arcs representing src tokens in addition to NULL (srcTokenId = 0)
  void CreateSrcFsts(const string& srcCorpusFilename, vector< VectorFst< LogArc > >& srcFsts); 
};

#endif

