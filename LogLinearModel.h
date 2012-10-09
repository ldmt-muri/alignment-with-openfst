#ifndef _LOG_LINEAR_MODEL_H_
#define _LOG_LINEAR_MODEL_H_

#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <set>

#include "StringUtils.h"
#include "FstUtils.h"
#include "LogLinearParams.h"

using namespace fst;
using namespace std;

#define NULL_SRC_TOKEN_ID 1

class LogLinearModel {

  // normalizes the parameters such that \sum_t p(t|s) = 1 \forall s
  void NormalizeParams();
  
  // creates an acceptor for a target sentence
  void CreateTgtFst(const vector<int>& tgtTokens, VectorFst<LogQuadArc>& tgtFst, set<int>& uniqueTgtTokens);

  // create an acceptor of many possible translations of the source sentence
  void CreateAllTgtFst(const vector<int>& srcTokens, 
		       int tgtSentLen, 
		       typename DiscriminativeLexicon::DiscriminativeLexicon lexicon, 
		       VectorFst<LogQuadArc>& allTgtFst,
		       set<int>& uniqueTgtTokens);

  void CreatePerSentGrammarFst(const vector<int>& srcTokens, const set<int>& uniqueTgtTokens, VectorFst<LogQuadArc>& grammarFst);
  
  void CreateSrcFst(const vector<int>& srcTokens, VectorFst<LogQuadArc>& srcFst);

  bool IsModelConverged();

  // zero all parameters
  void ClearParams();
  
  void LearnParameters(vector< VectorFst< LogArc > >& srcFsts, vector< VectorFst< LogArc > >& tgtFsts);
  
 public:

  LogLinearModel(const string& srcIntCorpusFilename, 
		 const string& tgtIntCorpusFilename, 
		 const string& outputFilenamePrefix, 
		 const Regularizer::Regularizer& regularizationType, 
		 const float regularizationConst, 
		 const LearningInfo& learningInfo);

  void PrintParams();

  void PersistParams(const string& outputFilename);
  
  // finds out what are the parameters needed by reading hte corpus, and assigning initial weights based on the number of co-occurences
  void InitParams();

  void Train();

  void Align();

  void BuildAlignmentFst(const vector<int>& srcTokens, const vector<int>& tgtTokens, VectorFst<LogQuadArc>& alignmentFst, bool tgtLineIsGiven, typename DiscriminativeLexicon::DiscriminativeLexicon lexicon, int sentId);

  void AddSentenceContributionToGradient(const VectorFst< LogQuadArc >& descriptorFst, 
					 const VectorFst< LogArc >& totalProbFst, 
					 LogLinearParams& gradient,
					 float logPartitionFunction,
					 int srcTokensCount,
					 int tgtTokensCount,
					 bool subtract);
  
  void AddRegularizerTerm(LogLinearParams& gradient);

 private:
  string srcCorpusFilename, tgtCorpusFilename, outputPrefix;
  LogLinearParams params;
  VectorFst<LogArc> grammarFst;
  LearningInfo learningInfo;
  // maps a srcTokenId to a map of tgtTokenIds and the number of times they co-occurred in a sent pair
  std::map<int, std::map<int, int> > srcTgtFreq;
  float regularizationConst;
  typename Regularizer::Regularizer regularizationType;
  OptUtils::OptMethod optimizationMethod;
};

#endif
