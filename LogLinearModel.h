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
#define NULL_SRC_TOKEN_POS 0
// the loglinear alignment model has some features as functions of the previous alignment 
// But, when evaluating the first tgt position, there's no previous alignemnt, 
// then we can use this constant whenever we need a_{-1}
#define INITIAL_SRC_POS -1

class LogLinearModel {

  // normalizes the parameters such that \sum_t p(t|s) = 1 \forall s
  void NormalizeParams();
  
  // creates an acceptor for a target sentence
  void CreateTgtFst(const vector<int>& tgtTokens, VectorFst<LogQuadArc>& tgtFst, set<int>& uniqueTgtTokens);

  // create an acceptor of many possible translations of the source sentence
  void CreateAllTgtFst(const vector<int>& srcTokens, 
		       int tgtSentLen, 
		       DiscriminativeLexicon::DiscriminativeLexicon lexicon, 
		       VectorFst<LogQuadArc>& allTgtFst,
		       set<int>& uniqueTgtTokens);

  void CreateGrammarFst();
  
  void Create1stOrderSrcFst(const vector<int>& srcTokens, VectorFst<LogQuadArc>& srcFst);

  void CreateSimpleSrcFst(const vector<int>& srcTokens, VectorFst<LogQuadArc>& srcFst);

  void CreateSampleAlignmentFst(const vector<int>& srcTokens,
				const vector< vector<int> >& translations, 
				const vector< vector<int> >& alignments, 
				const vector< double >& logProbs,
				VectorFst< LogQuadArc >& alignmentFst);

  bool IsModelConverged();

  // zero all parameters
  void ClearParams();
  
  void LearnParameters(vector< VectorFst< LogArc > >& srcFsts, vector< VectorFst< LogArc > >& tgtFsts);
  
 public:

  LogLinearModel(const string& srcIntCorpusFilename, 
		 const string& tgtIntCorpusFilename, 
		 const string& outputFilenamePrefix, 
		 const LearningInfo& learningInfo);

  void PrintParams();

  void PersistParams(const string& outputFilename);
  
  // finds out what are the parameters needed by reading hte corpus, and assigning initial weights based on the number of co-occurences
  void InitParams();

  void Train();

  void Align();

  void BuildAlignmentFst(const vector<int>& srcTokens, const vector<int>& tgtTokens, VectorFst<LogQuadArc>& alignmentFst, 
			 bool tgtLineIsGiven, DiscriminativeLexicon::DiscriminativeLexicon lexicon, 
			 int sentId, Distribution::Distribution distribution, VectorFst<LogQuadArc>& tgtFst);

  void AddSentenceContributionToGradient(const VectorFst< LogQuadArc >& descriptorFst, 
					 const VectorFst< LogArc >& totalProbFst, 
					 LogLinearParams& gradient,
					 const vector<int> &srcTokens,
					 int tgtTokensCount,
					 bool subtract);
  
  void AddRegularizerTerm(LogLinearParams& gradient);

 private:
  string srcCorpusFilename, tgtCorpusFilename, outputPrefix;
  LogLinearParams params;
  VectorFst<LogQuadArc> grammarFst;
  // lots of configurations, including but not limited to reguarlization, optimization method ...etc
  LearningInfo learningInfo;
  // maps a srcTokenId to a map of tgtTokenIds and the number of times they co-occurred in a sent pair
  std::map<int, std::map<int, int> > srcTgtFreq;
  // vectors specifiying which feature types to use (initialized in the constructor)
  std::vector<bool> enabledFeatureTypesFirstOrder, enabledFeatureTypesSimple;
  // number of sentences in the corpus
  int corpusSize;
};

#endif
