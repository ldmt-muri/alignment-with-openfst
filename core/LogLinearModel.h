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

class LogLinearModel : public IAlignmentModel {

  // normalizes the parameters such that \sum_t p(t|s) = 1 \forall s
  void NormalizeParams();
  
  // creates an acceptor for a target sentence
  void CreateTgtFst(const vector<int>& tgtTokens, VectorFst<FstUtils::LogQuadArc>& tgtFst, set<int>& uniqueTgtTokens);

  // create an acceptor of many possible translations of the source sentence
  void CreateAllTgtFst(const set<int>& srcTokens, 
		       int tgtSentLen, 
		       VectorFst<FstUtils::LogQuadArc>& allTgtFst,
		       set<int>& uniqueTgtTokens);

  void CreateGrammarFst();
  
  void Create1stOrderSrcFst(const vector<int>& srcTokens, VectorFst<FstUtils::LogQuadArc>& srcFst);

  void CreateSimpleSrcFst(const vector<int>& srcTokens, VectorFst<FstUtils::LogQuadArc>& srcFst);

  void CreateSampleAlignmentFst(const vector<int>& srcTokens,
				const vector< vector<int> >& translations, 
				const vector< vector<int> >& alignments, 
				const vector< double >& logProbs,
				VectorFst< FstUtils::LogQuadArc >& alignmentFst);

  bool IsModelConverged();

  // zero all parameters
  void ClearParams();
  
  void LearnParameters(vector< VectorFst< LogArc > >& srcFsts, vector< VectorFst< LogArc > >& tgtFsts);
  
 public:

  LogLinearModel(const string& srcIntCorpusFilename, 
		 const string& tgtIntCorpusFilename, 
		 const string& outputFilenamePrefix, 
		 const LearningInfo& learningInfo);

  virtual void PrintParams();

  virtual void PersistParams(const string& outputFilename);
  
  // finds out what are the parameters needed by reading hte corpus, and assigning initial weights based on the number of co-occurences
  virtual void InitParams();

  virtual void Train();

  string AlignSent(vector<int> srcTokens, vector<int> tgtTokens);

  virtual void Align();

  virtual void AlignTestSet(const string &srcTestSetFilename, const string &tgtTestSetFilename, const string &outputAlignmentsFilename);

  void BuildAlignmentFst(const vector<int>& srcTokens, const vector<int>& tgtTokens, VectorFst<FstUtils::LogQuadArc>& alignmentFst, 
			 bool tgtLineIsGiven, 
			 int sentId, Distribution::Distribution distribution, VectorFst<FstUtils::LogQuadArc>& tgtFst);

  void AddSentenceContributionToGradient(const VectorFst< FstUtils::LogQuadArc >& descriptorFst, 
					 const VectorFst< LogArc >& totalProbFst, 
					 LogLinearParams& gradient,
					 const vector<int> &srcTokens,
					 int tgtTokensCount,
					 bool subtract);
  
  void AddRegularizerTerm(LogLinearParams& gradient);
  void BuildAllSentTranslationFst(int tgtSentLength, fst::VectorFst<FstUtils::LogQuadArc>& sentTranslationFst);

 private:
  string srcCorpusFilename, tgtCorpusFilename, outputPrefix;
  LogLinearParams params;
  VectorFst<FstUtils::LogQuadArc> grammarFst;
  // lots of configurations, including but not limited to reguarlization, optimization method ...etc
  LearningInfo learningInfo;
  // maps a srcTokenId to a map of tgtTokenIds and the number of times they co-occurred in a sent pair
  std::map<int, std::map<int, int> > srcTgtFreq;
  // number of sentences in the corpus
  int corpusSize;
  // vector of (allTgtSentenceFst o grammarFst), indexed by target length
  std::vector<fst::VectorFst<FstUtils::LogQuadArc> > tgtLengthToSentTranslationFst;
  // set of tgt types
  std::set<int> tgtTypes;
  // set of src types
  std::set<int> srcTypes;
};

#endif
