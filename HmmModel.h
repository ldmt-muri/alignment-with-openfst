#ifndef _HMM_MODEL_H_
#define _HMM_MODEL_H_

#include <iostream>
#include <fstream>
#include <math.h>

#include "LearningInfo.h"
#include "StringUtils.h"
#include "FstUtils.h"
#include "IAlignmentSampler.h"

using namespace fst;
using namespace std;

// this id is reserved for the unique source word (NULL). no other source word is allowed to take this id.
#define NULL_SRC_TOKEN_ID 1
// this is the src position of the null src word in any src sentence
#define NULL_SRC_TOKEN_POS 0
// the HMM word alignment model has parameters for p(a_i|a_{i-1}). 
// But, when i == 0, there's no a_{-1}. we can use this constant whenever we need a_{-1}
#define INITIAL_SRC_POS -1

// parameters for describing a multinomial distribution p(x)=y such that x is the key and y is a log probability
typedef map<int, float> MultinomialParam;
// parameters for describing a set of conditional multinomial distributions p(x|y)=z such that y is the first key, x is the nested key, z is a log probability
typedef map<int, MultinomialParam> ConditionalMultinomialParam;

class HmmModel : public IAlignmentSampler {

  // normalizes the parameters such that \sum_t p(t|s) = 1 \forall s
  void NormalizeFractionalCounts();
  void NormalizeParams(ConditionalMultinomialParam& params);
  
  // creates an fst for each target sentence
  void CreateTgtFsts(vector< VectorFst< LogTripleArc > >& targetFsts);

  // creates a 1st order markov fst for each source sentence
  void CreateSrcFsts(vector< VectorFst< LogTripleArc > >& srcFsts);

  // creates an fst for each src sentence, which remembers the last visited src token
  void Create1stOrderSrcFst(const vector<int>& srcTokens, VectorFst<LogTripleArc>& srcFst);

  void CreateGrammarFst();

  void CreatePerSentGrammarFsts(vector< VectorFst< LogTripleArc > >& perSentGrammarFsts);
  
  // zero all parameters
  void ClearParams(ConditionalMultinomialParam& params);
  void ClearFractionalCounts();
  
  void LearnParameters(vector< VectorFst< LogTripleArc > >& tgtFsts);
  
 public:

  HmmModel(const string& srcIntCorpusFilename, const string& tgtIntCorpusFilename, const string& outputFilenamePrefix, const LearningInfo& learningInfo);

  void PrintParams();
  void PrintParams(const ConditionalMultinomialParam& params);

  void PersistParams(const string& outputFilename);
  void PersistParams(ofstream& paramsFile, const ConditionalMultinomialParam& params);

  // finds out what are the parameters needed by reading hte corpus, and assigning initial weights based on the number of co-occurences
  void InitParams();

  void Train();

  void Align();

  void DeepCopy(const ConditionalMultinomialParam& original, 
		ConditionalMultinomialParam& duplicate);

  int SampleFromMultinomial(const MultinomialParam params);

  virtual void SampleAT(const vector<int>& srcTokens, int tgtLength, vector<int>& tgtTokens, vector<int>& alignments, double& hmmLogProb);

 private:
  string srcCorpusFilename, tgtCorpusFilename, outputPrefix;

  // tFractionalCounts are used (when normalized) to describe p(tgt_word|src_word). first key is the context (i.e. src_word) and nested key is the tgt_word.
  ConditionalMultinomialParam tFractionalCounts;
  // aParams are used to describe p(a_i|a_{i-1}). first key is the context (i.e. previous alignment a_{i-1}) and nested key is the current alignment a_i.
  ConditionalMultinomialParam aFractionalCounts, aParams;

  // Compose(perSentTgtFst * grammarFst * perSentSrcFst) => alignment fst
  // weight = (currentSrcPos, prevSrcPos, arcWeight)
  VectorFst<LogTripleArc> grammarFst;

  // configurations
  LearningInfo learningInfo;
};

#endif

