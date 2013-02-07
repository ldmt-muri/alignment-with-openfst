#ifndef _HMM_MODEL_H_
#define _HMM_MODEL_H_

#include <iostream>
#include <fstream>
#include <math.h>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/thread/thread.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/nonblocking.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>
#include <boost/mpi/collectives.hpp>

#include "LearningInfo.h"
#include "StringUtils.h"
#include "FstUtils.h"
#include "IAlignmentModel.h"
#include "IAlignmentSampler.h"
#include "Samplers.h"
#include "MultinomialParams.h"

using namespace fst;
using namespace std;
using namespace MultinomialParams;

// this id is reserved for the unique source word (NULL). no other source word is allowed to take this id.
#define NULL_SRC_TOKEN_ID 1
// this is the src position of the null src word in any src sentence
#define NULL_SRC_TOKEN_POS 0
// the HMM word alignment model has parameters for p(a_i|a_{i-1}). 
// But, when i == 0, there's no a_{-1}. we can use this constant whenever we need a_{-1}
#define INITIAL_SRC_POS -1

class HmmModel : public IAlignmentSampler, public IAlignmentModel {

  // normalizes the parameters such that \sum_t p(t|s) = 1 \forall s
  void NormalizeFractionalCounts();
  
  // creates an fst for each target sentence
  void CreateTgtFsts(vector< VectorFst< LogQuadArc > >& targetFsts);

  // creates a 1st order markov fst for each source sentence
  void CreateSrcFsts(vector< VectorFst< LogQuadArc > >& srcFsts);

  // creates an fst for each src sentence, which remembers the last visited src token
  void Create1stOrderSrcFst(const vector<int>& srcTokens, VectorFst<LogQuadArc>& srcFst);

  void CreateGrammarFst();

  void CreatePerSentGrammarFsts(vector< VectorFst< LogQuadArc > >& perSentGrammarFsts);
  
  // zero all parameters
  void ClearFractionalCounts();
  
  void LearnParameters(vector< VectorFst< LogQuadArc > >& tgtFsts);
  
  void BuildAlignmentFst(const VectorFst< LogQuadArc > &tgtFst, 
			 const VectorFst< LogQuadArc > &srcFst, 
			 VectorFst< LogQuadArc > &alignmentFst);

  void CreateTgtFst(const vector<int> tgtTokens, VectorFst< LogQuadArc > &tgtFst);
    
 public:

  HmmModel(const string& srcIntCorpusFilename, const string& tgtIntCorpusFilename, const string& outputFilenamePrefix, const LearningInfo& learningInfo);

  virtual void PrintParams();

  virtual void PersistParams(const string& outputFilename);

  // finds out what are the parameters needed by reading hte corpus, and assigning initial weights based on the number of co-occurences
  virtual void InitParams();

  virtual void Train();

  string AlignSent(vector<int> srcTokens, vector<int> tgtTokens);

  virtual void AlignTestSet(const string &srcTestSetFilename, const string &tgtTestSetFilename, const string &alignmentsFilename);

  void Align(const string &alignmentsFilename);
  
  virtual void Align();

  void DeepCopy(const ConditionalMultinomialParam& original, 
		ConditionalMultinomialParam& duplicate);

  virtual void SampleATGivenS(const vector<int>& srcTokens, int tgtLength, vector<int>& tgtTokens, vector<int>& alignments, double& hmmLogProb);

  virtual void SampleAGivenST(const std::vector<int> &srcTokens,
			      const std::vector<int> &tgtTokens,
			      std::vector<int> &alignments,
			      double &logProb);
  private:
  string outputPrefix;

  // tFractionalCounts are used (when normalized) to describe p(tgt_word|src_word). first key is the context (i.e. src_word) and nested key is the tgt_word.
  ConditionalMultinomialParam tFractionalCounts;
  // aParams are used to describe p(a_i|a_{i-1}). first key is the context (i.e. previous alignment a_{i-1}) and nested key is the current alignment a_i.
  ConditionalMultinomialParam aFractionalCounts, aParams;

  // Compose(perSentTgtFst * grammarFst * perSentSrcFst) => alignment fst
  // weight = (currentSrcPos, prevSrcPos, arcWeight)
  VectorFst<LogQuadArc> grammarFst;

  // configurations
  LearningInfo learningInfo;
  
  // vocab encoders
  VocabEncoder vocabEncoder;

  // training data (src, tgt)
  vector< vector<int> > srcSents, tgtSents;
};

#endif
