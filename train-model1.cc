#include <iostream>
#include <fstream>
#include <math.h>

#include <fst/fstlib.h>

#define LOG_ZERO 30
#define NULL_SRC_TOKEN_ID 1

using namespace fst;
using namespace std;

typedef map<int, map< int, float > >  Model1Param;

struct LearningInfo{
  LearningInfo() {
    useMaxIterationsCount = false;
    useMinLikelihoodDiff = false;
    iterationsCount = 0;
  }

  // criteria 1
  bool useMaxIterationsCount;
  int maxIterationsCount;

  // criteria 2
  bool useMinLikelihoodDiff;
  float minLikelihoodDiff;

  // output
  int iterationsCount;
  vector<float> logLikelihood;
};

float nLog(float prob) {
  return -1.0 * log(prob);
}

void PrintFstSummary(VectorFst<LogArc>& fst) {
  cout << "states:" << endl;
  for(StateIterator< VectorFst<LogArc> > siter(fst); !siter.Done(); siter.Next()) {
    const LogArc::StateId &stateId = siter.Value();
    string final = fst.Final(stateId) == 0? " FINAL": "";
    string initial = fst.Start() == stateId? " START" : "";
      cout << "state:" << stateId << initial << final << endl;
    cout << "arcs:" << endl;
    for(ArcIterator< VectorFst<LogArc> > aiter(fst, stateId); !aiter.Done(); aiter.Next()) {
      const LogArc &arc = aiter.Value();
      cout << arc.ilabel << ":" << arc.olabel << " " <<  stateId << "-->" << arc.nextstate << " " << arc.weight << endl;
    } 
    cout << endl;
  }
}

void ParseParameters(int argc, char **argv, string& srcCorpusFilename, string &tgtCorpusFilename, string &outputFilepathPrefix) {
  assert(argc == 4);
  srcCorpusFilename = argv[1];
  tgtCorpusFilename = argv[2];
  outputFilepathPrefix = argv[3];
}

// string split
void SplitString(const std::string& s, char delim, std::vector<std::string>& elems) {
  std::stringstream ss(s);
  std::string item;
  while(std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
}

// read int tokens
void ReadIntTokens(const std::string& sentence, std::vector<int>& intTokens) {
  vector<string> stringTokens;
  SplitString(sentence, ' ', stringTokens);
  for (vector<string>::iterator tokensIter = stringTokens.begin(); 
    tokensIter < stringTokens.end(); tokensIter++) {
    int intToken;
    stringstream stringToken(*tokensIter);
    stringToken >> intToken;
    intTokens.push_back(intToken);
  }
}

// creates an fst for each target sentence
void CreateTgtFsts(const string& tgtCorpusFilename, vector< VectorFst< LogArc > >& targetFsts) {
  ifstream tgtCorpus(tgtCorpusFilename.c_str(), ios::in); 

  // for each line
  string line;
  while(getline(tgtCorpus, line)) {

    // read the list of integers representing target tokens
    vector< int > intTokens;
    ReadIntTokens(line, intTokens);

    // create the fst
    VectorFst< LogArc > tgtFst;
    int statesCount = intTokens.size() + 1;
    for(int stateId = 0; stateId < intTokens.size()+1; stateId++) {
      int temp = tgtFst.AddState();
      assert(temp == stateId);
      if(stateId == 0) continue;
      tgtFst.AddArc(stateId-1, LogArc(intTokens[stateId-1], intTokens[stateId-1], 0, stateId));
    }
    tgtFst.SetStart(0);
    tgtFst.SetFinal(intTokens.size(), 0);
    ArcSort(&tgtFst, ILabelCompare<LogArc>());
    targetFsts.push_back(tgtFst);

    // for debugging
	// PrintFstSummary(tgtFst);
  }
  tgtCorpus.close();
}

// normalizes the parameters such that \sum_t p(t|s) = 1 \forall s
void NormalizeParams(Model1Param& params) {
  // iterate over src tokens in the model
  for(Model1Param::iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
    //cout << "=======================" << endl << "normalizing srcToken = " << (*srcIter).first << endl;
    map< int, float > *translations = &(*srcIter).second;
    float fTotalProb = 0.0;
    //cout << "totalProb = " << fTotalProb << endl;
    // iterate over tgt tokens logsumming over the logprob(tgt|src) 
    for(map< int, float >::iterator tgtIter = translations->begin(); tgtIter != translations->end(); tgtIter++) {
      LogWeight temp = (*tgtIter).second;
      fTotalProb += exp(-1.0 * temp.Value());
      //cout << "fTotalProb += " << exp(-1.0 * temp.Value()) << "(i.e. e^-" << temp.Value() << ") ==> " << fTotalProb << endl;
    }
    // exponentiate to find p(*|src) before normalization
    //    float fLogTotalProb = (float) logTotalProb.Value();
    //cout << "totalProb = " << fTotalProb << endl << endl;
    // iterate again over tgt tokens dividing p(tgt|src) by p(*|src)
    float fVerifyTotalProb = 0.0;
    for(map< int, float >::iterator tgtIter = translations->begin(); tgtIter != translations->end(); tgtIter++) {
      float fUnnormalized = exp( -1.0 * (*tgtIter).second );
      float fNormalized = fUnnormalized / fTotalProb;
      fVerifyTotalProb += fNormalized;
      float fLogNormalized = -1 * log(fNormalized);
      //cout << "prob(" << (*tgtIter).first << "|" << (*srcIter).first << ") = " << fUnnormalized << " ==> " << fNormalized << endl;
      //cout << "-logprob(" << (*tgtIter).first << "|" << (*srcIter).first << ") = " << (*tgtIter).second << " ==> ";
      (*tgtIter).second = fLogNormalized;
      //cout << (*tgtIter).second << endl;
    }
    //cout << "verify totalProb = " << fVerifyTotalProb << endl << endl;
  }
}

void PrintParams(const Model1Param params) {
  // iterate over src tokens in the model
  int counter =0;
  for(Model1Param::const_iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
    map< int, float > translations = (*srcIter).second;
    // iterate over tgt tokens 
    for(map< int, float >::const_iterator tgtIter = translations.begin(); tgtIter != translations.end(); tgtIter++) {
      cout << "-logp(" << (*tgtIter).first << "|" << (*srcIter).first << ")=log(" << exp(-1.0 * (*tgtIter).second) << ")=" << (*tgtIter).second << endl;
    }
  } 
}

void PersistParams(Model1Param& params, string outputFilename) {
  ofstream paramsFile(outputFilename.c_str());
  cout << "writing model params at " << outputFilename << endl;

  for (Model1Param::const_iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
    for (map<int, float>::const_iterator tgtIter = srcIter->second.begin(); tgtIter != srcIter->second.end(); tgtIter++) {
      // line format: 
      // srcTokenId tgtTokenId logP(tgtTokenId|srcTokenId) p(tgtTokenId|srcTokenId)
      paramsFile << srcIter->first << " " << tgtIter->first << " " << tgtIter->second << " " << exp(-1.0 * tgtIter->second) << endl;
    }
  }
  paramsFile.close();
}

// finds out what are the parameters needed by reading hte corpus, and assigning initial weights based on the number of co-occurences
void InitParams(const string& srcCorpusFilename, const string& tgtCorpusFilename, Model1Param& params, const string& initModelFilename) {
  ifstream srcCorpus(srcCorpusFilename.c_str(), ios::in);
  ifstream tgtCorpus(tgtCorpusFilename.c_str(), ios::in); 

  // for each line
  string srcLine, tgtLine;
  while(getline(tgtCorpus, tgtLine) && getline(srcCorpus, srcLine)) {

    // read the list of integers representing target tokens
    vector< int > tgtTokens, srcTokens;
    ReadIntTokens(srcLine, srcTokens);
    // we want to allow target words to align to NULL (which has srcTokenId = 1).
    srcTokens.push_back(NULL_SRC_TOKEN_ID); 
    ReadIntTokens(tgtLine, tgtTokens);
    
    // for each srcToken
    for(int i=0; i<srcTokens.size(); i++) {
      int srcToken = srcTokens[i];
      // get the corresponding map of tgtTokens (and the corresponding probabilities)
      map<int, float> *translations = &(params[srcToken]);

      // for each tgtToken
      for (int j=0; j<tgtTokens.size(); j++) {
	int tgtToken = tgtTokens[j];
	// if this the first time the pair(tgtToken, srcToken) is experienced, give it a value of 1 (i.e. prob = exp(-1) ~= 1/3)
	if( translations->count(tgtToken) == 0) {
	  (*translations)[tgtToken] = nLog(1/3.0);
        } else {
	  // otherwise, add nLog(1/3) to the original value, effectively counting the number of times 
	  // this srcToken-tgtToken pair appears in the corpus
	  (*translations)[tgtToken] = Plus( LogWeight((*translations)[tgtToken]), LogWeight(nLog(1/3.0)) ).Value();
	}
      }
    }
  }

  srcCorpus.close();
  tgtCorpus.close();

  NormalizeParams(params);

  // persist initial model
  cout << "persisting initial model" << endl;
  PersistParams(params, initModelFilename);
}

void CreateGrammarFst(const Model1Param& params, VectorFst< LogArc >& grammarFst) {
  // clear grammar
  if (grammarFst.NumStates() > 0) {
    grammarFst.DeleteArcs(grammarFst.Start());
    grammarFst.DeleteStates();
  }

  // create the only state in this fst, and make it initial and final
  LogArc::StateId dummy = grammarFst.AddState();
  assert(dummy == 0);
  grammarFst.SetStart(0);
  grammarFst.SetFinal(0, 0);
  int fromState = 0, toState = 0;
  for(Model1Param::const_iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
    for(map<int,float>::const_iterator tgtIter = (*srcIter).second.begin(); tgtIter != (*srcIter).second.end(); tgtIter++) {
      int tgtToken = (*tgtIter).first;
      int srcToken = (*srcIter).first;
      float paramValue = (*tgtIter).second;
      grammarFst.AddArc(fromState, LogArc(tgtToken, srcToken, paramValue, toState));
    }
  }
  ArcSort(&grammarFst, ILabelCompare<LogArc>());
  //  PrintFstSummary(grammarFst);
}

bool IsModelConverged(const LearningInfo& learningInfo) {
  assert(learningInfo.useMaxIterationsCount || learningInfo.useMinLikelihoodDiff);

  // logging
  if(learningInfo.useMaxIterationsCount) {
    cout << "iterationsCount = " << learningInfo.iterationsCount << ". max = " << learningInfo.maxIterationsCount << endl;
  }
  if(learningInfo.useMinLikelihoodDiff && 
     learningInfo.iterationsCount > 1) {
    cout << "likelihoodDiff = " << abs(learningInfo.logLikelihood[learningInfo.iterationsCount-1] - 
				       learningInfo.logLikelihood[learningInfo.iterationsCount-2]) << ". min = " << learningInfo.minLikelihoodDiff << endl;
  }

  // check for convergnece conditions
  if(learningInfo.useMaxIterationsCount && 
     learningInfo.maxIterationsCount < learningInfo.iterationsCount) {
    return true;
  } 
  if(learningInfo.useMinLikelihoodDiff && 
     learningInfo.iterationsCount > 1 &&
     learningInfo.minLikelihoodDiff > abs(learningInfo.logLikelihood[learningInfo.iterationsCount-1] - 
					  learningInfo.logLikelihood[learningInfo.iterationsCount-2])) {
    return true;
  } 

  // none of the convergence conditions apply!
  return false;
}

// zero all parameters
void ClearParams(Model1Param& params) {
  for (Model1Param::iterator srcIter = params.begin(); srcIter != params.end(); srcIter++) {
    for (map<int, float>::iterator tgtIter = srcIter->second.begin(); tgtIter != srcIter->second.end(); tgtIter++) {
      tgtIter->second = LOG_ZERO;
    }
  }
}

void LearnParameters(Model1Param& params, VectorFst< LogArc >& grammarFst, vector< VectorFst< LogArc > >& srcFsts,
		     vector< VectorFst< LogArc > >& tgtFsts, LearningInfo& learningInfo, string outputFilenamePrefix) {
  do {
    float logLikelihood = 0;
    //    cout << "iteration's loglikelihood = " << logLikelihood << endl;
    
    // this vector will be used to accumulate fractional counts of parameter usages
    ClearParams(params);

    // iterate over sentences
    int sentsCounter = 0;
    for( vector< VectorFst< LogArc > >::const_iterator tgtIter = tgtFsts.begin(), srcIter = srcFsts.begin(); 
	 tgtIter != tgtFsts.end() && srcIter != srcFsts.end(); 
	 tgtIter++, srcIter++) {

      // build the alignment fst
      VectorFst< LogArc > tgtFst = *tgtIter, srcFst = *srcIter, temp, alignmentFst;
      Compose(tgtFst, grammarFst, &temp);
      ArcSort(&temp, ILabelCompare<LogArc>());
      Compose(temp, srcFst, &alignmentFst);
      //      PrintFstSummary(alignmentFst);

      // compute alignments count
      double alignmentsCount = 1;
      for (int stateId = 0; stateId < alignmentFst.NumStates() ;stateId++) {
	if (alignmentFst.Final(stateId) != LogWeight::Zero()) continue;
	int sources = 0;
	for (ArcIterator<VectorFst< LogArc > > arcIter(alignmentFst, stateId);
	     !arcIter.Done();
	     arcIter.Next()) {
	  sources++;
	}
	alignmentsCount *= sources;
	assert(alignmentsCount > 1);
      }

      // run forward/backward for this sentence
      vector<LogWeight> alphas, betas;
      ShortestDistance(alignmentFst, &alphas, false);
      ShortestDistance(alignmentFst, &betas, true);
      float fShiftedSentLogLikelihood = betas[alignmentFst.Start()].Value();
      //      cout << "sent's shifted log likelihood = " << fShiftedSentLogLikelihood << endl;
      //      float fSentLikelihood = exp(-1.0 * fShiftedSentLogLikelihood) / alignmentsCount;
      //      cout << "sent's likelihood = " << fSentLikelihood << endl;
      //      float fSentLogLikelihood = nLog(fSentLikelihood);
      //      cout << "nLog(alignmentsCount) = nLog(" << alignmentsCount << ") = " << nLog(alignmentsCount) << endl;
      float fSentLogLikelihood = fShiftedSentLogLikelihood - nLog(alignmentsCount);
      //      cout << "sent's loglikelihood = " << fSentLogLikelihood << endl;

      // compute fractional counts for model parameters
      for (int stateId = 0; stateId < alignmentFst.NumStates() ;stateId++) {
	for (ArcIterator<VectorFst< LogArc > > arcIter(alignmentFst, stateId);
	     !arcIter.Done();
	     arcIter.Next()) {
	  int srcToken = arcIter.Value().olabel, tgtToken = arcIter.Value().ilabel;
	  int fromState = stateId, toState = arcIter.Value().nextstate;

	  // probability of using this parameter given this sentence pair and the previous model
	  LogWeight currentParamLogProb = arcIter.Value().weight;
	  LogWeight unnormalizedPosteriorLogProb = Times(Times(alphas[fromState], currentParamLogProb), betas[toState]);
	  //float fUnnormalizedPosteriorProb = exp(-1.0 * unnormalizedPosteriorLogProb.Value());
	  //float fNormalizedPosteriorProb = (fUnnormalizedPosteriorProb / alignmentsCount) / fSentLikelihood;
	  //float fNormalizedPosteriorLogProb = -1.0 * log(fNormalizedPosteriorProb);
	  float fNormalizedPosteriorLogProb = unnormalizedPosteriorLogProb.Value() - nLog(alignmentsCount) - fSentLogLikelihood;

	  // logging
	  /*
	  if(srcToken == 2 && tgtToken == 2){

	    cout << "The alignment FST looks like this: " << endl;
	    cout << "================================== " << endl;
	    PrintFstSummary(alignmentFst);
	    
	    cout << endl << "Updates on prob(2|2): " << endl;
	    cout << "===================== " << endl;
	    cout << " before: " << params[srcToken][tgtToken] << endl;
	    cout << " before: logProb(2|2) = " << currentParamLogProb.Value() << endl;
	    cout << " alpha[fromState] = " << alphas[fromState] << endl;
	    cout << " beta[toState] = " << betas[toState] << endl;
	    cout << " unnormalized logProb(2-2|sent) = alpha[fromState] logTimes logProb(2|2) logTimes beta[toState] = " << unnormalizedPosteriorLogProb.Value() << endl;
	    cout << " unnormalized prob(2-2|sent) = " << fUnnormalizedPosteriorProb << endl;
	    cout << " prob(2-2|sent) = " << fNormalizedPosteriorProb << endl;
	    cout << " logProb(2-2|sent) = " << fNormalizedPosteriorLogProb << endl;
	    cout << " adding: " << fNormalizedPosteriorLogProb << endl;
	  }
	  */
	  
	  // append the fractional count for this parameter
	  params[srcToken][tgtToken] = Plus(LogWeight(params[srcToken][tgtToken]), LogWeight(fNormalizedPosteriorLogProb)).Value();

	  // logging
	  /*
	  if(srcToken == 2 && tgtToken == 2){
	    cout << " after: " << params[srcToken][tgtToken] << endl << endl;
	  }
	  */
	  
	}
      }   
      
      // update the iteration log likelihood with this sentence's likelihod
      logLikelihood += fSentLogLikelihood;
      //      cout << "iteration's loglikelihood = " << logLikelihood << endl;

      // logging
      if (++sentsCounter % 50 == 0) {
	cout << sentsCounter << " sents processed.." << endl;
      }
    }

    // normalize fractional counts such that \sum_t p(t|s) = 1 \forall s
    NormalizeParams(params);

    // persist parameters
    stringstream filename;
    filename << outputFilenamePrefix << ".param." << learningInfo.iterationsCount;
    PersistParams(params, filename.str());

    // create the new grammar
    CreateGrammarFst(params, grammarFst);

    // logging
    cout << "iterations # " << learningInfo.iterationsCount << " - total loglikelihood = " << logLikelihood << endl << endl;

    // update learningInfo
    learningInfo.logLikelihood.push_back(logLikelihood);
    learningInfo.iterationsCount++;

    // check for convergence
  } while(!IsModelConverged(learningInfo));
}

// returns a list of acceptors of the source sentences in any order. 
// Each acceptor has a single state with arcs representing src tokens in addition to NULL (srcTokenId = 0)
void CreateSrcFsts(const string& srcCorpusFilename, vector< VectorFst< LogArc > >& srcFsts) {
  ifstream srcCorpus(srcCorpusFilename.c_str(), ios::in); 

  // for each line
  string line;
  while(getline(srcCorpus, line)) {

    // read the list of integers representing source tokens
    vector< int > intTokens;
    ReadIntTokens(line, intTokens);
    // allow null alignments
    intTokens.push_back(NULL_SRC_TOKEN_ID);

    // create the fst
    VectorFst< LogArc > srcFst;
    int stateId = srcFst.AddState();
    assert(stateId == 0);
    for(vector<int>::const_iterator tokenIter = intTokens.begin(); tokenIter != intTokens.end(); tokenIter++) {
      srcFst.AddArc(stateId, LogArc(*tokenIter, *tokenIter, 0, stateId));
    }
    srcFst.SetStart(stateId);
    srcFst.SetFinal(stateId, 0);
    ArcSort(&srcFst, ILabelCompare<LogArc>());
    srcFsts.push_back(srcFst);

    // for debugging
    //    PrintFstSummary(srcFst);
  }
  srcCorpus.close();
}

void Experimental() {
  cout << "nLog(1.0) = " << nLog(1.0) << endl;
  cout << "nLog(0.5) = " << nLog(0.5) << endl;
  cout << "nLog(0.25) = " << nLog(0.25) << endl;

  VectorFst< LogArc > fst1;
  int state0 = fst1.AddState();
  int state1 = fst1.AddState();
  int state2 = fst1.AddState();
  fst1.SetStart(state0);
  fst1.SetFinal(state1, 0);
  fst1.SetFinal(state2, 0);
  fst1.AddArc(state0, LogArc(1, 11, nLog(0.5), state0));
  fst1.AddArc(state0, LogArc(2, 11, nLog(0.5), state0));
  fst1.AddArc(state0, LogArc(1, 22, nLog(0.5), state0));
  fst1.AddArc(state0, LogArc(3, 22, nLog(0.5), state0));

  VectorFst< LogArc > temp, fst2;
  Compose(fst1, fst2, &temp);

  cout << "The temp fst looks like this:" << endl;
  PrintFstSummary(temp);

  /*
  vector<LogWeight> alphas, betas;
  ShortestDistance(alignment, &alphas, false);
  ShortestDistance(alignment, &betas, true);
  int stateId = 0;
  for (vector<LogWeight>::const_iterator alphasIter = alphas.begin(); alphasIter != alphas.end(); alphasIter++) {
    cout << "alphas[" << stateId++ << "] = " << alphasIter->Value() << " = e^" << exp(-1.0 * alphasIter->Value()) << endl;
  }
  stateId = 0;
  for (vector<LogWeight>::const_iterator betasIter = betas.begin(); betasIter != betas.end(); betasIter++) {
    cout << "betas[" << stateId++ << "] = " << betasIter->Value() << " = e^" << exp(-1.0 * betasIter->Value()) << endl;
  }
  */
}

int main(int argc, char **argv) {
  //  Experimental();
  //  return 0;

  // parse arguments
  cout << "parsing arguments" << endl;
  string srcCorpusFilename, tgtCorpusFilename, outputFilenamePrefix;
  ParseParameters(argc, argv, srcCorpusFilename, tgtCorpusFilename, outputFilenamePrefix);

  // create tgt fsts
  cout << "create tgt fsts" << endl;
  vector< VectorFst <LogArc> > tgtFsts;
  CreateTgtFsts(tgtCorpusFilename, tgtFsts);

  // initialize model 1 scores
  cout << "init model1 params" << endl;
  Model1Param params;
  stringstream initialModelFilename;
  initialModelFilename << outputFilenamePrefix << ".param.init";
  InitParams(srcCorpusFilename, tgtCorpusFilename, params, initialModelFilename.str());

  // create the initial grammar FST
  cout << "create grammar fst" << endl;
  VectorFst<LogArc> grammarFst;
  CreateGrammarFst(params, grammarFst);

  // create srcFsts
  cout << "create src fsts" << endl;
  vector< VectorFst <LogArc> > srcFsts;
  CreateSrcFsts(srcCorpusFilename, srcFsts);

  // training iterations
  cout << "train!" << endl;
  float convergenceCriterion = 1.0;
  LearningInfo learningInfo;
  learningInfo.useMaxIterationsCount = true;
  learningInfo.maxIterationsCount = 10;
  LearnParameters(params, grammarFst, srcFsts, tgtFsts, learningInfo, outputFilenamePrefix);

  // persist parameters
  cout << "persist" << endl;
  PersistParams(params, outputFilenamePrefix + ".param.final");
}
