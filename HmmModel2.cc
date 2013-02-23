#include "HmmModel2.h"

using namespace std;
using namespace fst;
using namespace MultinomialParams;
using namespace boost;

HmmModel2::HmmModel2(const string &textFilename, 
		     const string &outputPrefix, 
		     LearningInfo &learningInfo,
		     unsigned numberOfLabels) : 
  vocabEncoder(textFilename),
  gaussianSampler(0.0, 1.0),
  START_OF_SENTENCE_Y_VALUE(2),
  FIRST_ALLOWED_LABEL_VALUE(4) {
  
  this->outputPrefix = outputPrefix;
  this->learningInfo = learningInfo;
  
  assert(numberOfLabels > 1);
  yDomain.insert(START_OF_SENTENCE_Y_VALUE); // the conceptual yValue of word at position -1 in a sentence
  for(unsigned labelId = START_OF_SENTENCE_Y_VALUE + 1; labelId < START_OF_SENTENCE_Y_VALUE + numberOfLabels + 1 ; labelId++) {
    yDomain.insert(labelId);
  }

  // populate the X domain with all types in the vocabEncoder
  for(map<int,string>::const_iterator vocabIter = vocabEncoder.intToToken.begin();
      vocabIter != vocabEncoder.intToToken.end();
      vocabIter++) {
    if(vocabIter->second == "_unk_") {
      continue;
    }
    xDomain.insert(vocabIter->first);
  }
  // zero is reserved for FST epsilon
  assert(xDomain.count(0) == 0);
  
  // populate the observations vector with encoded sentences
  vocabEncoder.Read(textFilename, observations);

  // initialize theta and gamma parameters
  InitParams();
}

// gaussian initialization of the multinomial params
void HmmModel2::InitParams(){
  nlogTheta.GaussianInit();
  nlogGamma.GaussianInit();
}

// builds the lattice of all possible label sequences
void HmmModel2::BuildThetaGammaFst(unsigned sentId, VectorFst<LogArc> &fst) {
  assert(false);
}

// builds the lattice of all possible label sequences, also computes potentials
void HmmModel2::BuildThetaGammaFst(unsigned sentId, VectorFst<LogArc> &fst, vector<fst::LogWeight> &alphas, vector<fst::LogWeight> &betas) {
  assert(false);
}

void HmmModel2::UpdateMle(const VectorFst<LogArc> &fst, 
	       const vector<fst::LogWeight> &alphas, 
	       const vector<fst::LogWeight> &betas, 
	       ConditionalMultinomialParam<int> &thetaMle, 
	       ConditionalMultinomialParam<int> &gammaMle){ 
  assert(false);
}
  
// EM training of the HMM
void HmmModel2::Train(){
  do {
    
    // expectation
    double nloglikelihood = 0;
    ConditionalMultinomialParam<int> thetaMle, gammaMle;
    for(unsigned sentId = 0; sentId < observations.size(); sentId++) {
      VectorFst<LogArc> fst;
      vector<fst::LogWeight> alphas, betas; 
      BuildThetaGammaFst(sentId, fst, alphas, betas);
      UpdateMle(fst, alphas, betas, thetaMle, gammaMle);
      nloglikelihood += betas[0].Value();
    }

    // maximization
    thetaMle.ConvertUnnormalizedParamsIntoNormalizedNlogParams();
    nlogTheta = thetaMle;
    gammaMle.ConvertUnnormalizedParamsIntoNormalizedNlogParams();
    nlogGamma = gammaMle;

    // check convergence
    learningInfo.logLikelihood.push_back(nloglikelihood);
  } while(learningInfo.IsModelConverged());
}

void HmmModel2::Label(vector<int> &tokens, vector<int> &labels){
  assert(false);
}

void HmmModel2::Label(vector<string> &tokens, vector<int> &labels){
  assert(false);
}

void HmmModel2::Label(vector<vector<int> > &tokens, vector<vector<int> > &lables) {
  assert(false);
}

void HmmModel2::Label(vector<vector<string> > &tokens, vector<vector<int> > &labels) {
  assert(false);
}

void HmmModel2::Label(string &inputFilename, string &outputFilename) {
  assert(false);
}

// evaluate
double HmmModel2::ComputeVariationOfInformation(std::string &labelsFilename, std::string &goldLabelsFilename) {
  assert(false);
  return 0;
}

double HmmModel2::ComputeManyToOne(std::string &aLabelsFilename, std::string &bLabelsFilename) {
  assert(false);
  return 0;
}
