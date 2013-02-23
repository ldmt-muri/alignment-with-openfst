#include "HmmModel2.h"

using namespace std;
using namespace fst;
using namespace MultinomialParams;
using namespace boost;

HmmModel2::HmmModel2(const string &textFilename, 
		     const string &outputPrefix, 
		     LearningInfo &learningInfo) : 
  vocabEncoder(textFilename) {
  assert(false);
}

void HmmModel2::InitParams(){
  assert(false);
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
  InitParams();
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
    thetaMle.UnnormalizedParamsIntoNormalizedNlogParams();
    nlogTheta = thetaMle;
    gammaMle.UnnormalizedParamsIntoNormalizedNlogParams();
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
