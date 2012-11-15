#include "LogLinearModel.h"
#include "HmmModel.h"

using namespace fst;
using namespace std;

// TODO: add parameters to set convergence criteria
void ParseParameters(int argc, char **argv, string& srcCorpusFilename, string &tgtCorpusFilename, string &outputFilepathPrefix) {
  assert(argc == 4);
  srcCorpusFilename = argv[1];
  tgtCorpusFilename = argv[2];
  outputFilepathPrefix = argv[3];
}

void Experimental() {
  cout << "nLog(1.0) = " << FstUtils::nLog(1.0) << endl;
  cout << "nLog(0.5) = " << FstUtils::nLog(0.5) << endl;
  cout << "nLog(0.25) = " << FstUtils::nLog(0.25) << endl;

  VectorFst< LogArc > fst1;
  int state0 = fst1.AddState();
  int state1 = fst1.AddState();
  int state2 = fst1.AddState();
  fst1.SetStart(state0);
  fst1.SetFinal(state1, 0);
  fst1.SetFinal(state2, 0);
  fst1.AddArc(state0, LogArc(1, 11, FstUtils::nLog(0.5), state0));
  fst1.AddArc(state0, LogArc(2, 11, FstUtils::nLog(0.5), state0));
  fst1.AddArc(state0, LogArc(1, 22, FstUtils::nLog(0.5), state0));
  fst1.AddArc(state0, LogArc(3, 22, FstUtils::nLog(0.5), state0));

  VectorFst< LogArc > temp, fst2;
  Compose(fst1, fst2, &temp);

  cout << "The temp fst looks like this:" << endl;
  FstUtils::PrintFstSummary(temp);

  LogWeight x = FstUtils::nLog(0.25);
  LogWeight y = FstUtils::nLog(0.5);
  LogWeight z = Divide(x, y);
  cerr << "0.25 / 0.5 = " << FstUtils::nExp(z.Value());
  cerr << x << " - " << y << " = " << z;

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

  // train the hmm model
  LearningInfo hmmLearningInfo;
  hmmLearningInfo.maxIterationsCount = 100;
  hmmLearningInfo.useMaxIterationsCount = true;
  hmmLearningInfo.minLikelihoodDiff = 1.0;
  hmmLearningInfo.useMinLikelihoodDiff = true;
  HmmModel hmmModel(srcCorpusFilename, tgtCorpusFilename, outputFilenamePrefix, hmmLearningInfo);
  hmmModel.Train();

  // initialize the loglinear model
  Regularizer::Regularizer regularizationType = Regularizer::NONE;
  float regularizationConst = 0.01;
  LearningInfo learningInfo;
  learningInfo.useMaxIterationsCount = true;
  learningInfo.useMinLikelihoodDiff = true;
  learningInfo.minLikelihoodDiff = 0.01;
  learningInfo.maxIterationsCount = 10;
  learningInfo.optimizationMethod.algorithm = OptUtils::STOCHASTIC_GRADIENT_DESCENT;
  learningInfo.optimizationMethod.miniBatchSize = 1;
  learningInfo.samplesCount = 1000;
  learningInfo.distATGivenS = Distribution::LOCAL;
  learningInfo.customDistribution = &hmmModel;
  LogLinearModel model(srcCorpusFilename, tgtCorpusFilename, outputFilenamePrefix, regularizationType, regularizationConst, learningInfo);

  // train model parameters
  model.Train();

}
