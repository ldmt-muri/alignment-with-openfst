#include "LogLinearModel.h"
#include "HmmModel.h"
#include "VocabEncoder.h"
#include "IbmModel1.h"

using namespace fst;
using namespace std;

void ParseParameters(int argc, char **argv, 
		     string &srcTrainSetFilename, 
		     string &tgtTrainSetFilename, 
		     string &srcTestSetFilename,
		     string &tgtTestSetFilename,
		     string &srcVocabFilename, 
		     string &tgtVocabFilename, 
		     string &outputFilepathPrefix) {
  assert(argc == 8);
  srcTrainSetFilename = argv[1];
  tgtTrainSetFilename = argv[2];
  srcTestSetFilename = argv[3];
  tgtTestSetFilename = argv[4];
  srcVocabFilename = argv[5];
  tgtVocabFilename = argv[6];
  outputFilepathPrefix = argv[7];
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

}

int main(int argc, char **argv) {
  //  Experimental();
  //  return 0;

  // parse arguments
  cerr << "parsing arguments" << endl;
  string srcCorpusFilename, tgtCorpusFilename, srcTestFilename, tgtTestFilename, srcVocabFilename, tgtVocabFilename, outputFilenamePrefix;
  ParseParameters(argc, argv, srcCorpusFilename, tgtCorpusFilename, srcTestFilename, tgtTestFilename, srcVocabFilename, tgtVocabFilename, outputFilenamePrefix);

  // train the hmm model
  LearningInfo hmmLearningInfo;
  hmmLearningInfo.maxIterationsCount = 1;
  hmmLearningInfo.useMaxIterationsCount = true;
  hmmLearningInfo.minLikelihoodDiff = 0.01;
  hmmLearningInfo.useMinLikelihoodDiff = true;
  string proposalTrainingOutputFilenamePrefix = outputFilenamePrefix + ".proposal";
  HmmModel hmmModel(srcCorpusFilename, tgtCorpusFilename, proposalTrainingOutputFilenamePrefix, hmmLearningInfo);
  hmmModel.Train();
  cerr << "finished HMM training." << endl << endl;
  string dummy;
  //  cin >> dummy;

  // initialize ibm 1 forward logprobs
  LearningInfo model1LearningInfo;
  model1LearningInfo.minLikelihoodDiff = 0.01;
  model1LearningInfo.useMinLikelihoodDiff = true;
  string ibm1ForwardOutputFilenamePrefix = outputFilenamePrefix + ".ibm1fwd";
  IbmModel1 ibm1ForwardModel(srcCorpusFilename, tgtCorpusFilename, ibm1ForwardOutputFilenamePrefix, model1LearningInfo);
  ibm1ForwardModel.Train();
  cerr << "finished forward ibm1 training." << endl << endl;
  //  cin >> dummy;

  // initialize ibm 1 backward logprobs
  string ibm1BackwardOutputFilenamePrefix = outputFilenamePrefix + ".ibm1bwd";
  IbmModel1 ibm1BackwardModel(tgtCorpusFilename, srcCorpusFilename, ibm1BackwardOutputFilenamePrefix, model1LearningInfo);
  ibm1BackwardModel.Train();
  cerr << "finished backward ibm1 training." << endl << endl;
  //  cin >> dummy;

  // initialize int-to-string src types map
  VocabDecoder srcTypes(srcVocabFilename);

  // initialize int-to-string tgt types map
  VocabDecoder tgtTypes(tgtVocabFilename);

  // initialize the loglinear model
  LearningInfo learningInfo;
  learningInfo.useMaxIterationsCount = true;
  learningInfo.useMinLikelihoodDiff = false;
  learningInfo.minLikelihoodDiff = 0.01;
  learningInfo.maxIterationsCount = 30;
  learningInfo.optimizationMethod.algorithm = OptUtils::STOCHASTIC_GRADIENT_DESCENT;
  learningInfo.optimizationMethod.miniBatchSize = 1;
  learningInfo.optimizationMethod.regularizer = Regularizer::L1;
  learningInfo.optimizationMethod.regularizationStrength = 0.01;
  learningInfo.samplesCount = 100;
  learningInfo.distATGivenS = Distribution::LOCAL;
  learningInfo.customDistribution = &hmmModel;
  learningInfo.unionAllCompatibleAlignments = true;
  learningInfo.srcVocabDecoder = &srcTypes;
  learningInfo.tgtVocabDecoder = &tgtTypes;
  learningInfo.ibm1ForwardLogProbs = &ibm1ForwardModel.params;
  learningInfo.ibm1BackwardLogProbs = &ibm1BackwardModel.params;
  LogLinearModel model(srcCorpusFilename, tgtCorpusFilename, outputFilenamePrefix, learningInfo);

  // train model parameters
  model.Train();

  // use the trained model to align the testset
  string outputAlignmentsFilename = outputFilenamePrefix + ".align";
  model.AlignTestSet(srcTestFilename, tgtTestFilename, outputAlignmentsFilename);
}
