#include "LearningInfo.h"
#include "FstUtils.h"
#include "StringUtils.h"
#include "LatentCrfModel.h"

using namespace fst;
using namespace std;

typedef ProductArc<LogWeight, LogWeight> ProductLogArc;

void ParseParameters(int argc, char **argv, string &textFilename, string &outputFilenamePrefix) {
  assert(argc == 3);
  textFilename = argv[1];
  outputFilenamePrefix = argv[2];
}

int main(int argc, char **argv) {
  // parse arguments
  cout << "parsing arguments" << endl;
  string textFilename, outputFilenamePrefix;
  ParseParameters(argc, argv, textFilename, outputFilenamePrefix);

  // configurations
  LearningInfo learningInfo;
  // block coord 
  learningInfo.debugInfo = DebugLevel::CORPUS;
  learningInfo.optimizationMethod.algorithm = OptAlgorithm::BLOCK_COORD_DESCENT;
  learningInfo.useMaxIterationsCount = true;
  learningInfo.maxIterationsCount = 50;
  learningInfo.useMinLikelihoodDiff = true;
  learningInfo.minLikelihoodDiff = 10
  // lbfgs
  learningInfo.optimizationMethod.subOptMethod = new OptMethod();
  learningInfo.optimizationMethod.subOptMethod->regularizer = Regularizer::L1;
  learningInfo.optimizationMethod.subOptMethod->regularizationStrength = 0.1;
  learningInfo.optimizationMethod.subOptMethod->miniBatchSize = 0;
  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxIterations = 5;
  //  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.memoryBuffer = 500;
  //  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.precision = 0.0000000000001;
  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.l1 = (learningInfo.optimizationMethod.subOptMethod->regularizer == Regularizer::L1);

  // train the model
  LatentCrfModel& model = LatentCrfModel::GetInstance(textFilename, outputFilenamePrefix, learningInfo);
  model.Train();
  cerr << "training finished!" << endl;
  
  // find the most likely labels given the trianed model
  //  model.Label(textFilename, outputFilenamePrefix + ".labels");
  
  // compute some statistics on a test set
  string analysisFilename = outputFilenamePrefix + ".analysis";
  model.Analyze(textFilename, analysisFilename);
  cerr << "analysis can be found at " << analysisFilename << endl;
}
