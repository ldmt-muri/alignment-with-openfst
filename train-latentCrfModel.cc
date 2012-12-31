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
  learningInfo.debugLevel = DebugLevel::MINI_BATCH;
  learningInfo.optimizationMethod.algorithm = OptAlgorithm::BLOCK_COORD_DESCENT;
  learningInfo.useMaxIterationsCount = true;
  learningInfo.maxIterationsCount = 50;
  learningInfo.useMinLikelihoodDiff = true;
  learningInfo.minLikelihoodDiff = 10;
  // lbfgs
  learningInfo.optimizationMethod.subOptMethod = new OptMethod();
  learningInfo.optimizationMethod.subOptMethod->regularizer = Regularizer::L1;
  learningInfo.optimizationMethod.subOptMethod->regularizationStrength = 0.1;
  learningInfo.optimizationMethod.subOptMethod->miniBatchSize = 0;
  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxIterations = 5;
  //  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.memoryBuffer = 500;
  //  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.precision = 0.0000000000001;
  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.l1 = (learningInfo.optimizationMethod.subOptMethod->regularizer == Regularizer::L1);

  // add constraints
  learningInfo.constraints.resize(6);
  learningInfo.constraints[0].SetConstraintOfType_yI_xIString(7, "I");
  learningInfo.constraints[1].SetConstraintOfType_yI_xIString(8, "a");
  learningInfo.constraints[2].SetConstraintOfType_yI_xIString(9, "in");
  learningInfo.constraints[3].SetConstraintOfType_yI_xIString(10, "1");
  learningInfo.constraints[4].SetConstraintOfType_yI_xIString(11, "and");
  learningInfo.constraints[5].SetConstraintOfType_yI_xIString(13, ".");
  /*
  learningInfo.constraints[7]["I"] = true;
  learningInfo.constraints[7]["you"] = true;
  learningInfo.constraints[8]["a"] = true;
  learningInfo.constraints[8]["an"] = true;
  learningInfo.constraints[8]["the"] = true;
  learningInfo.constraints[9]["in"] = true;
  learningInfo.constraints[9]["on"] = true;
  learningInfo.constraints[9]["at"] = true;
  learningInfo.constraints[9]["by"] = true;
  learningInfo.constraints[9]["from"] = true;
  learningInfo.constraints[9]["after"] = true;
  learningInfo.constraints[10]["1"] = true;
  learningInfo.constraints[10]["2"] = true;
  learningInfo.constraints[10]["3"] = true;
  learningInfo.constraints[10]["4"] = true;
  learningInfo.constraints[10]["5"] = true;
  learningInfo.constraints[10]["6"] = true;
  learningInfo.constraints[11]["and"] = true;
  learningInfo.constraints[11]["or"] = true;
  learningInfo.constraints[13]["."] = true;
  learningInfo.constraints[13][","] = true;
  learningInfo.constraints[13]["?"] = true;
  */

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
