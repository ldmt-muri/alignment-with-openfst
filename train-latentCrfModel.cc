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
  learningInfo.constraints.resize(52);
  learningInfo.constraints[0].SetConstraintOfType_yI_xIString(3, "neighbour");
  learningInfo.constraints[1].SetConstraintOfType_yI_xIString(3, "prison");
  learningInfo.constraints[2].SetConstraintOfType_yI_xIString(3, "sisters");
  learningInfo.constraints[3].SetConstraintOfType_yI_xIString(3, "house");
  learningInfo.constraints[4].SetConstraintOfType_yI_xIString(3, "event");
  learningInfo.constraints[5].SetConstraintOfType_yI_xIString(3, "man");
  learningInfo.constraints[6].SetConstraintOfType_yI_xIString(3, "woman");
  learningInfo.constraints[7].SetConstraintOfType_yI_xIString(4, "kill");
  learningInfo.constraints[8].SetConstraintOfType_yI_xIString(4, "killed");
  learningInfo.constraints[9].SetConstraintOfType_yI_xIString(4, "killing");
  learningInfo.constraints[10].SetConstraintOfType_yI_xIString(4, "commit");
  learningInfo.constraints[11].SetConstraintOfType_yI_xIString(4, "commited");
  learningInfo.constraints[12].SetConstraintOfType_yI_xIString(4, "commiting");
  learningInfo.constraints[13].SetConstraintOfType_yI_xIString(4, "take");
  learningInfo.constraints[14].SetConstraintOfType_yI_xIString(4, "took");
  learningInfo.constraints[15].SetConstraintOfType_yI_xIString(4, "taken");
  learningInfo.constraints[16].SetConstraintOfType_yI_xIString(4, "live");
  learningInfo.constraints[17].SetConstraintOfType_yI_xIString(4, "lived");
  learningInfo.constraints[18].SetConstraintOfType_yI_xIString(5, "simple");
  learningInfo.constraints[19].SetConstraintOfType_yI_xIString(5, "honourable");
  learningInfo.constraints[20].SetConstraintOfType_yI_xIString(5, "nice");
  learningInfo.constraints[21].SetConstraintOfType_yI_xIString(5, "strong");
  learningInfo.constraints[22].SetConstraintOfType_yI_xIString(6, "clearly");
  learningInfo.constraints[23].SetConstraintOfType_yI_xIString(6, "immediately");
  learningInfo.constraints[24].SetConstraintOfType_yI_xIString(6, "particularly");
  learningInfo.constraints[25].SetConstraintOfType_yIExclusive_xIString(7, "i");
  learningInfo.constraints[26].SetConstraintOfType_yIExclusive_xIString(7, "you");
  learningInfo.constraints[27].SetConstraintOfType_yIExclusive_xIString(7, "he");
  learningInfo.constraints[28].SetConstraintOfType_yIExclusive_xIString(7, "they");
  learningInfo.constraints[29].SetConstraintOfType_yIExclusive_xIString(8, "an");
  learningInfo.constraints[30].SetConstraintOfType_yIExclusive_xIString(8, "the");
  learningInfo.constraints[31].SetConstraintOfType_yIExclusive_xIString(9, "in");
  learningInfo.constraints[32].SetConstraintOfType_yIExclusive_xIString(9, "at");
  learningInfo.constraints[33].SetConstraintOfType_yIExclusive_xIString(9, "by");
  learningInfo.constraints[34].SetConstraintOfType_yIExclusive_xIString(10, "0");
  learningInfo.constraints[35].SetConstraintOfType_yIExclusive_xIString(10, "1");
  learningInfo.constraints[36].SetConstraintOfType_yIExclusive_xIString(10, "2");
  learningInfo.constraints[37].SetConstraintOfType_yIExclusive_xIString(10, "3");
  learningInfo.constraints[38].SetConstraintOfType_yIExclusive_xIString(10, "4");
  learningInfo.constraints[39].SetConstraintOfType_yIExclusive_xIString(10, "5");
  learningInfo.constraints[40].SetConstraintOfType_yIExclusive_xIString(10, "6");
  learningInfo.constraints[41].SetConstraintOfType_yIExclusive_xIString(10, "7");
  learningInfo.constraints[42].SetConstraintOfType_yIExclusive_xIString(10, "8");
  learningInfo.constraints[43].SetConstraintOfType_yIExclusive_xIString(10, "9");
  learningInfo.constraints[44].SetConstraintOfType_yIExclusive_xIString(11, "and");
  learningInfo.constraints[45].SetConstraintOfType_yIExclusive_xIString(11, "or");
  learningInfo.constraints[46].SetConstraintOfType_yIExclusive_xIString(13, ".");
  learningInfo.constraints[47].SetConstraintOfType_yIExclusive_xIString(13, ",");
  learningInfo.constraints[48].SetConstraintOfType_yIExclusive_xIString(13, ";");
  learningInfo.constraints[49].SetConstraintOfType_yIExclusive_xIString(13, "-");
  learningInfo.constraints[50].SetConstraintOfType_yIExclusive_xIString(13, "?");
  learningInfo.constraints[51].SetConstraintOfType_yIExclusive_xIString(13, "!");

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
