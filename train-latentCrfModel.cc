#include <fenv.h>
#include "LearningInfo.h"
#include "FstUtils.h"
#include "StringUtils.h"
#include "LatentCrfModel.h"

using namespace fst;
using namespace std;

typedef ProductArc<LogWeight, LogWeight> ProductLogArc;

void ParseParameters(int argc, char **argv, string &textFilename, string &outputFilenamePrefix, string &goldLabelsFilename) {
  assert(argc >= 3);
  textFilename = argv[1];
  outputFilenamePrefix = argv[2];
  if(argc >= 4) {
    goldLabelsFilename = argv[3];
  } else {
    goldLabelsFilename = "";
  }
}

int main(int argc, char **argv) {

  //  feenableexcept(FE_INVALID | FE_OVERFLOW | FE_DIVBYZERO);

  // parse arguments
  cerr << "parsing arguments...";
  string textFilename, outputFilenamePrefix, goldLabelsFilename;
  ParseParameters(argc, argv, textFilename, outputFilenamePrefix, goldLabelsFilename);
  cerr << "done." << endl;

  // randomize draws
  int seed = time(NULL);
  cerr << "executing srand(" << seed << ")" << endl;
  srand(seed);

  // configurations
  cerr << "setting configurations...";
  LearningInfo learningInfo;
  // general 
  learningInfo.debugLevel = DebugLevel::MINI_BATCH;
  learningInfo.useMaxIterationsCount = true;
  learningInfo.maxIterationsCount = 50;
  //  learningInfo.useMinLikelihoodDiff = true;
  //  learningInfo.minLikelihoodDiff = 10;
  learningInfo.useMinLikelihoodRelativeDiff = true;
  learningInfo.minLikelihoodRelativeDiff = 0.001;
  learningInfo.useSparseVectors = true;
  // block coordinate descent
  learningInfo.optimizationMethod.algorithm = OptAlgorithm::BLOCK_COORD_DESCENT;
  // lbfgs
  learningInfo.optimizationMethod.subOptMethod = new OptMethod();
  learningInfo.optimizationMethod.subOptMethod->algorithm = OptAlgorithm::LBFGS;
  learningInfo.optimizationMethod.subOptMethod->regularizer = Regularizer::NONE;
  learningInfo.optimizationMethod.subOptMethod->regularizationStrength = 0.1;
  learningInfo.optimizationMethod.subOptMethod->miniBatchSize = 0;
  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxIterations = 5;
  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxEvalsPerIteration = 5;
  //  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.memoryBuffer = 50;
  //  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.precision = 0.00000000000000000000000001;
  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.l1 = (learningInfo.optimizationMethod.subOptMethod->regularizer == Regularizer::L1);
  learningInfo.optimizationMethod.subOptMethod->moveAwayPenalty = 0.0;

  // add constraints
  learningInfo.constraints.clear();
  /*
  Constraint constraint;
  constraint.SetConstraintOfType_yI_xIString(3, "neighbour");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(3, "prison");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(3, "sisters");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(3, "house");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(3, "event");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(3, "man");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(3, "woman");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(4, "kill");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(4, "killed");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(4, "killing");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(4, "commit");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(4, "commited");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(4, "commiting");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(4, "take");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(4, "took");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(4, "taken");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(4, "live");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(4, "lived");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(5, "simple");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(5, "honourable");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(5, "nice");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(5, "strong");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(6, "clearly");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(6, "immediately");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yI_xIString(6, "particularly");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(7, "i");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(7, "you");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(7, "he");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(7, "they");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(8, "an");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(8, "the");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(9, "in");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(9, "at");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(9, "by");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(10, "0");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(10, "1");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(10, "2");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(10, "3");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(10, "4");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(10, "5");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(10, "6");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(10, "7");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(10, "8");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(10, "9");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(11, "and");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(11, "or");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(13, ".");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(13, ",");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(13, ";");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(13, "-");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(13, "?");
  learningInfo.constraints.push_back(constraint);
  constraint.SetConstraintOfType_yIExclusive_xIString(13, "!");
  learningInfo.constraints.push_back(constraint);
  */
  cerr << "done." << endl;

  // train the model
  cerr << "train the model..." << endl;
  LatentCrfModel& model = LatentCrfModel::GetInstance(textFilename, outputFilenamePrefix, learningInfo);

  model.Train();
  cerr << "training finished!" << endl;
    
  // compute some statistics on a test set
  cerr << "analyze the data using the trained model..." << endl;
  string analysisFilename = outputFilenamePrefix + ".analysis";
  model.Analyze(textFilename, analysisFilename);
  cerr << "analysis can be found at " << analysisFilename << endl;
  
  // viterbi
  string labelsFilename = outputFilenamePrefix + ".labels";
  model.Label(textFilename, labelsFilename);
  cerr << "automatic labels can be found at " << labelsFilename << endl;

  // compare to gold standard
  if(goldLabelsFilename != "") {
    cerr << "comparing to gold standard tagging..." << endl;
    double vi = model.ComputeVariationOfInformation(labelsFilename, goldLabelsFilename);
    cerr << "done. variation of information = " << vi << endl;
  }
}
