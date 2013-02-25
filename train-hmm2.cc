#include "LearningInfo.h"
#include "FstUtils.h"
#include "StringUtils.h"
#include "HmmModel2.h"

using namespace fst;
using namespace std;
using namespace boost;

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
  // boost mpi initialization
  mpi::environment env(argc, argv);
  mpi::communicator world;

  // parse arguments
  if(world.rank() == 0) {
    cerr << "master" << world.rank() << ": parsing arguments...";
  }
  string textFilename, outputFilenamePrefix, goldLabelsFilename;
  ParseParameters(argc, argv, textFilename, outputFilenamePrefix, goldLabelsFilename);
  if(world.rank() == 0) {
    cerr << "done." << endl;
  }

  // randomize draws
  int seed = time(NULL);
  if(world.rank() == 0) {
    cerr << "master" << world.rank() << ": executing srand(" << seed << ")" << endl;
  }
  srand(seed);

  // configurations
  if(world.rank() == 0) {
    cerr << "master" << world.rank() << ": setting configurations..." << endl;
  }
  LearningInfo learningInfo;
  learningInfo.maxIterationsCount = 100;
  learningInfo.useMaxIterationsCount = true;
  learningInfo.useMinLikelihoodRelativeDiff = true;
  learningInfo.minLikelihoodRelativeDiff = 0.0001;
  learningInfo.debugLevel = DebugLevel::CORPUS;
  //  learningInfo.useEarlyStopping = true;
  learningInfo.mpiWorld = &world;
  learningInfo.persistParamsAfterEachIteration = false;
  learningInfo.persistFinalParams = false;
  learningInfo.smoothMultinomialParams = false;
  learningInfo.optimizationMethod.algorithm = OptAlgorithm::EXPECTATION_MAXIMIZATION;

  // initialize the model
  unsigned NUMBER_OF_LABELS = 45;
  HmmModel2 model(textFilename, outputFilenamePrefix, learningInfo, NUMBER_OF_LABELS);

  // train model parameters
  if(world.rank() == 0) {
    cerr << "master" << world.rank() << ": train the model..." << endl;
  }
  model.Train();
  if(world.rank() == 0) {
    cerr << "training finished!" << endl;
  }
  
  // we don't need the slaves anymore
  if(world.rank() > 0) {
    return 0;
  }

  // viterbi
  string labelsFilename = outputFilenamePrefix + ".labels";
  model.Label2(textFilename, labelsFilename);
  cerr << "automatic labels can be found at " << labelsFilename << endl;

  // compare to gold standard
  if(goldLabelsFilename != "") {
    cerr << "comparing to gold standard tagging..." << endl;
    double vi = model.ComputeVariationOfInformation(labelsFilename, goldLabelsFilename);
    cerr << "done. \nvariation of information = " << vi << endl;
    double manyToOne = model.ComputeManyToOne(labelsFilename, goldLabelsFilename);
    cerr << "many-to-one = " << manyToOne;
  }

}
