#include "LearningInfo.h"
#include "FstUtils.h"
#include "StringUtils.h"
#include "HmmAligner.h"

using namespace fst;
using namespace std;
using namespace boost;

void ParseParameters(int argc, char **argv, string& bitextFilename, string &testBitextFilename, string &outputFilepathPrefix) {
  assert(argc == 4 || argc == 3);
  bitextFilename = argv[1];
  if(argc == 4) {
    testBitextFilename = argv[2];
    outputFilepathPrefix = argv[3];
  } else if(argc == 3) {
    outputFilepathPrefix = argv[2];
  } else {
    cerr << "invalid arguments" << endl;
  }
}

int main(int argc, char **argv) {
  // boost mpi initialization
  mpi::environment env(argc, argv);
  mpi::communicator world;

  // parse arguments
  cout << "parsing arguments" << endl;
  string bitextFilename, testBitextFilename, outputFilenamePrefix;
  ParseParameters(argc, argv, bitextFilename, testBitextFilename, outputFilenamePrefix);

  // specify stopping criteria
  LearningInfo learningInfo;
  learningInfo.maxIterationsCount = 1000;
  learningInfo.useMaxIterationsCount = true;
  learningInfo.minLikelihoodDiff = 100.0;
  learningInfo.useMinLikelihoodDiff = false;
  learningInfo.minLikelihoodRelativeDiff = 0.01;
  learningInfo.useMinLikelihoodRelativeDiff = true;
  learningInfo.debugLevel = DebugLevel::CORPUS;
  learningInfo.useEarlyStopping = false;
  learningInfo.mpiWorld = &world;
  learningInfo.persistParamsAfterNIteration = 10;
  learningInfo.persistFinalParams = false;

  // initialize the model
  HmmAligner model(bitextFilename, outputFilenamePrefix, learningInfo);

  // train model parameters
  model.Train();

  // align the test set
  if(testBitextFilename.size() > 0) {
    string outputAlignmentsFilename = outputFilenamePrefix + ".test.align";
    model.AlignTestSet(testBitextFilename, outputAlignmentsFilename);
  } else {
    string outputAlignmentsFilename = outputFilenamePrefix + ".train.align";
    model.Align(outputAlignmentsFilename);
  }

  /*
  // sample a few translations
  vector<int> srcTokens;
  srcTokens.push_back(1);
  srcTokens.push_back(3);
  srcTokens.push_back(2);
  srcTokens.push_back(4);
  for(int i = 0; i < 500; i++) {
    vector<int> tgtTokens, alignments;
    double hmmLogProb;
    model.SampleATGivenS(srcTokens, 3, tgtTokens, alignments, hmmLogProb);
    cerr << endl << "translation: ";
    for(int j = 0; j < tgtTokens.size(); j++) {
      cerr << tgtTokens[j] << "(" << srcTokens[alignments[j]] << ") ";
    }
    cerr << hmmLogProb << "(" << FstUtils::nExp(hmmLogProb) << ")";
  }
  */
}
