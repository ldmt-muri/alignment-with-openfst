#include "LearningInfo.h"
#include "FstUtils.h"
#include "StringUtils.h"
#include "HmmModel.h"

using namespace fst;
using namespace std;

void ParseParameters(int argc, char **argv, string& srcCorpusFilename, string &tgtCorpusFilename, string &srcTestSetFilename, string &tgtTestSetFilename, string &outputFilepathPrefix) {
  assert(argc == 6 || argc == 4);
  srcCorpusFilename = argv[1];
  tgtCorpusFilename = argv[2];
  if(argc == 6) {
    srcTestSetFilename = argv[3];
    tgtTestSetFilename = argv[4];
    outputFilepathPrefix = argv[5];
  } else if(argc == 4) {
    outputFilepathPrefix = argv[3];
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
  string srcCorpusFilename, tgtCorpusFilename, srcTestSetFilename, tgtTestSetFilename, outputFilenamePrefix;
  ParseParameters(argc, argv, srcCorpusFilename, tgtCorpusFilename, srcTestSetFilename, tgtTestSetFilename, outputFilenamePrefix);

  // specify stopping criteria
  LearningInfo learningInfo;
  learningInfo.maxIterationsCount = 1;
  learningInfo.useMaxIterationsCount = true;
  learningInfo.minLikelihoodDiff = 10.0;
  learningInfo.useMinLikelihoodDiff = true;
  learningInfo.debugLevel = DebugLevel::CORPUS;
  //  learningInfo.useEarlyStopping = true;
  learningInfo.mpiWorld = &world;
  learningInfo.persistParamsAfterEachIteration = true;

  // initialize the model
  HmmModel model(srcCorpusFilename, tgtCorpusFilename, outputFilenamePrefix, learningInfo);

  // train model parameters
  model.Train();

  // align the test set
  if(srcTestSetFilename.size() > 0) {
    string outputAlignmentsFilename = outputFilenamePrefix + ".test.align";
    model.AlignTestSet(srcTestSetFilename, tgtTestSetFilename, outputAlignmentsFilename);
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
