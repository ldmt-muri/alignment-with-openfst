#include "LearningInfo.h"
#include "FstUtils.h"
#include "StringUtils.h"
#include "HmmModel.h"

using namespace fst;
using namespace std;

void ParseParameters(int argc, char **argv, string& srcCorpusFilename, string &tgtCorpusFilename, string &outputFilepathPrefix) {
  assert(argc == 4);
  srcCorpusFilename = argv[1];
  tgtCorpusFilename = argv[2];
  outputFilepathPrefix = argv[3];
}

int main(int argc, char **argv) {
  // parse arguments
  cout << "parsing arguments" << endl;
  string srcCorpusFilename, tgtCorpusFilename, outputFilenamePrefix;
  ParseParameters(argc, argv, srcCorpusFilename, tgtCorpusFilename, outputFilenamePrefix);

  // specify stopping criteria
  LearningInfo learningInfo;
  learningInfo.maxIterationsCount = 100;
  learningInfo.useMaxIterationsCount = true;
  learningInfo.minLikelihoodDiff = 100.0;
  learningInfo.useMinLikelihoodDiff = true;
  //  learningInfo.useEarlyStopping = true;

  // initialize the model
  HmmModel model(srcCorpusFilename, tgtCorpusFilename, outputFilenamePrefix, learningInfo);

  // train model parameters
  model.Train();

}
