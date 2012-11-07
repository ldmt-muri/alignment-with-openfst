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
  learningInfo.minLikelihoodDiff = 1.0;
  learningInfo.useMinLikelihoodDiff = true;
  //  learningInfo.useEarlyStopping = true;

  // initialize the model
  HmmModel model(srcCorpusFilename, tgtCorpusFilename, outputFilenamePrefix, learningInfo);

  // train model parameters
  model.Train();

  // sample a few translations
  vector<int> srcTokens;
  srcTokens.push_back(1);
  srcTokens.push_back(3);
  srcTokens.push_back(2);
  srcTokens.push_back(4);
  for(int i = 0; i < 500; i++) {
    vector<int> tgtTokens, alignments;
    double hmmLogProb;
    model.SampleAT(srcTokens, 3, tgtTokens, alignments, hmmLogProb);
    cerr << endl << "translation: ";
    for(int j = 0; j < tgtTokens.size(); j++) {
      cerr << tgtTokens[j] << "(" << srcTokens[alignments[j]] << ") ";
    }
    cerr << hmmLogProb << "(" << FstUtils::nExp(hmmLogProb) << ")";
  }
}
