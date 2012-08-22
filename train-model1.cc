#include "LearningInfo.h"
#include "FstUtils.h"
#include "StringUtils.h"
#include "IbmModel1.h"

using namespace fst;
using namespace std;

void ParseParameters(int argc, char **argv, string& srcCorpusFilename, string &tgtCorpusFilename, string &outputFilepathPrefix) {
  assert(argc == 4);
  srcCorpusFilename = argv[1];
  tgtCorpusFilename = argv[2];
  outputFilepathPrefix = argv[3];
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

  /*
  vector<LogWeight> alphas, betas;
  ShortestDistance(alignment, &alphas, false);
  ShortestDistance(alignment, &betas, true);
  int stateId = 0;
  for (vector<LogWeight>::const_iterator alphasIter = alphas.begin(); alphasIter != alphas.end(); alphasIter++) {
    cout << "alphas[" << stateId++ << "] = " << alphasIter->Value() << " = e^" << exp(-1.0 * alphasIter->Value()) << endl;
  }
  stateId = 0;
  for (vector<LogWeight>::const_iterator betasIter = betas.begin(); betasIter != betas.end(); betasIter++) {
    cout << "betas[" << stateId++ << "] = " << betasIter->Value() << " = e^" << exp(-1.0 * betasIter->Value()) << endl;
  }
  */
}

int main(int argc, char **argv) {
  //  Experimental();
  //  return 0;

  // parse arguments
  cout << "parsing arguments" << endl;
  string srcCorpusFilename, tgtCorpusFilename, outputFilenamePrefix;
  ParseParameters(argc, argv, srcCorpusFilename, tgtCorpusFilename, outputFilenamePrefix);

  // initialize the model
  IbmModel1 model(srcCorpusFilename, tgtCorpusFilename, outputFilenamePrefix);

  // train model parameters
  model.Train();

}
