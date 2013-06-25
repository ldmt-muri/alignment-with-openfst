#include "LearningInfo.h"
#include "FstUtils.h"
#include "StringUtils.h"
#include "IbmModel1.h"

using namespace fst;
using namespace std;
using namespace boost;

typedef ProductArc<LogWeight, LogWeight> ProductLogArc;

void ParseParameters(int argc, char **argv, string& bitextFilename, string &outputFilepathPrefix) {
  assert(argc == 3);
  bitextFilename = argv[1];
  outputFilepathPrefix = argv[2];
}

void Experimental() {
  cout << "nLog(1.0) = " << FstUtils::nLog(1.0) << endl;
  cout << "nLog(0.5) = " << FstUtils::nLog(0.5) << endl;
  cout << "nLog(0.25) = " << FstUtils::nLog(0.25) << endl;
  /*
  ProductWeight<LogWeight, LogWeight> x;
  VectorFst< ProductArc<LogWeight, LogWeight> > fst1;
  int state0 = fst1.AddState();
  int state1 = fst1.AddState();
  int state2 = fst1.AddState();
  int state3 = fst1.AddState();
  fst1.SetStart(state0);
  fst1.SetFinal(state3, x.One());
  fst1.AddArc(state0, ProductLogArc(11, 11, FstUtils::EncodePair(1,0), state1));
  fst1.AddArc(state1, ProductLogArc(22, 22, FstUtils::EncodePair(2,0), state2));
  fst1.AddArc(state2, ProductLogArc(22, 22, FstUtils::EncodePair(3,0), state3));

  VectorFst< ProductArc<LogWeight, LogWeight> > fst2;
  state0 = fst2.AddState();
  state1 = fst2.AddState();
  state2 = fst2.AddState();
  fst2.SetStart(state0);
  fst2.SetFinal(state1, x.One());
  fst2.SetFinal(state2, x.One());
  fst2.AddArc(state0, ProductLogArc(11, 111, FstUtils::EncodePair(0,0), state1));
  fst2.AddArc(state0, ProductLogArc(11, 222, FstUtils::EncodePair(0,0), state2));
  fst2.AddArc(state0, ProductLogArc(22, 111, FstUtils::EncodePair(0,0), state1));
  fst2.AddArc(state0, ProductLogArc(22, 222, FstUtils::EncodePair(0,0), state2));
  fst2.AddArc(state1, ProductLogArc(11, 111, FstUtils::EncodePair(0,0), state1));
  fst2.AddArc(state1, ProductLogArc(11, 222, FstUtils::EncodePair(0,0), state2));
  fst2.AddArc(state1, ProductLogArc(22, 111, FstUtils::EncodePair(0,0), state1));
  fst2.AddArc(state1, ProductLogArc(22, 222, FstUtils::EncodePair(0,0), state2));
  fst2.AddArc(state2, ProductLogArc(11, 111, FstUtils::EncodePair(0,0), state1));
  fst2.AddArc(state2, ProductLogArc(11, 222, FstUtils::EncodePair(0,0), state2));
  fst2.AddArc(state2, ProductLogArc(22, 111, FstUtils::EncodePair(0,0), state1));
  fst2.AddArc(state2, ProductLogArc(22, 222, FstUtils::EncodePair(0,0), state2));

  VectorFst< ProductArc<LogWeight, LogWeight> > fst3;
  state0 = fst3.AddState();
  fst3.SetStart(state0);
  fst3.SetFinal(state0, x.One());
  fst3.AddArc(state0, ProductLogArc(111, 111, FstUtils::EncodePair(0,1), state0));
  fst3.AddArc(state0, ProductLogArc(222, 222, FstUtils::EncodePair(0,2), state0));
  fst3.AddArc(state0, ProductLogArc(111, 111, FstUtils::EncodePair(0,3), state0));

  VectorFst< ProductArc<LogWeight, LogWeight> > temp, final;
  Compose(fst1, fst2, &temp);
  Compose(temp, fst3, &final);

  final.Write("example/test.bin");

  VectorFst< ProductArc<LogWeight, LogWeight> > temp2;
  temp2.Read("example/test.bin");

  //  FstUtils::PrintFstSummary(final);
  //  FstUtils::PrintFstSummary(temp2);

  cout << "===================FST1=============" << endl;
  //  FstUtils::PrintFstSummary(fst1);
  cout << "===================FST2=============" << endl;
  //  FstUtils::PrintFstSummary(fst2);
  cout << "===================FST3=============" << endl;
  //  FstUtils::PrintFstSummary(fst3);
  cout << "===================TEMP=============" << endl;
  //  FstUtils::PrintFstSummary(temp);
  cout << "===================FINAL=============" << endl;  
  //  FstUtils::PrintFstSummary(final);
  */
}

int main(int argc, char **argv) {
  // boost mpi initialization
  mpi::environment env(argc, argv);
  mpi::communicator world;

  //  Experimental();
  //  return 0;

  // parse arguments
  cout << "parsing arguments" << endl;
  string bitextFilename, outputFilenamePrefix;
  ParseParameters(argc, argv, bitextFilename, outputFilenamePrefix);

  // specify stopping criteria
  LearningInfo learningInfo;
  learningInfo.maxIterationsCount = 100;
  learningInfo.useMaxIterationsCount = true;
  //  learningInfo.useEarlyStopping = true;
  learningInfo.minLikelihoodDiff = 100.0;
  learningInfo.useMinLikelihoodDiff = false;
  learningInfo.minLikelihoodRelativeDiff = 0.01;
  learningInfo.useMinLikelihoodRelativeDiff = true;
  learningInfo.mpiWorld = &world;
  
  // initialize the model
  IbmModel1 model(bitextFilename, outputFilenamePrefix, learningInfo);

  // train model parameters
  model.Train();
  model.Align();

}
