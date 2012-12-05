#include "LearningInfo.h"
#include "FstUtils.h"
#include "StringUtils.h"
#include "AutoEncoder.h"

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

  // TODO

}
