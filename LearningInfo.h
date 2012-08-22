#ifndef _LEARNING_INFO_H_
#define _LEARNING_INFO_H_

#include <vector>

using namespace std;

class LearningInfo {
 public:
  LearningInfo() {
    useMaxIterationsCount = false;
    useMinLikelihoodDiff = false;
    iterationsCount = 0;
  }

  // criteria 1
  bool useMaxIterationsCount;
  int maxIterationsCount;

  // criteria 2
  bool useMinLikelihoodDiff;
  float minLikelihoodDiff;

  // output
  int iterationsCount;
  vector< float > logLikelihood;
};

#endif
