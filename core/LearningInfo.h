#ifndef _LEARNING_INFO_H_
#define _LEARNING_INFO_H_

#include <vector>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <map>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

#include "BasicTypes.h"
#include "../alignment/IAlignmentSampler.h"

using namespace std;

class LearningInfo {
 public:

 LearningInfo(boost::mpi::communicator *mpiWorld, string outputFilenamePrefix) : 
  mpiWorld(mpiWorld), 
  outputFilenamePrefix(outputFilenamePrefix) {
    checkGradient = false;
    minTokenFrequency = 100;
    featureGaussianMeanFilename = "";
    SetSharedMemorySegment(mpiWorld->rank() == 0);
    useMaxIterationsCount = false;
    useMinLikelihoodDiff = false;
    useEarlyStopping = false;
    trainToDevDataSize = 10;
    iterationsCount = 0;
    minLikelihoodDiff = 1.0;
    minIterationsCount = 5;
    maxIterationsCount = 10;
    saveAlignmentFstsOnDisk = false;
    neighborhood = DiscriminativeLexicon::ALL;
    distATGivenS = Distribution::TRUE;
    customDistribution = 0;
    neighborhoodMinIbm1FwdScore = 0.001;
    neighborhoodMinIbm1BckScore = 0.001;
    neighborhoodMinCoocc = 3;
    debugLevel = 1;
    useSparseVectors = true;
    persistParamsAfterNIteration = 100;
    persistFinalParams = true;
    retryLbfgsOnRoundingErrors = true;
    supervisedTraining = false;
    babySteps = false;
    inductive = false;
    unspecified = 0;
    unspecified2 = 0;
    firstKExamplesToLabel = 0;
    invokeCallbackFunctionEveryKIterations = 10;
    endOfKIterationsCallbackFunction = 0;
    nSentsPerDot = 1;
    emIterationsCount = 1;
    thetaOptMethod = 0;
    initializeThetasWithKleinManning = true;
    initializeThetasWithModel1 = false;
    initializeThetasWithGaussian = false;
    initializeThetasWithUniform = false;
    initializeLambdasWithZero = true;
    initializeLambdasWithOne = false;
    initializeLambdasWithGaussian = false;
    allowNullAlignments = true;
    maxSequenceLength = 0;
    hiddenSequenceIsMarkovian = true;
    cacheActiveFeatures = false;
    multinomialSymmetricDirichletAlpha = 1.0;
    variationalInferenceOfMultinomials = false;
    testWithCrfOnly = false;
    oneBasedConllFieldIdReconstructed = 2;
    oneBasedConllFieldIdConditioned = 2;
    optimizeLambdasFirst = false;
    firePrecomputedFeaturesForXIM2 = firePrecomputedFeaturesForXIM1 = firePrecomputedFeaturesForXIP1 = firePrecomputedFeaturesForXIP2 = false;
    firePrecomputedFeaturesForXI = true;
    tagDictFilename="";
    preventSelfAlignments=false;
    // this is specific for dependnecy parsing
    generateChildAndDirection=false;
    generateChildConditionalOnDirection = false;
    hackK=0;
  }
  
  bool IsModelConverged() {
    assert(useMaxIterationsCount || useMinLikelihoodDiff || useEarlyStopping || useMinLikelihoodRelativeDiff);
    // logging
    cerr << "CONVERGENCE CHECK" << endl;
    cerr << "minIterationsCount=" << minIterationsCount << endl;
    cerr << "useMaxIterationsCount=" << useMaxIterationsCount << endl;
    cerr << "maxIterationsCount=" << maxIterationsCount << endl;
    cerr << "useMinLikelihoodDiff=" << useMinLikelihoodDiff << endl;
    cerr << "minLikelihoodDiff=" << minLikelihoodDiff << endl;
    cerr << "useMinLikelihoodRelativeDiff=" << useMinLikelihoodRelativeDiff << endl;
    cerr << "minLikelihoodRelativeDiff=" << minLikelihoodRelativeDiff << endl;
    cerr << "useEarlyStopping=" << useEarlyStopping << endl;
    cerr << endl;
    cerr << "iterationsCount=" << iterationsCount << endl;
    cerr << "ll[t  ] = " << logLikelihood[iterationsCount-1] << endl;
    if(useEarlyStopping) {
      cerr << "devll[t  ]=" << validationLogLikelihood[iterationsCount-1] << endl;
    }
    assert(minIterationsCount >= 3);
    assert(logLikelihood.size() >= 1);
    if(iterationsCount < (unsigned) minIterationsCount) {
      cerr << "min iterations count not met ==> NOT CONVERGED" << endl;
      return false;
    }
    assert( logLikelihood[iterationsCount-1] != 0.0 ); 
    assert( logLikelihood[iterationsCount-2] != 0.0 );
    assert( logLikelihood[iterationsCount-3] != 0.0 );
    cerr << "ll[t-1] = " << logLikelihood[iterationsCount-2] << endl;
    if(useEarlyStopping) {
      cerr << "devll[t-1] = " << validationLogLikelihood[iterationsCount-2] << endl;
    }
    cerr << "(ll[t  ]-ll[t-1])             =" << logLikelihood[iterationsCount-1] - logLikelihood[iterationsCount-2] << endl;
    cerr << "(ll[t  ]-ll[t-1])/abs(ll[t-1])=" << (logLikelihood[iterationsCount-1] - logLikelihood[iterationsCount-2])/ fabs(logLikelihood[iterationsCount-2]) << endl;
    cerr << "(ll[t-1]-ll[t-2])             =" << logLikelihood[iterationsCount-2] - logLikelihood[iterationsCount-3] << endl;
    cerr << "(ll[t-1]-ll[t-2])/abs(ll[t-2])=" << (logLikelihood[iterationsCount-2] - logLikelihood[iterationsCount-3])/ fabs(logLikelihood[iterationsCount-3]) << endl;
    if(useEarlyStopping) {
      cerr << "(devll[t  ]-devll[t-1])=" << validationLogLikelihood[iterationsCount-1] - validationLogLikelihood[iterationsCount-2];
      cerr << "(devll[t-1]-devll[t-2])=" << validationLogLikelihood[iterationsCount-2] - validationLogLikelihood[iterationsCount-3];
    }
    cerr << endl;
    
    // check for convergnece conditions
    if(useMaxIterationsCount && 
       (unsigned)maxIterationsCount <= iterationsCount) {
      cerr << "maxIterationsCount is met ==> CONVERGED" << endl;
      return true;
    } 
    if(useMinLikelihoodDiff && 
       minLikelihoodDiff > logLikelihood[iterationsCount-1] - logLikelihood[iterationsCount-2] &&
       minLikelihoodDiff > logLikelihood[iterationsCount-2] - logLikelihood[iterationsCount-3]) {
      cerr << "loglikelihood improvement was less than the minimum allowed for two consecutive iterations ==> CONVERGED" << endl;
      return true;
    } 
    if(useEarlyStopping &&
       0 > validationLogLikelihood[iterationsCount-1] - validationLogLikelihood[iterationsCount-2] &&
       0 > validationLogLikelihood[iterationsCount-2] - validationLogLikelihood[iterationsCount-3]) {
      cerr << "validation loglikelihood didn't improve for two consecutive iterations ==> CONVERGED" << endl;
      return true;
    }
    if(useMinLikelihoodRelativeDiff && 
       minLikelihoodRelativeDiff > (logLikelihood[iterationsCount-1] - logLikelihood[iterationsCount-2]) / fabs(logLikelihood[iterationsCount-2]) &&
       minLikelihoodRelativeDiff > (logLikelihood[iterationsCount-2] - logLikelihood[iterationsCount-3]) / fabs(logLikelihood[iterationsCount-3])) {
      cerr << "loglikelihood relative improvement was less than the minimum allowed for two consecutive iterations ==> CONVERGED" << endl;
      return true;
    }
    // none of the convergence conditions apply!
    cerr << "none of the conditions apply ==> NOT CONVERGED" << endl;
    return false;
  }

  int GetBestIterationNumber() {
    assert(iterationsCount > 0);
    int bestIteration = 0;
    if(useMinLikelihoodRelativeDiff || useMinLikelihoodDiff) {
      // the best iteration is the one with the highest regularized likelihood
      for(unsigned iteration = 1; iteration < logLikelihood.size(); iteration++) {
        if(logLikelihood[iteration] > logLikelihood[bestIteration]) {
          bestIteration = iteration;
        }
      }
    } else if(useEarlyStopping) {
      assert(validationLogLikelihood.size() > 0);
      // the best iteration is the one with the highest unregularized dev set likelihood
      for(unsigned iteration = 1; iteration < validationLogLikelihood.size(); iteration++) {
        if(validationLogLikelihood[iteration] > validationLogLikelihood[bestIteration]) {
          bestIteration = iteration;
        }
      }
    } else if(useMaxIterationsCount) {
      bestIteration = iterationsCount - 1;
    } else {
      // don't know how to determine best iteration?!
      assert(false);
    }
    return bestIteration;
  }

  // should be called only once when the shared memory is no longer needed
  void ClearSharedMemorySegment() {
    if(sharedMemorySegment != 0 && mpiWorld->rank() == 0) {
      cerr << "deleting shared memory" << endl;
      delete sharedMemorySegment;
      sharedMemorySegment = 0;
    }
  }
  
  // should be called only once in the constructor
  void SetSharedMemorySegment(bool create) {
    size_t segmentSize = 256 * 1024; // in GBs
    segmentSize *= 1024 * 1024;
    string SEGMENT_NAME = outputFilenamePrefix + ".segment";
    using namespace boost::interprocess;
    // Shared memory front-end that is able to construct objects
    // associated with a c-string. Erase previous shared memory with the name
    // to be used and create the memory segment at the specified address and initialize resources
    if(create) {
      cerr << "remove any shared memory object with the same name '" << SEGMENT_NAME << "'...";
      shared_memory_object::remove(SEGMENT_NAME.c_str());
      cerr << "done" << endl;
      
      // create or open the shared memory segments
      cerr << "requesting " << segmentSize << " bytes of managed shared memory for segment " << SEGMENT_NAME << "...";
      sharedMemorySegment = new managed_shared_memory(open_or_create, SEGMENT_NAME.c_str(), segmentSize);
      assert(sharedMemorySegment);
      cerr << "request granted." << endl;
      // sync with slaves
      bool dummy = true;
      boost::mpi::broadcast<bool>(*mpiWorld, dummy, 0);
    } else {
      // sync with master
      bool dummy = false;
      boost::mpi::broadcast<bool>(*mpiWorld, dummy, 0);
      sharedMemorySegment = new managed_shared_memory(open_only, SEGMENT_NAME.c_str());      
    }
    assert(sharedMemorySegment->get_size() == segmentSize);
  }
  
  // you can't converge before this many iterations no matter what
  int minIterationsCount;

  // criteria 1
  bool useMaxIterationsCount;
  int maxIterationsCount;
  
  // criteria 2
  bool useMinLikelihoodDiff;
  float minLikelihoodDiff;

  // criteria 3
  // by early stopping, i mean assume convergence as soon as the likelihood of a validation set cease to increase
  bool useEarlyStopping;
  int trainToDevDataSize;

  // criteria 4
  bool useMinLikelihoodRelativeDiff;
  float minLikelihoodRelativeDiff;

  // optimization method
  OptMethod optimizationMethod;

  // discriminative lexicon
  DiscriminativeLexicon::DiscriminativeLexicon neighborhood;
  float neighborhoodMinIbm1FwdScore;
  float neighborhoodMinIbm1BckScore;
  float neighborhoodMinCoocc;

  // save alignment FSTs on disk
  bool saveAlignmentFstsOnDisk;

  // number of samples used to approximate the posterior expectations
  //int samplesCount;

  // output
  unsigned iterationsCount;
  vector< float > logLikelihood;
  vector< float > validationLogLikelihood;  

  // max lbfgs iterations count for supervised training
  int supervisedMaxLbfgsIterCount;

  // distribution used to model p(a,T|S)
  Distribution::Distribution distATGivenS;
  IAlignmentSampler *customDistribution;

  // when using a proposal distribution for p(a,T|S), would you like to union the alignments of p(a|T,S) as well?
  bool unionAllCompatibleAlignments;

  // ibm 1 forward log probs
  // [srcToken][tgtToken]
  std::map<int, std::map<int, float> > *ibm1ForwardLogProbs;

  // ibm 1 backward log probs
  // [tgtToken][srcToken]
  std::map<int, std::map<int, float> > *ibm1BackwardLogProbs;

  // list of feature templates to be fired
  std::vector<FeatureTemplate> featureTemplates;
  bool firePrecomputedFeaturesForXIM2,
    firePrecomputedFeaturesForXIM1,
    firePrecomputedFeaturesForXI,
    firePrecomputedFeaturesForXIP1,
    firePrecomputedFeaturesForXIP2;
 
  // 0 = no debug info. 
  // 1 = corpus level debug info.
  // 2 = mini-batch level debug info.
  // 3 = sentence level debug info. 
  // 4 = token level debug info.
  unsigned debugLevel;

  /*
  // this field can be used to communicate to the underlying model that certain combinations are required/forbidden
  std::vector<Constraint> constraints;
  */

  // do we use cdec's FastSparseVector when applicable?
  bool useSparseVectors;
  
  // frequency of persisting the params
  unsigned persistParamsAfterNIteration;

  // persist the parameters after training?
  bool persistFinalParams;

  // boost mpi communicator
  boost::mpi::communicator *mpiWorld;

  // if lbfgs returns a rounding error. should we retry?
  bool retryLbfgsOnRoundingErrors;

  bool supervisedTraining;
  
  bool babySteps;

  bool inductive;

  int unspecified, unspecified2;

  unsigned firstKExamplesToLabel;

  unsigned invokeCallbackFunctionEveryKIterations;

  void (*endOfKIterationsCallbackFunction)();
  
  //bool smoothMultinomialParams;

  unsigned nSentsPerDot;

  unsigned emIterationsCount;

  OptMethod *thetaOptMethod;
  
  bool initializeThetasWithKleinManning;
  bool initializeThetasWithModel1;
  bool initializeThetasWithGaussian;
  bool initializeThetasWithUniform;

  bool initializeLambdasWithGaussian;  
  bool initializeLambdasWithZero;
  bool initializeLambdasWithOne;

  bool allowNullAlignments;

  unsigned maxSequenceLength;
  
  bool hiddenSequenceIsMarkovian;

  bool cacheActiveFeatures;
  
  // this makes the optimization problem convex
  bool fixPosteriorExpectationsAccordingToPZGivenXWhileOptimizingLambdas;
  
  double multinomialSymmetricDirichletAlpha;

  bool variationalInferenceOfMultinomials;
  
  bool testWithCrfOnly;

  int oneBasedConllFieldIdReconstructed;
  int oneBasedConllFieldIdConditioned;
  
  bool optimizeLambdasFirst;

  // train models for the reverse corpus direction
  bool reverse;

  string featureGaussianMeanFilename;

  // shared memory segment to efficiently share objects across processes
  boost::interprocess::managed_shared_memory *sharedMemorySegment;

  // the filenames specifying output of other word aligners for this dataset
  vector< string > otherAlignersOutputFilenames;
  
  // phrase list filenames
  vector< string > phraseListFilenames;

  // the id of the sentence pair currently being processed
  int currentSentId;

  string tgtWordClassesFilename;

  string outputFilenamePrefix;

  string goldFilename;
 
  string tagDictFilename;

  bool preventSelfAlignments;

  bool generateChildAndDirection, generateChildConditionalOnDirection;
  
  // tokens which appear in the corpus less frequently than this number will be replaced with 
  // "unk" in feature extraction
  int minTokenFrequency;

  bool checkGradient;

  int hackK;

  string vocabFilename;
};

#endif
