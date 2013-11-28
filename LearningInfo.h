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
#include "IAlignmentSampler.h"

using namespace std;

class LearningInfo {
 public:

 LearningInfo(boost::mpi::communicator *mpiWorld) : mpiWorld(mpiWorld)  {
    SetSharedMemorySegment(mpiWorld->rank() == 0);
    useMaxIterationsCount = false;
    useMinLikelihoodDiff = false;
    useEarlyStopping = false;
    trainToDevDataSize = 10;
    iterationsCount = 0;
    minLikelihoodDiff = 1.0;
    maxIterationsCount = 10;
    saveAlignmentFstsOnDisk = false;
    neighborhood = DiscriminativeLexicon::ALL;
    samplesCount = 1000;
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
    unspecified = 0;
    unspecified2 = 0;
    firstKExamplesToLabel = 1;
    invokeCallbackFunctionEveryKIterations = 10;
    endOfKIterationsCallbackFunction = 0;
    nSentsPerDot = 1;
    emIterationsCount = 1;
    thetaOptMethod = 0;
    initializeThetasWithModel1 = false;
    initializeThetasWithGaussian = true;
    initializeThetasWithUniform = false;
    initializeLambdasWithZero = true;
    initializeLambdasWithOne = false;
    initializeLambdasWithGaussian = false;
    allowNullAlignments = true;
    maxSequenceLength = 40;
    hiddenSequenceIsMarkovian = true;
    cacheActiveFeatures = false;
    multinomialSymmetricDirichletAlpha = 1.0;
    variationalInferenceOfMultinomials = false;
    testWithCrfOnly = false;
    optimizeLambdasFirst = false;
  }

  bool IsModelConverged() {
    assert(useMaxIterationsCount || useMinLikelihoodDiff || useEarlyStopping || useMinLikelihoodRelativeDiff);
    // logging
    if(useMaxIterationsCount) {
      cerr << "rank #" << mpiWorld->rank() << ": iterationsCount = " << iterationsCount << ". max = " << maxIterationsCount << endl;
    }
    if(useMinLikelihoodDiff && 
       iterationsCount > 1) {
      cerr << "rank #" << mpiWorld->rank() << ": loglikelihoodDiff = " << fabs(logLikelihood[iterationsCount-1] - 
					  logLikelihood[iterationsCount-2]) << ". min = " << minLikelihoodDiff << endl;
    }
    if(useEarlyStopping &&
       iterationsCount > 1) {
      cerr << "rank #" << mpiWorld->rank() << ": validationLikelihood[" << iterationsCount-2 << "] = " << validationLogLikelihood[iterationsCount-2] << endl;
      cerr << "rank #" << mpiWorld->rank() << ": validationLikelihood[" << iterationsCount-1 << "] = " << validationLogLikelihood[iterationsCount-1] << endl;
      cerr << "rank #" << mpiWorld->rank() << ": convergence criterion: stop training when loglikelihood no longer decreases, after the second iteration" << endl;
    }
    
    double absoluteDiff = 0.0, relativeDiff = 0.0;
    if(useMinLikelihoodRelativeDiff &&
       iterationsCount > 1) {
      absoluteDiff = fabs(logLikelihood[iterationsCount-1] - logLikelihood[iterationsCount-2]);
      assert(logLikelihood[iterationsCount-2] != 0);
      relativeDiff = fabs(absoluteDiff / logLikelihood[iterationsCount-2]);
      cerr << "rank #" << mpiWorld->rank() << ": loglikelihoodRelativeDiff = " << relativeDiff << ".min = " << minLikelihoodRelativeDiff << endl;
    }
    
    cerr << endl;
    // check for convergnece conditions
    if(useMaxIterationsCount && 
       maxIterationsCount <= iterationsCount) {
      return true;
    } 
    if(useMinLikelihoodDiff && 
       iterationsCount > 2 &&
       minLikelihoodDiff > fabs(logLikelihood[iterationsCount-1] - 
					    logLikelihood[iterationsCount-2])) {
      return true;
    } 
    if(useEarlyStopping &&
       iterationsCount > 5 &&
       validationLogLikelihood[iterationsCount-1] - validationLogLikelihood[iterationsCount-2] > 0) {
      return true;
    }
    if(useMinLikelihoodRelativeDiff && 
       iterationsCount > 2 && 
       minLikelihoodRelativeDiff > relativeDiff) {
      return true;
    }
    // none of the convergence conditions apply!
    return false;
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
    size_t segmentSize = 30 * 1024; // in GBs
    segmentSize *= 1024 * 1024;
    string SEGMENT_NAME = "segment";
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
  int samplesCount;

  // output
  int iterationsCount;
  vector< float > logLikelihood;
  vector< float > validationLogLikelihood;  

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

  int unspecified, unspecified2;

  unsigned firstKExamplesToLabel;

  unsigned invokeCallbackFunctionEveryKIterations;

  void (*endOfKIterationsCallbackFunction)();
  
  //bool smoothMultinomialParams;

  unsigned nSentsPerDot;

  unsigned emIterationsCount;

  OptMethod *thetaOptMethod;
  
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

  bool optimizeLambdasFirst;

  // train models for the reverse corpus direction
  bool reverse;

  // shared memory segment to efficiently share objects across processes
  boost::interprocess::managed_shared_memory *sharedMemorySegment;

  // the filenames specifying output of other word aligners for this dataset
  vector< string > otherAlignersOutputFilenames;

  // the id of the sentence pair currently being processed
  int currentSentId;

};

#endif
