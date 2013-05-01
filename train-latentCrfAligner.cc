#include <fenv.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/thread/thread.hpp>

#include "LatentCrfAligner.h"
#include "IbmModel1.h"

using namespace fst;
using namespace std;
namespace mpi = boost::mpi;

typedef ProductArc<FstUtils::LogWeight, FstUtils::LogWeight> ProductLogArc;

void my_handler(int s) {
  
  cerr << "___________________//////////////////////// INTERRUPTED " << s << "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\___________________" << endl;
  cerr << "stopped training." << endl;
  LatentCrfModel *model = LatentCrfAligner::GetInstance();
  LatentCrfAligner &aligner = *( (LatentCrfAligner*) model );
  if(aligner.learningInfo.mpiWorld->rank() == 0) {
    cerr << "rank #" << aligner.learningInfo.mpiWorld->rank() << ": running viterbi..." << endl;
  } else {
    cerr << "rank #" << aligner.learningInfo.mpiWorld->rank() << ": will exit." << endl;
    exit(0);
  }
  string suffix = ".interrupted-labels";
  string labelsFilename = aligner.outputPrefix + suffix;
  aligner.Label(labelsFilename);
  cerr << "viterbi word alignment can be found at " << labelsFilename << endl;
  cerr << "now, persist the current model parameters..." << endl;
  suffix = ".interrupted-theta";
  string thetaFilename = aligner.outputPrefix + suffix;
  aligner.PersistTheta(thetaFilename);
  cerr << "done persisting theta params" << endl;
  cerr << "theta params can be found at " << thetaFilename << endl;
  suffix = ".interrupted-lambda";
  string lambdaFilename = aligner.outputPrefix + suffix;
  aligner.lambda->PersistParams(lambdaFilename);
  cerr << "done persisting lambda params." << endl;
  cerr << "lambda params can be found at " << lambdaFilename << endl;
  exit(0);
}

void register_my_handler() {
  struct sigaction sigIntHandler;

  sigIntHandler.sa_handler = my_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;

  sigaction(SIGINT, &sigIntHandler, NULL);
  sigaction(SIGTERM, &sigIntHandler, NULL);
  sigaction(SIGUSR1, &sigIntHandler, NULL);
  sigaction(SIGUSR2, &sigIntHandler, NULL);
}

void ParseParameters(int argc, char **argv, string &textFilename, string &initialLambdaParamsFilename, string &initialThetaParamsFilename, string &wordPairFeaturesFilename, string &outputFilenamePrefix) {
  assert(argc >= 5);
  textFilename = argv[1];
  initialLambdaParamsFilename = argv[2];
  if(initialLambdaParamsFilename == "none") {
    initialLambdaParamsFilename.clear();
  }
  initialThetaParamsFilename = argv[3];
  if(initialThetaParamsFilename == "none") {
    initialThetaParamsFilename.clear();
  }
  wordPairFeaturesFilename = argv[4];
  if(wordPairFeaturesFilename == "none") {
    wordPairFeaturesFilename.clear();
  }
  outputFilenamePrefix = argv[5];
}

// returns the rank of the process which have found the best HMM parameters
void IbmModel1Initialize(mpi::communicator world, string textFilename, string outputFilenamePrefix, LatentCrfAligner &latentCrfAligner, string &NULL_SRC_TOKEN, string &initialThetaParamsFilename) {

  // only the master does this
  if(world.rank() != 0){
    return;
  }

  outputFilenamePrefix += ".ibm1";

  // ibm model1 initializer can't initialize the latent crf multinomials when zI dpeends on both y_{i-1} and y_i
  assert(latentCrfAligner.learningInfo.zIDependsOnYIM1 == false);

  // configurations
  cerr << "rank #" << world.rank() << ": training the ibm model 1 to initialize latentCrfAligner parameters..." << endl;

  LearningInfo learningInfo;
  learningInfo.useMaxIterationsCount = true;
  learningInfo.maxIterationsCount = 15;
  learningInfo.useMinLikelihoodRelativeDiff = true;
  learningInfo.minLikelihoodRelativeDiff = 0.01;
  learningInfo.debugLevel = DebugLevel::CORPUS;
  learningInfo.mpiWorld = &world;
  learningInfo.persistParamsAfterNIteration = 10;
  learningInfo.optimizationMethod.algorithm = OptAlgorithm::EXPECTATION_MAXIMIZATION;

  // initialize the model
  IbmModel1 ibmModel1(textFilename, outputFilenamePrefix, learningInfo, NULL_SRC_TOKEN, latentCrfAligner.vocabEncoder);

  // train model parameters
  cerr << "rank #" << world.rank() << ": train the model..." << endl;
  ibmModel1.Train();
  cerr << "rank #" << world.rank() << ": training finished!" << endl;
  
  // only override theta params if initialThetaParamsFilename is not specified
  if(initialThetaParamsFilename.size() == 0 && learningInfo.initializeThetasWithModel1) {
    // now initialize the latentCrfAligner's theta parameters, and also augment the precomputed features with ibm model 1 features
    cerr << "rank #" << world.rank() << ": now update the multinomial params of the latentCrfALigner model." << endl;
    for(map<int, MultinomialParams::MultinomialParam>::iterator contextIter = latentCrfAligner.nLogThetaGivenOneLabel.params.begin(); 
	contextIter != latentCrfAligner.nLogThetaGivenOneLabel.params.end();
	contextIter++) {
      
      for(map<int, double>::iterator probIter = contextIter->second.begin(); probIter != contextIter->second.end(); probIter++) {
	
	assert(ibmModel1.params[contextIter->first].count(probIter->first) > 0);
	probIter->second = ibmModel1.params[contextIter->first][probIter->first];
      }
    }
  }
  
  cerr << "rank #" << world.rank() << ": ibm model 1 initialization finished." << endl;
}

void endOfKIterationsCallbackFunction() {
  // get hold of the model
  LatentCrfModel *model = LatentCrfAligner::GetInstance();
  LatentCrfAligner &aligner = *( (LatentCrfAligner*) model );
  if(aligner.learningInfo.mpiWorld->rank() != 0) {
    return;
  } 

  // find viterbi alignment for the top K examples of the training set (i.e. our test set)
  stringstream labelsFilename;
  labelsFilename << aligner.outputPrefix << ".labels.iter" << aligner.learningInfo.iterationsCount;
  aligner.Label(labelsFilename.str());
}

int main(int argc, char **argv) {  

  // register interrupt handlers
  register_my_handler();

  // boost mpi initialization
  mpi::environment env(argc, argv);
  mpi::communicator world;

  // parse arguments
  if(world.rank() == 0) {
    cerr << "master" << world.rank() << ": parsing arguments...";
  }
  string textFilename, outputFilenamePrefix, initialLambdaParamsFilename, initialThetaParamsFilename, wordPairFeaturesFilename;
  ParseParameters(argc, argv, textFilename, initialLambdaParamsFilename, initialThetaParamsFilename, wordPairFeaturesFilename, outputFilenamePrefix);
  if(world.rank() == 0) {
    cerr << "done." << endl;
  }

  unsigned NUMBER_OF_LABELS = 45;
  unsigned FIRST_LABEL_ID = 4;

  // randomize draws
  int seed = time(NULL);
  srand(seed);

  // configurations
  if(world.rank() == 0) {
    cerr << "master" << world.rank() << ": setting configurations...";
  }
  LearningInfo learningInfo;
  // general 
  learningInfo.debugLevel = DebugLevel::MINI_BATCH;
  learningInfo.useMaxIterationsCount = true;
  learningInfo.mpiWorld = &world;
  //  learningInfo.useMinLikelihoodDiff = true;
  //  learningInfo.minLikelihoodDiff = 10;
  learningInfo.useMinLikelihoodRelativeDiff = true;
  learningInfo.minLikelihoodRelativeDiff = 0.01;
  learningInfo.useSparseVectors = true;
  learningInfo.zIDependsOnYIM1 = false;
  learningInfo.persistParamsAfterNIteration = 1;
  // block coordinate descent
  learningInfo.optimizationMethod.algorithm = OptAlgorithm::BLOCK_COORD_DESCENT;
  // lbfgs
  learningInfo.optimizationMethod.subOptMethod = new OptMethod();
  learningInfo.optimizationMethod.subOptMethod->algorithm = OptAlgorithm::LBFGS;
  learningInfo.optimizationMethod.subOptMethod->regularizer = Regularizer::NONE;
  learningInfo.optimizationMethod.subOptMethod->regularizationStrength = 1.0;
  learningInfo.optimizationMethod.subOptMethod->miniBatchSize = 0;
  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxIterations = 4;
  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxEvalsPerIteration = 4;
  //  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.memoryBuffer = 50;
  //  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.precision = 0.00000000000000000000000001;
  learningInfo.optimizationMethod.subOptMethod->moveAwayPenalty = 0.0;
  learningInfo.retryLbfgsOnRoundingErrors = true;
  // thetas
  learningInfo.thetaOptMethod = new OptMethod();
  learningInfo.thetaOptMethod->algorithm = OptAlgorithm::EXPECTATION_MAXIMIZATION;
  //  learningInfo.thetaOptMethod->learningRate = 0.01;
  // general
  learningInfo.supervisedTraining = false;
  learningInfo.invokeCallbackFunctionEveryKIterations = 5;
  learningInfo.endOfKIterationsCallbackFunction = endOfKIterationsCallbackFunction;
  learningInfo.fixDOverC = false;

  // hot configs
  learningInfo.allowNullAlignments = true;
  learningInfo.firstKExamplesToLabel = 515;//447;
  learningInfo.emIterationsCount = 2;
  learningInfo.nSentsPerDot = 250;
  learningInfo.maxIterationsCount = 50;

  learningInfo.initializeThetasWithGaussian = false;
  learningInfo.initializeThetasWithUniform = false;
  learningInfo.initializeThetasWithModel1 = true;

  learningInfo.initializeLambdasWithGaussian = false;
  learningInfo.initializeLambdasWithZero = true;
  learningInfo.initializeLambdasWithOne = false;

  // add constraints
  learningInfo.constraints.clear();
  if(world.rank() == 0) {
    cerr << "done." << endl;
  }
  
  // initialize the model
  LatentCrfModel* model = LatentCrfAligner::GetInstance(textFilename, 
							outputFilenamePrefix, 
							learningInfo, 
							FIRST_LABEL_ID, 
							initialLambdaParamsFilename, 
							initialThetaParamsFilename,
							wordPairFeaturesFilename);
  LatentCrfAligner &latentCrfAligner = *((LatentCrfAligner*)model);
  
  // ibm model 1 initialization of theta params. also updates the lambda precomputed features by adding ibm model 1 probs
  IbmModel1Initialize(world, textFilename, outputFilenamePrefix, latentCrfAligner, latentCrfAligner.NULL_TOKEN_STR, initialThetaParamsFilename);

  latentCrfAligner.BroadcastTheta(0);
  latentCrfAligner.BroadcastLambdas(0);

  string ibm1PrecomputedFeatureId = "_ibm1";
  for(map<int, MultinomialParams::MultinomialParam>::iterator contextIter = latentCrfAligner.nLogThetaGivenOneLabel.params.begin(); 
      contextIter != latentCrfAligner.nLogThetaGivenOneLabel.params.end();
      contextIter++) {
    
    for(map<int, double>::iterator probIter = contextIter->second.begin(); probIter != contextIter->second.end(); probIter++) {

      latentCrfAligner.lambda->AddToPrecomputedFeaturesWith2Inputs(contextIter->first, probIter->first, ibm1PrecomputedFeatureId, probIter->second);
    }
  }

  // unsupervised training of the model
  model->Train();

  // print best params
  if(learningInfo.mpiWorld->rank() == 0) {
    model->lambda->PersistParams(outputFilenamePrefix + string(".final.lambda"));
    model->PersistTheta(outputFilenamePrefix + string(".final.theta"));
  }

  // we don't need the slaves anymore
  if(world.rank() > 0) {
    return 0;
  }
    
  // run viterbi (and write alignments in giza format)
  string labelsFilename = outputFilenamePrefix + ".labels";
  ((LatentCrfAligner*)model)->Label(labelsFilename);
  cerr << "alignments can be found at " << labelsFilename << endl;
}
