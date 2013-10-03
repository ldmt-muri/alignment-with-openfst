#include <fenv.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/thread/thread.hpp>
#include <boost/program_options.hpp>

#include "LatentCrfAligner.h"
#include "IbmModel1.h"

using namespace fst;
using namespace std;
namespace mpi = boost::mpi;
namespace po = boost::program_options;

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
    //exit(0);
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

bool ParseParameters(int argc, char **argv, string &textFilename, 
  string &initialLambdaParamsFilename, string &initialThetaParamsFilename, 
  string &wordPairFeaturesFilename, string &outputFilenamePrefix, 
                     LearningInfo &learningInfo, int &maxModel1IterCount) {
  
  string HELP = "help",
    TRAIN_DATA = "train-data", 
    INIT_LAMBDA = "init-lambda",
    INIT_THETA = "init-theta", 
    WORDPAIR_FEATS = "wordpair-feats",
    OUTPUT_PREFIX = "output-prefix", 
    TEST_SIZE = "test-size",
    FEAT = "feat",
    L2_STRENGTH = "l2-strength",
    L1_STRENGTH = "l1-strength",
    MAX_ITER_COUNT = "max-iter-count",
    MIN_RELATIVE_DIFF = "min-relative-diff",
    MAX_LBFGS_ITER_COUNT = "max-lbfgs-iter-count",
    MAX_ADAGRAD_ITER_COUNT = "max-adagrad-iter-count",
    MAX_EM_ITER_COUNT = "max-em-iter-count",
    MAX_MODEL1_ITER_COUNT = "max-model1-iter-count",
    NO_DIRECT_DEP_BTW_HIDDEN_LABELS = "no-direct-dep-btw-hidden-labels",
    CACHE_FEATS = "cache-feats",
    OPTIMIZER = "optimizer",
    MINIBATCH_SIZE = "minibatch-size",
    LOGLINEAR_OPT_FIX_Z_GIVEN_X = "loglinear-opt-fix-z-given-x",
    DIRICHLET_ALPHA = "dirichlet-alpha";

    

  // Declare the supported options.
  po::options_description desc("train-latentCrfAligner options");
  desc.add_options()
    (HELP.c_str(), "produce help message")
    (TRAIN_DATA.c_str(), po::value<string>(&textFilename), "(filename) parallel data used for training the model")
    (INIT_LAMBDA.c_str(), po::value<string>(&initialLambdaParamsFilename), "(filename) initial weights of lambda parameters")
    (INIT_THETA.c_str(), po::value<string>(&initialThetaParamsFilename), "(filename) initial weights of theta parameters")
    (WORDPAIR_FEATS.c_str(), po::value<string>(&wordPairFeaturesFilename), "(filename) features defined for pairs of source-target word pairs")
    (OUTPUT_PREFIX.c_str(), po::value<string>(&outputFilenamePrefix), "(filename prefix) all filenames written by this program will have this prefix")
     // deen=150 // czen=515 // fren=447;
    (TEST_SIZE.c_str(), po::value<unsigned int>(&learningInfo.firstKExamplesToLabel), "(int) specifies the number of sentence pairs in train-data to eventually generate alignments for") 
    (FEAT.c_str(), po::value< vector< int > >(&learningInfo.featureTemplates), "(multiple ints) specifies feature templates to be fired")
    (L2_STRENGTH.c_str(), po::value<float>(&learningInfo.optimizationMethod.subOptMethod->regularizationStrength)->default_value(1.0), "(double) strength of an l2 regularizer")
    (L1_STRENGTH.c_str(), po::value<float>(&learningInfo.optimizationMethod.subOptMethod->regularizationStrength)->default_value(0.0), "(double) strength of an l1 regularizer")
    (MAX_ITER_COUNT.c_str(), po::value<int>(&learningInfo.maxIterationsCount)->default_value(50), "(int) max number of coordinate descent iterations after which the model is assumed to have converged")
    (MIN_RELATIVE_DIFF.c_str(), po::value<float>(&learningInfo.minLikelihoodRelativeDiff)->default_value(0.01), "(double) convergence threshold for the relative difference between the objective value in two consecutive coordinate descent iterations")
    (MAX_LBFGS_ITER_COUNT.c_str(), po::value<int>(&learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxIterations)->default_value(6), "(int) quit LBFGS optimization after this many iterations")
//    (MAX_ADAGRAD_ITER_COUNT.c_str(), po::value<int>(&learningInfo.optimizationMethod.subOptMethod->adagradParams.maxIterations)->default_value(4), "(int) quit Adagrad optimization after this many iterations")
    (MAX_EM_ITER_COUNT.c_str(), po::value<unsigned int>(&learningInfo.emIterationsCount)->default_value(3), "(int) quit EM optimization after this many iterations")
    (NO_DIRECT_DEP_BTW_HIDDEN_LABELS.c_str(), "(flag) consecutive labels are independent given observation sequence")
    (CACHE_FEATS.c_str(), po::value<bool>(&learningInfo.cacheActiveFeatures)->default_value(false), "(flag) (set by default) maintains and uses a map from a factor to its active features to speed up training, at the expense of higher memory requirements.")
    (OPTIMIZER.c_str(), po::value<string>(), "(string) optimization algorithm to use for updating loglinear parameters")
    (MINIBATCH_SIZE.c_str(), po::value<int>(&learningInfo.optimizationMethod.subOptMethod->miniBatchSize)->default_value(0), "(int) minibatch size for optimizing loglinear params. Defaults to zero which indicates batch training.")
    (LOGLINEAR_OPT_FIX_Z_GIVEN_X.c_str(), po::value<bool>(&learningInfo.fixPosteriorExpectationsAccordingToPZGivenXWhileOptimizingLambdas)->default_value(false), "(flag) (clera by default) fix the feature expectations according to p(Z|X), which involves both multinomial and loglinear parameters. This speeds up the optimization of loglinear parameters and makes it convex; but it does not have principled justification.")
    (MAX_MODEL1_ITER_COUNT.c_str(), po::value<int>(&maxModel1IterCount)->default_value(15), "(int) (defaults to 15) number of model 1 iterations to use for initializing theta parameters")
    (DIRICHLET_ALPHA.c_str(), po::value<double>(&learningInfo.multinomialSymmetricDirichletAlpha)->default_value(1.0), "(double) (defaults to 1.0) alpha of the symmetric dirichlet prior of the multinomial parameters.")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count(HELP.c_str())) {
    cerr << desc << endl;
    return false;
  }

  if (vm.count(TRAIN_DATA.c_str()) == 0) {
    cerr << TRAIN_DATA << " option is mandatory" << endl;
    cerr << desc << endl;
    return false;
  }
  
  if (vm.count(FEAT.c_str()) == 0) {
    cerr << "No features were specified. We will enable src-tgt word pair identities features by default." << endl;
    int srcTgtWordPairIdentitiesFeatureTemplateId = 103;
    learningInfo.featureTemplates.push_back(srcTgtWordPairIdentitiesFeatureTemplateId);
  }

  if(vm[L2_STRENGTH.c_str()].as<float>() > 0.0) {
    learningInfo.optimizationMethod.subOptMethod->regularizer = Regularizer::L2;
  } else if (vm[L1_STRENGTH.c_str()].as<float>() > 0.0) {
    learningInfo.optimizationMethod.subOptMethod->regularizer = Regularizer::L1;
  }

  if(vm.count(NO_DIRECT_DEP_BTW_HIDDEN_LABELS.c_str())) {
    learningInfo.hiddenSequenceIsMarkovian = false;
  }
  
  if(vm.count(OPTIMIZER.c_str())) {
    if(vm[OPTIMIZER.c_str()].as<string>() == "adagrad") {
      learningInfo.optimizationMethod.subOptMethod->algorithm = OptAlgorithm::ADAGRAD;
    } else {
      cerr << "option --optimizer cannot take the value " << vm[OPTIMIZER.c_str()].as<string>() << endl;
      return false;
    }
  }
  
  // validation
  if(vm[L2_STRENGTH.c_str()].as<float>() < 0.0 || vm[L1_STRENGTH.c_str()].as<float>() < 0.0) {
    cerr << "you can't give " << L2_STRENGTH.c_str() << " nor " << L1_STRENGTH.c_str() << 
      " negative values" << endl;
    cerr << desc << endl;
    return false;
  } else if(vm[L2_STRENGTH.c_str()].as<float>() > 0.0 && vm[L1_STRENGTH.c_str()].as<float>() > 0.0) {
    cerr << "you can't set both " << L2_STRENGTH << " AND " << L1_STRENGTH  << 
      ". sorry :-/" << endl;
    cerr << desc << endl;
    return false;
  }
  
  return true;
}

// returns the rank of the process which have found the best HMM parameters
void IbmModel1Initialize(mpi::communicator world, string textFilename, string outputFilenamePrefix, LatentCrfAligner &latentCrfAligner, string &NULL_SRC_TOKEN, string &initialThetaParamsFilename, int maxIterCount) {

  // only the master does this
  if(world.rank() != 0){
    return;
  }

  outputFilenamePrefix += ".ibm1";

  
  // configurations
  cerr << "rank #" << world.rank() << ": training the ibm model 1 to initialize latentCrfAligner parameters..." << endl;

  LearningInfo learningInfo;
  learningInfo.useMaxIterationsCount = true;
  learningInfo.maxIterationsCount = maxIterCount;
  learningInfo.useMinLikelihoodRelativeDiff = true;
  // learningInfo.minLikelihoodRelativeDiff set by ParseParameters;
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
    for(auto contextIter = latentCrfAligner.nLogThetaGivenOneLabel.params.begin(); 
	contextIter != latentCrfAligner.nLogThetaGivenOneLabel.params.end();
	contextIter++) {
      
      for(auto probIter = contextIter->second.begin(); probIter != contextIter->second.end(); probIter++) {
	
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
  if(aligner.learningInfo.mpiWorld->rank() == 0) {
    //  return;
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
  LearningInfo learningInfo;
  if(world.rank() == 0) {
    cerr << "done." << endl;
  }

  unsigned FIRST_LABEL_ID = 4;

  // randomize draws
  int seed = time(NULL);
  srand(seed);

  // configurations
  if(world.rank() == 0) {
    cerr << "master" << world.rank() << ": setting configurations...";
  }
  // general 
  learningInfo.debugLevel = DebugLevel::MINI_BATCH;
  learningInfo.useMaxIterationsCount = true;
  learningInfo.mpiWorld = &world;
  learningInfo.useMinLikelihoodDiff = false;
  learningInfo.minLikelihoodDiff = 2;
  learningInfo.useMinLikelihoodRelativeDiff = true;
  //learningInfo.minLikelihoodRelativeDiff set by ParseParameters
  learningInfo.useSparseVectors = true;
  learningInfo.persistParamsAfterNIteration = 1;
  // block coordinate descent
  learningInfo.optimizationMethod.algorithm = OptAlgorithm::BLOCK_COORD_DESCENT;
  // lbfgs
  learningInfo.optimizationMethod.subOptMethod = new OptMethod();
  learningInfo.optimizationMethod.subOptMethod->algorithm = OptAlgorithm::LBFGS;
  learningInfo.optimizationMethod.subOptMethod->miniBatchSize = 0;
  // learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxIterations set by ParseParameters
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
  //learningInfo.fixDOverC = false;

  // hot configs
  learningInfo.allowNullAlignments = true;
  // learningInfo.firstKExamplesToLabel set by ParseParameters
  // learningInfo.emIterationsCount set by ParseParameters;
  learningInfo.nSentsPerDot = 250;
  //learningInfo.maxIterationsCount set by ParseParameters

  learningInfo.initializeThetasWithGaussian = false;
  learningInfo.initializeThetasWithUniform = false;
  learningInfo.initializeThetasWithModel1 = true;

  learningInfo.initializeLambdasWithGaussian = false;
  learningInfo.initializeLambdasWithZero = true;
  learningInfo.initializeLambdasWithOne = true;

  // parse cmd params
  string textFilename, outputFilenamePrefix, initialLambdaParamsFilename, initialThetaParamsFilename, wordPairFeaturesFilename;
  int ibmModel1MaxIterCount = 15;
  if(!ParseParameters(argc, argv, textFilename, initialLambdaParamsFilename, 
                      initialThetaParamsFilename, wordPairFeaturesFilename, outputFilenamePrefix, 
                      learningInfo, ibmModel1MaxIterCount)){
    assert(false);
  }

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
  IbmModel1Initialize(world, textFilename, outputFilenamePrefix, latentCrfAligner, latentCrfAligner.NULL_TOKEN_STR, initialThetaParamsFilename, ibmModel1MaxIterCount);

  latentCrfAligner.BroadcastTheta(0);
  latentCrfAligner.BroadcastLambdas(0);

  // unsupervised training of the model
  model->Train();

  // print best params
  if(learningInfo.mpiWorld->rank() == 0) {
    model->lambda->PersistParams(outputFilenamePrefix + string(".final.lambda.humane"), true);
    model->lambda->PersistParams(outputFilenamePrefix + string(".final.lambda"), false);
    model->PersistTheta(outputFilenamePrefix + string(".final.theta"));
  }

  // we don't need the slaves anymore
  if(world.rank() > 0) {
    //return 0;
  }
    
  // run viterbi (and write alignments in giza format)
  string labelsFilename = outputFilenamePrefix + ".labels";
  ((LatentCrfAligner*)model)->Label(labelsFilename);
  cerr << "alignments can be found at " << labelsFilename << endl;
}
