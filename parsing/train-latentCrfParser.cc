#include <fenv.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/thread/thread.hpp>
#include <boost/program_options.hpp>
#include <boost/foreach.hpp>

#include "LatentCrfParser.h"

using namespace fst;
using namespace std;
namespace mpi = boost::mpi;
namespace po = boost::program_options;

/*
void my_handler(int s) {
  
  cerr << "___________________//////////////////////// INTERRUPTED " << s << "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_________" << endl;
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
*/

string GetOutputPrefix(int argc, char **argv) {
  string OUTPUT_PREFIX_OPTION("--output-prefix");
  for(int i = 0; i < argc; i++) {
    string currentOption(argv[i]);
    if(currentOption == OUTPUT_PREFIX_OPTION) {
      if(i+1 == argc) assert(false);
      return string(argv[i+1]);
    }
  }
  assert(false);
}


bool ParseParameters(int argc, char **argv, string &textFilename, 
		     string &initialLambdaParamsFilename, string &initialThetaParamsFilename, 
		     string &wordPairFeaturesFilename, string &outputFilenamePrefix, 
                     LearningInfo &learningInfo) {
  
  string HELP = "help",
    TRAIN_DATA = "train-data", 
    INIT_LAMBDA = "init-lambda",
    INIT_THETA = "init-theta", 
    WORDPAIR_FEATS = "wordpair-feats",
    OUTPUT_PREFIX = "output-prefix", 
    TEST_SIZE = "test-size",
    FEAT = "feat",
    WEIGHTED_L2_STRENGTH = "weighted-l2-strength",
    L2_STRENGTH = "l2-strength",
    L1_STRENGTH = "l1-strength",
    MAX_ITER_COUNT = "max-iter-count",
    MIN_RELATIVE_DIFF = "min-relative-diff",
    MAX_LBFGS_ITER_COUNT = "max-lbfgs-iter-count",
    //MAX_ADAGRAD_ITER_COUNT = "max-adagrad-iter-count",
    MAX_EM_ITER_COUNT = "max-em-iter-count",
    MAX_MODEL1_ITER_COUNT = "max-model1-iter-count",
    OPTIMIZER = "optimizer",
    MINIBATCH_SIZE = "minibatch-size",
    //LOGLINEAR_OPT_FIX_Z_GIVEN_X = "loglinear-opt-fix-z-given-x",
    DIRICHLET_ALPHA = "dirichlet-alpha",
    VARIATIONAL_INFERENCE = "variational-inference",
    TEST_WITH_CRF_ONLY = "test-with-crf-only",
    REVERSE = "reverse",
    OPTIMIZE_LAMBDAS_FIRST = "optimize-lambdas-first",
    //TGT_WORD_CLASSES_FILENAME = "tgt-word-classes-filename"
    ;

  // Declare the supported options.
  po::options_description desc("train-latentCrfParser options");
  desc.add_options()
    (HELP.c_str(), "produce help message")
    (TRAIN_DATA.c_str(), po::value<string>(&textFilename), "(filename) parallel data used for training the model")
    (INIT_LAMBDA.c_str(), po::value<string>(&initialLambdaParamsFilename), "(filename) initial weights of lambda parameters")
    (INIT_THETA.c_str(), po::value<string>(&initialThetaParamsFilename), "(filename) initial weights of theta parameters")
    (WORDPAIR_FEATS.c_str(), po::value<string>(&wordPairFeaturesFilename), "(filename) features defined for pairs of source-target word pairs")
    (OUTPUT_PREFIX.c_str(), po::value<string>(&outputFilenamePrefix), "(filename prefix) all filenames written by this program will have this prefix")
     // deen=150 // czen=515 // fren=447;
    (TEST_SIZE.c_str(), po::value<unsigned int>(&learningInfo.firstKExamplesToLabel), "(int) specifies the number of sentence pairs in train-data to eventually generate alignments for") 
    (FEAT.c_str(), po::value< vector< string > >(), "(multiple strings) specifies feature templates to be fired")
    (WEIGHTED_L2_STRENGTH.c_str(), po::value<float>()->default_value(0.0), "(double) strength of a weighted l2 regularizer")
    (L2_STRENGTH.c_str(), po::value<float>()->default_value(0.0), "(double) strength of an l2 regularizer")
    (L1_STRENGTH.c_str(), po::value<float>()->default_value(0.0), "(double) strength of an l1 regularizer")
    (MAX_ITER_COUNT.c_str(), po::value<int>(&learningInfo.maxIterationsCount)->default_value(50), "(int) max number of coordinate descent iterations after which the model is assumed to have converged")
    (MIN_RELATIVE_DIFF.c_str(), po::value<float>(&learningInfo.minLikelihoodRelativeDiff)->default_value(0.03), "(double) convergence threshold for the relative difference between the objective value in two consecutive coordinate descent iterations")
    (MAX_LBFGS_ITER_COUNT.c_str(), po::value<int>(&learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxIterations)->default_value(6), "(int) quit LBFGS optimization after this many iterations")
    //(MAX_ADAGRAD_ITER_COUNT.c_str(), po::value<int>(&learningInfo.optimizationMethod.subOptMethod->adagradParams.maxIterations)->default_value(4), "(int) quit Adagrad optimization after this many iterations")
    (MAX_EM_ITER_COUNT.c_str(), po::value<unsigned int>(&learningInfo.emIterationsCount)->default_value(3), "(int) quit EM optimization after this many iterations")
    (NO_DIRECT_DEP_BTW_HIDDEN_LABELS.c_str(), "(flag) consecutive labels are independent given observation sequence")
    (CACHE_FEATS.c_str(), po::value<bool>(&learningInfo.cacheActiveFeatures)->default_value(false), "(flag) (set by default) maintains and uses a map from a factor to its active features to speed up training, at the expense of higher memory requirements.")
    (OPTIMIZER.c_str(), po::value<string>(), "(string) optimization algorithm to use for updating loglinear parameters")
    (MINIBATCH_SIZE.c_str(), po::value<int>(&learningInfo.optimizationMethod.subOptMethod->miniBatchSize)->default_value(0), "(int) minibatch size for optimizing loglinear params. Defaults to zero which indicates batch training.")
    (LOGLINEAR_OPT_FIX_Z_GIVEN_X.c_str(), po::value<bool>(&learningInfo.fixPosteriorExpectationsAccordingToPZGivenXWhileOptimizingLambdas)->default_value(false), "(flag) (clera by default) fix the feature expectations according to p(Z|X), which involves both multinomial and loglinear parameters. This speeds up the optimization of loglinear parameters and makes it convex; but it does not have principled justification.")
    (MAX_MODEL1_ITER_COUNT.c_str(), po::value<int>(&maxModel1IterCount)->default_value(15), "(int) (defaults to 15) number of model 1 iterations to use for initializing theta parameters")
    (DIRICHLET_ALPHA.c_str(), po::value<double>(&learningInfo.multinomialSymmetricDirichletAlpha)->default_value(1.01), "(double) (defaults to 1.01) alpha of the symmetric dirichlet prior of the multinomial parameters.")
    (VARIATIONAL_INFERENCE.c_str(), po::value<bool>(&learningInfo.variationalInferenceOfMultinomials)->default_value(false), "(bool) (defaults to false) use variational inference approximation of the dirichlet prior of multinomial parameters.")
    (TEST_WITH_CRF_ONLY.c_str(), po::value<bool>(&learningInfo.testWithCrfOnly)->default_value(false), "(bool) (defaults to false) only use the crf model (i.e. not the multinomials) to make predictions.")
    (REVERSE.c_str(), po::value<bool>(&learningInfo.reverse)->default_value(false), "(flag) (defaults to false) train models for the reverse direction.")
    (OPTIMIZE_LAMBDAS_FIRST.c_str(), po::value<bool>(&learningInfo.optimizeLambdasFirst)->default_value(false), "(flag) (defaults to false) in the very first coordinate descent iteration, don't update thetas.")
    (OTHER_ALIGNERS_OUTPUT_FILENAMES.c_str(), po::value< vector< string > >(&learningInfo.otherAlignersOutputFilenames), "(multiple strings) specifies filenames which consist of word alignment output for the training corpus")
    (TGT_WORD_CLASSES_FILENAME.c_str(), po::value<string>(&learningInfo.tgtWordClassesFilename), "(string) specifies filename of word classes for the target vocabulary. Each line consists of three fields: word class, word type and frequency (tab-separated)")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count(HELP.c_str())) {
    cerr << desc << endl;
    return false;
  }

  if (vm.count(MAX_LBFGS_ITER_COUNT.c_str())) {
    learningInfo.optimizationMethod.subOptMethod->lbfgsParams.memoryBuffer = 
      vm[MAX_LBFGS_ITER_COUNT.c_str()].as<int>();
  }
  

  if (vm.count(TRAIN_DATA.c_str()) == 0) {
    cerr << TRAIN_DATA << " option is mandatory" << endl;
    cerr << desc << endl;
    return false;
  }
  
  if (vm.count(FEAT.c_str()) == 0) {
    cerr << "No features were specified. We will enable src-tgt word pair identities features by default." << endl;
    learningInfo.featureTemplates.push_back(FeatureTemplate::SRC0_TGT0);
  }

  if(vm[L2_STRENGTH.c_str()].as<float>() > 0.0) {
    learningInfo.optimizationMethod.subOptMethod->regularizationStrength = vm[L2_STRENGTH.c_str()].as<float>();
    learningInfo.optimizationMethod.subOptMethod->regularizer = Regularizer::L2;
  } else if (vm[L1_STRENGTH.c_str()].as<float>() > 0.0) {
    learningInfo.optimizationMethod.subOptMethod->regularizationStrength = vm[L1_STRENGTH.c_str()].as<float>();
    learningInfo.optimizationMethod.subOptMethod->regularizer = Regularizer::L1;
  } else if (vm[WEIGHTED_L2_STRENGTH.c_str()].as<float>() > 0.0) {
    learningInfo.optimizationMethod.subOptMethod->regularizationStrength = vm[WEIGHTED_L2_STRENGTH.c_str()].as<float>();
    learningInfo.optimizationMethod.subOptMethod->regularizer = Regularizer::WeightedL2;
  } else {
    learningInfo.optimizationMethod.subOptMethod->regularizationStrength = 0.0;
    learningInfo.optimizationMethod.subOptMethod->regularizer = Regularizer::NONE;
  }
  
  for (auto featIter = vm[FEAT.c_str()].as<vector<string> >().begin();
      featIter != vm[FEAT.c_str()].as<vector<string> >().end(); ++featIter) {
    if(*featIter == "LABEL_BIGRAM") {
      assert(false); // this feature does not make sense for word alignment
      learningInfo.featureTemplates.push_back(FeatureTemplate::LABEL_BIGRAM);
    } else if(*featIter == "SRC_BIGRAM") {
      learningInfo.featureTemplates.push_back(FeatureTemplate::SRC_BIGRAM);
    } else if(*featIter == "ALIGNMENT_JUMP") {
      learningInfo.featureTemplates.push_back(FeatureTemplate::ALIGNMENT_JUMP);
    } else if(*featIter == "LOG_ALIGNMENT_JUMP") {
      learningInfo.featureTemplates.push_back(FeatureTemplate::LOG_ALIGNMENT_JUMP);
    } else if(*featIter == "ALIGNMENT_JUMP_IS_ZERO") {
      learningInfo.featureTemplates.push_back(FeatureTemplate::ALIGNMENT_JUMP_IS_ZERO);
    } else if(*featIter == "SRC0_TGT0") {
      learningInfo.featureTemplates.push_back(FeatureTemplate::SRC0_TGT0);
    } else if(*featIter == "PRECOMPUTED") {
      learningInfo.featureTemplates.push_back(FeatureTemplate::PRECOMPUTED);
    } else if (*featIter == "DIAGONAL_DEVIATION") {
      learningInfo.featureTemplates.push_back(FeatureTemplate::DIAGONAL_DEVIATION);
    } else if (*featIter == "SRC_WORD_BIAS") {
      learningInfo.featureTemplates.push_back(FeatureTemplate::SRC_WORD_BIAS);
    } else if(*featIter == "SYNC_START" ){
      learningInfo.featureTemplates.push_back(FeatureTemplate::SYNC_START);
    } else if(*featIter == "SYNC_END") {
      learningInfo.featureTemplates.push_back(FeatureTemplate::SYNC_END);
    } else if(*featIter == "OTHER_ALIGNERS") {
      learningInfo.featureTemplates.push_back(FeatureTemplate::OTHER_ALIGNERS);
    } else if(*featIter == "NULL_ALIGNMENT") {
      learningInfo.featureTemplates.push_back(FeatureTemplate::NULL_ALIGNMENT);
    } else if(*featIter == "NULL_ALIGNMENT_LENGTH_RATIO") {
      learningInfo.featureTemplates.push_back(FeatureTemplate::NULL_ALIGNMENT_LENGTH_RATIO);
    } else {
      assert(false);
    }
  }
  
  learningInfo.hiddenSequenceIsMarkovian = false;
  
  if(vm.count(OPTIMIZER.c_str())) {
    if(vm[OPTIMIZER.c_str()].as<string>() == "adagrad") {
      learningInfo.optimizationMethod.subOptMethod->algorithm = OptAlgorithm::ADAGRAD;
    } else {
      cerr << "option --optimizer cannot take the value " << vm[OPTIMIZER.c_str()].as<string>() << endl;
      return false;
    }
  }
  
  // logging
  if(learningInfo.mpiWorld->rank() == 0) {
    cerr << "program options are as follows:" << endl;
    cerr << TRAIN_DATA << "=" << textFilename << endl;
    cerr << INIT_LAMBDA << "=" << initialLambdaParamsFilename << endl;
    cerr << INIT_THETA << "=" << initialThetaParamsFilename << endl;
    cerr << WORDPAIR_FEATS << "=" << wordPairFeaturesFilename << endl;
    cerr << OUTPUT_PREFIX << "=" << outputFilenamePrefix << endl;
    cerr << TEST_SIZE << "=" << learningInfo.firstKExamplesToLabel << endl;
    cerr << FEAT << "=";
    for (auto featIter = vm[FEAT.c_str()].as<vector<string> >().begin();
	 featIter != vm[FEAT.c_str()].as<vector<string> >().end(); ++featIter) {
      cerr << *featIter << " ";
    }
    cerr << endl;
    cerr << L2_STRENGTH << "=" << vm[L2_STRENGTH.c_str()].as<float>() << endl;
    cerr << WEIGHTED_L2_STRENGTH << "=" << vm[WEIGHTED_L2_STRENGTH.c_str()].as<float>() << endl;
    cerr << L1_STRENGTH << "=" << vm[L1_STRENGTH.c_str()].as<float>() << endl;
    cerr << MAX_ITER_COUNT << "=" << learningInfo.maxIterationsCount << endl;
    cerr << MIN_RELATIVE_DIFF << "=" << learningInfo.minLikelihoodRelativeDiff << endl;
    cerr << MAX_LBFGS_ITER_COUNT << "=" << learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxIterations << endl;
    cerr << MAX_EM_ITER_COUNT << "=" << learningInfo.emIterationsCount << endl;
    cerr << NO_DIRECT_DEP_BTW_HIDDEN_LABELS << "=" << !learningInfo.hiddenSequenceIsMarkovian << endl;
    cerr << CACHE_FEATS << "=" << learningInfo.cacheActiveFeatures << endl;
    if(vm.count(OPTIMIZER.c_str())) {
      cerr << OPTIMIZER << "=" << vm[OPTIMIZER.c_str()].as<string>() << endl;
    }
    cerr << MINIBATCH_SIZE << "=" << learningInfo.optimizationMethod.subOptMethod->miniBatchSize << endl;
    cerr << LOGLINEAR_OPT_FIX_Z_GIVEN_X << "=" << learningInfo.fixPosteriorExpectationsAccordingToPZGivenXWhileOptimizingLambdas << endl;
    cerr << MAX_MODEL1_ITER_COUNT << "=" << maxModel1IterCount << endl;
    cerr << DIRICHLET_ALPHA << "=" << learningInfo.multinomialSymmetricDirichletAlpha << endl;
    cerr << VARIATIONAL_INFERENCE << "=" << learningInfo.variationalInferenceOfMultinomials << endl;
    cerr << TEST_WITH_CRF_ONLY << "=" << learningInfo.testWithCrfOnly << endl;
    cerr << REVERSE << "=" << learningInfo.reverse << endl;
    cerr << OPTIMIZE_LAMBDAS_FIRST << "=" << learningInfo.optimizeLambdasFirst << endl;
    cerr << OTHER_ALIGNERS_OUTPUT_FILENAMES << "=";
    for(auto filename = learningInfo.otherAlignersOutputFilenames.begin();
	filename != learningInfo.otherAlignersOutputFilenames.end(); ++filename) {
      cerr << *filename << " ";
    }
    cerr << endl << "=====================" << endl;
  }
    
  // validation
  if(vm[L2_STRENGTH.c_str()].as<float>() < 0.0 || \
     vm[L1_STRENGTH.c_str()].as<float>() < 0.0 || \
     vm[WEIGHTED_L2_STRENGTH.c_str()].as<float>() < 0.0) {
    cerr << "you can't give " << L2_STRENGTH.c_str() << " nor " << WEIGHTED_L2_STRENGTH.c_str() << " nor " << L1_STRENGTH.c_str() << 
      " negative values" << endl;
    cerr << desc << endl;
    return false;
  } else if((vm[L2_STRENGTH.c_str()].as<float>() > 0.0 && vm[L1_STRENGTH.c_str()].as<float>() > 0.0) || \
            (vm[L2_STRENGTH.c_str()].as<float>() > 0.0 && vm[WEIGHTED_L2_STRENGTH.c_str()].as<float>() > 0.0) || \
            (vm[WEIGHTED_L2_STRENGTH.c_str()].as<float>() > 0.0 && vm[L1_STRENGTH.c_str()].as<float>() > 0.0)) {
    cerr << "you can't only set " << L2_STRENGTH << " OR " << L1_STRENGTH  << " OR " << WEIGHTED_L2_STRENGTH  << 
      ". sorry :-/" << endl;
    cerr << desc << endl;
    return false;
  }
  
  return true;
}

/*
// returns the rank of the process which have found the best HMM parameters
void IbmModel1Initialize(mpi::communicator world, string textFilename, string outputFilenamePrefix, LatentCrfAligner &latentCrfAligner, string &NULL_SRC_TOKEN, string &initialThetaParamsFilename, int maxIterCount, LearningInfo& originalLearningInfo) {

  // only the master does this
  if(world.rank() != 0){
    return;
  }

  outputFilenamePrefix += ".ibm1";

  
  // configurations
  cerr << "rank #" << world.rank() << ": training the ibm model 1 to initialize latentCrfAligner parameters..." << endl;

  LearningInfo learningInfo = originalLearningInfo;
  learningInfo.useMaxIterationsCount = true;
  learningInfo.maxIterationsCount = maxIterCount;
  learningInfo.useMinLikelihoodRelativeDiff = true;
  // learningInfo.minLikelihoodRelativeDiff set by ParseParameters;
  learningInfo.debugLevel = DebugLevel::CORPUS;
  learningInfo.mpiWorld = &world;
  learningInfo.persistParamsAfterNIteration = 1;
  learningInfo.optimizationMethod.algorithm = OptAlgorithm::EXPECTATION_MAXIMIZATION;

  // initialize the model
  cerr << "initializing IbmModel1...";
  IbmModel1 ibmModel1(textFilename, outputFilenamePrefix, learningInfo, NULL_SRC_TOKEN, latentCrfAligner.vocabEncoder);
  cerr << "done." << endl;

  // train model parameters
  cerr << "rank #" << world.rank() << ": train the model..." << endl;
  ibmModel1.Train();
  cerr << "rank #" << world.rank() << ": training finished!" << endl;
  
  // only override theta params if initialThetaParamsFilename is not specified
  if(initialThetaParamsFilename.size() == 0 && learningInfo.initializeThetasWithModel1) {
    cerr << "rank #" << world.rank() << ": now update the multinomail params of the latentCrfAligner model." << endl;
    for(auto contextIter = ibmModel1.params.params.begin(); 
        contextIter != ibmModel1.params.params.end();
        contextIter++) {
      for(auto probIter = contextIter->second.begin(); probIter != contextIter->second.end(); probIter++) {
        if(learningInfo.tgtWordClassesFilename.size() == 0) {
          latentCrfAligner.nLogThetaGivenOneLabel.params[contextIter->first][probIter->first] = probIter->second;
        } else {
          int64_t tgtWordClass = latentCrfAligner.tgtWordToClass[probIter->first];
          latentCrfAligner.nLogThetaGivenOneLabel.params[contextIter->first][probIter->first] = probIter->second;
        }
      }
    }
  }
  
  // nLogThetaGivenOneLabel is not normalized
  // TODO: normalize it
  //MultinomialParams::NormalizeParams<int64_t>(latentCrfAligner.nLogThetaGivenOneLabel, 
  //                                            learningInfo.multinomialSymmetricDirichletAlpha, 
  //                                            true, true, 
  //                                            learningInfo.variationalInferenceOfMultinomials);


  cerr << "rank #" << world.rank() << ": ibm model 1 initialization finished." << endl;
}
*/

void endOfKIterationsCallbackFunction() {
  // get hold of the model
  LatentCrfModel *model = LatentCrfParser::GetInstance();
  LatentCrfParser &parser = *( (LatentCrfParser*) model );

  // fix learningInfo.test_size
  cerr << "firstKExamplesToLabel = " << parser.learningInfo.firstKExamplesToLabel << endl;
  if(parser.learningInfo.firstKExamplesToLabel <= 0) {
    parser.learningInfo.firstKExamplesToLabel = parser.examplesCount;
    cerr << "firstKExamplesToLabel = " << parser.learningInfo.firstKExamplesToLabel << endl;
  }

  // find viterbi alignment for the top K examples of the training set (i.e. our test set)
  stringstream labelsFilename;
  labelsFilename << parser.outputPrefix << ".labels.iter" << parser.learningInfo.iterationsCount;
  parser.Label(labelsFilename.str());
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

int main(int argc, char **argv) {  

  // register interrupt handlers
  register_my_handler();

  // boost mpi initialization
  mpi::environment env(argc, argv);
  mpi::communicator world;

  LearningInfo learningInfo(&world, GetOutputPrefix(argc, argv));
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
  learningInfo.useSparseVectors = true;
  learningInfo.persistParamsAfterNIteration = 1;
  // block coordinate descent
  learningInfo.optimizationMethod.algorithm = OptAlgorithm::BLOCK_COORD_DESCENT;
  // lbfgs
  learningInfo.optimizationMethod.subOptMethod = new OptMethod();
  learningInfo.optimizationMethod.subOptMethod->algorithm = OptAlgorithm::LBFGS;
  learningInfo.optimizationMethod.subOptMethod->miniBatchSize = 0;
  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxEvalsPerIteration = 4;
  learningInfo.optimizationMethod.subOptMethod->moveAwayPenalty = 0.0;
  learningInfo.retryLbfgsOnRoundingErrors = true;
  // thetas
  learningInfo.thetaOptMethod = new OptMethod();
  learningInfo.thetaOptMethod->algorithm = OptAlgorithm::EXPECTATION_MAXIMIZATION;
  // general
  learningInfo.supervisedTraining = false;
  learningInfo.invokeCallbackFunctionEveryKIterations = 1;
  learningInfo.endOfKIterationsCallbackFunction = endOfKIterationsCallbackFunction;

  // hot configs
  learningInfo.allowNullAlignments = true;
  learningInfo.nSentsPerDot = 250;

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
  
  // initialize the model
  LatentCrfModel* model = LatentCrfParser::GetInstance(textFilename, 
							outputFilenamePrefix, 
							learningInfo, 
							FIRST_LABEL_ID, 
							initialLambdaParamsFilename, 
							initialThetaParamsFilename,
							wordPairFeaturesFilename);
  
  LatentCrfParser &latentCrfAligner = *((LatentCrfParser*)model);

  /*
  if(initialThetaParamsFilename.size() == 0) {
    // ibm model 1 initialization of theta params. 
    IbmModel1Initialize(world, textFilename, outputFilenamePrefix, latentCrfAligner, latentCrfAligner.NULL_TOKEN_STR, initialThetaParamsFilename, ibmModel1MaxIterCount, learningInfo);
  }
  */
  
  latentCrfAligner.BroadcastTheta(0);
  
  assert(model->lambda->IsSealed());

  // unsupervised training of the model
  model->Train();

  // print best params
  if(world.rank() == 0) {
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

  // fix learningInfo.test_size
  LatentCrfParser &aligner = *( (LatentCrfParser*) model );
  cerr << "firstKExamplesToLabel = " << aligner.learningInfo.firstKExamplesToLabel << endl;
  if(aligner.learningInfo.firstKExamplesToLabel <= 0) {
    aligner.learningInfo.firstKExamplesToLabel = aligner.examplesCount;
    cerr << "firstKExamplesToLabel = " << aligner.learningInfo.firstKExamplesToLabel << endl;
  }
  
  ((LatentCrfParser*)model)->Label(labelsFilename);
  if(learningInfo.mpiWorld->rank() == 0) {
    cerr << "alignments can be found at " << labelsFilename << endl;
  }

  learningInfo.ClearSharedMemorySegment();
}
