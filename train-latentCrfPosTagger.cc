#include <fenv.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/thread/thread.hpp>
#include <boost/program_options.hpp>
#include <boost/foreach.hpp>

#include "LatentCrfPosTagger.h"
#include "HmmModel.h"

using namespace fst;
using namespace std;
namespace mpi = boost::mpi;
namespace po = boost::program_options;

typedef ProductArc<FstUtils::LogWeight, FstUtils::LogWeight> ProductLogArc;

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

void ParseParameters(int argc, char **argv, string &textFilename, string &outputFilenamePrefix, string &goldLabelsFilename, LearningInfo &learningInfo) {
  assert(argc >= 3);
  textFilename = argv[1];
  outputFilenamePrefix = argv[2];
  if(argc >= 4) {
    goldLabelsFilename = argv[3];
  } else {
    goldLabelsFilename = "";
  }

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
    //MAX_ADAGRAD_ITER_COUNT = "max-adagrad-iter-count",
    MAX_EM_ITER_COUNT = "max-em-iter-count",
    MAX_MODEL1_ITER_COUNT = "max-model1-iter-count",
    NO_DIRECT_DEP_BTW_HIDDEN_LABELS = "no-direct-dep-btw-hidden-labels",
    CACHE_FEATS = "cache-feats",
    OPTIMIZER = "optimizer",
    MINIBATCH_SIZE = "minibatch-size",
    LOGLINEAR_OPT_FIX_Z_GIVEN_X = "loglinear-opt-fix-z-given-x",
    DIRICHLET_ALPHA = "dirichlet-alpha",
    VARIATIONAL_INFERENCE = "variational-inference",
    TEST_WITH_CRF_ONLY = "test-with-crf-only",
    REVERSE = "reverse",
    OPTIMIZE_LAMBDAS_FIRST = "optimize-lambdas-first",
    OTHER_ALIGNERS_OUTPUT_FILENAMES = "other-aligners-output-filenames",
    TGT_WORD_CLASSES_FILENAME = "tgt-word-classes-filename",
    GOLD_LABELS_FILENAME = "gold-labels-filename";

  string initialLambdaParamsFilename, initialThetaParamsFilename;

  // Declare the supported options.
  po::options_description desc("train-latentCrfAligner options");
  desc.add_options()
    (HELP.c_str(), "produce help message")
    (TRAIN_DATA.c_str(), po::value<string>(&textFilename), "(filename) parallel data used for training the model")
    (INIT_LAMBDA.c_str(), po::value<string>(&initialLambdaParamsFilename), "(filename) initial weights of lambda parameters")
    (INIT_THETA.c_str(), po::value<string>(&initialThetaParamsFilename), "(filename) initial weights of theta parameters")
    (OUTPUT_PREFIX.c_str(), po::value<string>(&outputFilenamePrefix), "(filename prefix) all filenames written by this program will have this prefix")
     // deen=150 // czen=515 // fren=447;
    (TEST_SIZE.c_str(), po::value<unsigned int>(&learningInfo.firstKExamplesToLabel), "(int) specifies the number of sentence pairs in train-data to eventually generate alignments for") 
    (FEAT.c_str(), po::value< vector< string > >(), "(multiple strings) specifies feature templates to be fired")
    (L2_STRENGTH.c_str(), po::value<float>(&learningInfo.optimizationMethod.subOptMethod->regularizationStrength)->default_value(1.0), "(double) strength of an l2 regularizer")
    (L1_STRENGTH.c_str(), po::value<float>(&learningInfo.optimizationMethod.subOptMethod->regularizationStrength)->default_value(0.0), "(double) strength of an l1 regularizer")
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
    (DIRICHLET_ALPHA.c_str(), po::value<double>(&learningInfo.multinomialSymmetricDirichletAlpha)->default_value(1.01), "(double) (defaults to 1.01) alpha of the symmetric dirichlet prior of the multinomial parameters.")
    (VARIATIONAL_INFERENCE.c_str(), po::value<bool>(&learningInfo.variationalInferenceOfMultinomials)->default_value(false), "(bool) (defaults to false) use variational inference approximation of the dirichlet prior of multinomial parameters.")
    (TEST_WITH_CRF_ONLY.c_str(), po::value<bool>(&learningInfo.testWithCrfOnly)->default_value(false), "(bool) (defaults to false) only use the crf model (i.e. not the multinomials) to make predictions.")
    (OPTIMIZE_LAMBDAS_FIRST.c_str(), po::value<bool>(&learningInfo.optimizeLambdasFirst)->default_value(false), "(flag) (defaults to false) in the very first coordinate descent iteration, don't update thetas.")
    (OTHER_ALIGNERS_OUTPUT_FILENAMES.c_str(), po::value< vector< string > >(&learningInfo.otherAlignersOutputFilenames), "(multiple strings) specifies filenames which consist of word alignment output for the training corpus")
    (TGT_WORD_CLASSES_FILENAME.c_str(), po::value<string>(&learningInfo.tgtWordClassesFilename), "(string) specifies filename of word classes for the target vocabulary. Each line consists of three fields: word class, word type and frequency (tab-separated)")
    (GOLD_LABELS_FILENAME.c_str(), po::value<string>(&goldLabelsFilename), "(string) specifies filename of the hand-annotated POS tags corresponding to training data") 
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count(HELP.c_str())) {
    cerr << desc << endl;
    assert(false);
  }

  if (vm.count(MAX_LBFGS_ITER_COUNT.c_str())) {
    learningInfo.optimizationMethod.subOptMethod->lbfgsParams.memoryBuffer = 
      vm[MAX_LBFGS_ITER_COUNT.c_str()].as<int>();
  }
  

  if (vm.count(TRAIN_DATA.c_str()) == 0) {
    cerr << TRAIN_DATA << " option is mandatory" << endl;
    cerr << desc << endl;
    assert(false);
  }
  
  if (vm.count(FEAT.c_str()) == 0) {
    cerr << "No features were specified. We will enable src-tgt word pair identities features by default." << endl;
    learningInfo.featureTemplates.push_back(FeatureTemplate::SRC0_TGT0);
  }

  if(vm[L2_STRENGTH.c_str()].as<float>() > 0.0) {
    learningInfo.optimizationMethod.subOptMethod->regularizer = Regularizer::L2;
  } else if (vm[L1_STRENGTH.c_str()].as<float>() > 0.0) {
    learningInfo.optimizationMethod.subOptMethod->regularizer = Regularizer::L1;
  }

  for (auto featIter = vm[FEAT.c_str()].as<vector<string> >().begin();
      featIter != vm[FEAT.c_str()].as<vector<string> >().end(); ++featIter) {
    if(*featIter == "LABEL_BIGRAM") {
      learningInfo.featureTemplates.push_back(FeatureTemplate::LABEL_BIGRAM);
    } else if(*featIter == "BOUNDARY_LABELS") {
      learningInfo.featureTemplates.push_back(FeatureTemplate::BOUNDARY_LABELS);
    } else if(*featIter == "EMISSION") {
      learningInfo.featureTemplates.push_back(FeatureTemplate::EMISSION);
    } else if(*featIter == "SRC_BIGRAM") {
      assert(false); // this feature does not make sense for POS tagging
      learningInfo.featureTemplates.push_back(FeatureTemplate::SRC_BIGRAM);
    } else if(*featIter == "ALIGNMENT_JUMP") {
      assert(false); // this feature does not make sense for POS tagging
      learningInfo.featureTemplates.push_back(FeatureTemplate::ALIGNMENT_JUMP);
    } else if(*featIter == "LOG_ALIGNMENT_JUMP") {
      assert(false); // this feature does not make sense for POS tagging
      learningInfo.featureTemplates.push_back(FeatureTemplate::LOG_ALIGNMENT_JUMP);
    } else if(*featIter == "ALIGNMENT_JUMP_IS_ZERO") {
      assert(false); // this feature does not make sense for POS tagging
      learningInfo.featureTemplates.push_back(FeatureTemplate::ALIGNMENT_JUMP_IS_ZERO);
    } else if(*featIter == "SRC0_TGT0") {
      assert(false); // this feature does not make sense for POS tagging
      learningInfo.featureTemplates.push_back(FeatureTemplate::SRC0_TGT0);
    } else if(*featIter == "PRECOMPUTED") {
      assert(false); // this feature does not make sense for POS tagging
      learningInfo.featureTemplates.push_back(FeatureTemplate::PRECOMPUTED);
    } else if (*featIter == "DIAGONAL_DEVIATION") {
      assert(false); // this feature does not make sense for POS tagging
      learningInfo.featureTemplates.push_back(FeatureTemplate::DIAGONAL_DEVIATION);
    } else if(*featIter == "SYNC_START" ){
      assert(false); // this feature does not make sense for POS tagging
      learningInfo.featureTemplates.push_back(FeatureTemplate::SYNC_START);
    } else if(*featIter == "SYNC_END") {
      assert(false); // this feature does not make sense for POS tagging
      learningInfo.featureTemplates.push_back(FeatureTemplate::SYNC_END);
    } else if(*featIter == "OTHER_ALIGNERS") {
      learningInfo.featureTemplates.push_back(FeatureTemplate::OTHER_ALIGNERS);
    } else if(*featIter == "NULL_ALIGNMENT") {
      assert(false); // this feature does not make sense for POS tagging
      learningInfo.featureTemplates.push_back(FeatureTemplate::NULL_ALIGNMENT);
    } else if(*featIter == "NULL_ALIGNMENT_LENGTH_RATIO") {
      assert(false); // this feature does not make sense for POS tagging
      learningInfo.featureTemplates.push_back(FeatureTemplate::NULL_ALIGNMENT_LENGTH_RATIO);
    } else {
      assert(false);
    }
  }
  
  if(vm.count(NO_DIRECT_DEP_BTW_HIDDEN_LABELS.c_str())) {
    learningInfo.hiddenSequenceIsMarkovian = false;
  }
  
  if(vm.count(OPTIMIZER.c_str())) {
    if(vm[OPTIMIZER.c_str()].as<string>() == "adagrad") {
      learningInfo.optimizationMethod.subOptMethod->algorithm = OptAlgorithm::ADAGRAD;
    } else {
      cerr << "option --optimizer cannot take the value " << vm[OPTIMIZER.c_str()].as<string>() << endl;
      assert(false);
    }
  }
  
  // logging
  if(learningInfo.mpiWorld->rank() == 0) {
    cerr << "program options are as follows:" << endl;
    cerr << TRAIN_DATA << "=" << textFilename << endl;
    cerr << INIT_LAMBDA << "=" << initialLambdaParamsFilename << endl;
    cerr << INIT_THETA << "=" << initialThetaParamsFilename << endl;
    //cerr << WORDPAIR_FEATS << "=" << wordPairFeaturesFilename << endl;
    cerr << OUTPUT_PREFIX << "=" << outputFilenamePrefix << endl;
    cerr << TEST_SIZE << "=" << learningInfo.firstKExamplesToLabel << endl;
    cerr << FEAT << "=";
    for (auto featIter = vm[FEAT.c_str()].as<vector<string> >().begin();
	 featIter != vm[FEAT.c_str()].as<vector<string> >().end(); ++featIter) {
      cerr << *featIter << " ";
    }
    cerr << endl;
    cerr << L2_STRENGTH << "=" << vm[L2_STRENGTH.c_str()].as<float>() << endl;
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
    //cerr << MAX_MODEL1_ITER_COUNT << "=" << maxModel1IterCount << endl;
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
    cerr << GOLD_LABELS_FILENAME << "=" << goldLabelsFilename << endl;
    cerr << endl << "=====================" << endl;
  }
    
  // validation
  if(vm[L2_STRENGTH.c_str()].as<float>() < 0.0 || vm[L1_STRENGTH.c_str()].as<float>() < 0.0) {
    cerr << "you can't give " << L2_STRENGTH.c_str() << " nor " << L1_STRENGTH.c_str() << 
      " negative values" << endl;
    cerr << desc << endl;
    assert(false);
  } else if(vm[L2_STRENGTH.c_str()].as<float>() > 0.0 && vm[L1_STRENGTH.c_str()].as<float>() > 0.0) {
    cerr << "you can't set both " << L2_STRENGTH << " AND " << L1_STRENGTH  << 
      ". sorry :-/" << endl;
    cerr << desc << endl;
    assert(false);
  }

}

// returns the rank of the process which have found the best HMM parameters
unsigned HmmInitialize(mpi::communicator world, string textFilename, string outputFilenamePrefix, int NUMBER_OF_LABELS, LatentCrfPosTagger &latentCrfPosTagger, int FIRST_LABEL_ID, string goldLabelsFilename) {

  outputFilenamePrefix += ".hmm";

  // configurations
  cerr << "rank #" << world.rank() << ": training the hmm model to initialize latentCrfPosTagger parameters..." << endl;

  bool persistHmmParams = false;

  LearningInfo learningInfo(&world, outputFilenamePrefix);
  learningInfo.useMaxIterationsCount = true;
  learningInfo.maxIterationsCount = 10;
  learningInfo.useMinLikelihoodRelativeDiff = true;
  learningInfo.minLikelihoodRelativeDiff = 0.001;
  learningInfo.debugLevel = DebugLevel::CORPUS;
  learningInfo.mpiWorld = &world;
  learningInfo.persistParamsAfterNIteration = 10;
  learningInfo.optimizationMethod.algorithm = OptAlgorithm::EXPECTATION_MAXIMIZATION;

  // initialize the model
  HmmModel2 hmmModel(textFilename, outputFilenamePrefix, learningInfo, NUMBER_OF_LABELS, FIRST_LABEL_ID);

  // train model parameters
  cerr << "rank #" << world.rank() << ": train the model..." << endl;
  hmmModel.Train();
  cerr << "rank #" << world.rank() << ": training finished!" << endl;
  
  // determine which rank got the best HMM model based on optimized likelihood
  assert(learningInfo.logLikelihood.size() > 0);
  // this processor's minimized nloglikelihood
  double rankOptimizedLoglikelihood = learningInfo.logLikelihood[learningInfo.logLikelihood.size()-1];
  cerr << "rank #" << world.rank() << ": the local maximum logliklihood i found is " << rankOptimizedLoglikelihood << endl; 
  // the minimum nloglikelihood obtained by any of the processors
  double globallyOptimizedLoglikelihood = 0;
  mpi::all_reduce<double>(world, rankOptimizedLoglikelihood, globallyOptimizedLoglikelihood, mpi::maximum<double>());
  assert(globallyOptimizedLoglikelihood != 0);
  cerr << "rank #" << world.rank() << ": the global max loglikelihood is assumed to be " << globallyOptimizedLoglikelihood << endl;
  // find the process that acheived the best nloglikelihood
  bool localEqualsGlobal = (globallyOptimizedLoglikelihood == rankOptimizedLoglikelihood);
  if(localEqualsGlobal) {
    cerr << "rank #" << world.rank() << ": i think i'm the one that found the best loglikelihood" << endl;
  }
  int bestRank = localEqualsGlobal? world.rank() : -1; // -1 means i don't know!
  mpi::all_reduce<int>(world, bestRank, bestRank, mpi::maximum<int>());
  assert(bestRank >= 0);
  cerr << "rank #" << world.rank() << ": i think the one that found the best loglikelihood is " << bestRank << endl;
  localEqualsGlobal = (bestRank == world.rank());

  // the process which found the best HMM parameters will do some work now
  if(localEqualsGlobal) {
    // persist hmm params
    if(persistHmmParams) {
      string finalParamsPrefix = outputFilenamePrefix + ".final";
      hmmModel.PersistParams(finalParamsPrefix);
    }
    
    // viterbi
    string labelsFilename = outputFilenamePrefix + ".labels";
    hmmModel.Label(textFilename, labelsFilename);
    cerr << "automatic labels can be found at " << labelsFilename << endl;

    // compare to gold standard
    if(goldLabelsFilename != "") {
      cerr << "======================================" << endl;
      cerr << "HMM model vs. gold standard tagging..." << endl;
      double vi = hmmModel.ComputeVariationOfInformation(labelsFilename, goldLabelsFilename);
      cerr << "done. \nvariation of information = " << vi << endl;
      double manyToOne = hmmModel.ComputeManyToOne(labelsFilename, goldLabelsFilename);
      cerr << "many-to-one = " << manyToOne << endl;
    }

    // now initialize the latentCrfPosTagger's theta parameters
    for(auto contextIter = latentCrfPosTagger.nLogThetaGivenOneLabel.params.begin(); 
	contextIter != latentCrfPosTagger.nLogThetaGivenOneLabel.params.end();
	contextIter++) {
      for(auto probIter = contextIter->second.begin(); probIter != contextIter->second.end(); probIter++) {
	probIter->second = hmmModel.nlogTheta[contextIter->first][probIter->second];
      }
    }
    
    // then initialize the "transition" latentCrfPosTagger's lambda parameters
    for(auto contextIter = hmmModel.nlogGamma.params.begin();
	contextIter != hmmModel.nlogGamma.params.end();
	contextIter++) {
      const int yIM1 = contextIter->first;
      for(auto probIter = contextIter->second.begin(); probIter != contextIter->second.end(); probIter++) {
	int yI = probIter->first;
	const double hmmNlogProb = probIter->second;

	FeatureId temp;
  temp.type = FeatureTemplate::LABEL_BIGRAM;
  temp.bigram.current = yI;
  temp.bigram.previous = yIM1;
	if(!latentCrfPosTagger.lambda->ParamExists(temp)) {
	  cerr << "parameter " << temp << " exists as a transition feature in the hmm model, but was not found in the latentCrfPosTagger." << endl;
	  cerr << "============================================" << endl;
	  cerr << "hmm params: " << endl;
	  hmmModel.nlogGamma.PrintParams();
	  cerr << "============================================" << endl;
	  cerr << "latentCrfPosTagger params: " << endl;
	  latentCrfPosTagger.lambda->PrintParams();
	  assert(false);
	}
	latentCrfPosTagger.lambda->UpdateParam(temp, hmmNlogProb);
      }
    }
  }
  return bestRank;
}

int main(int argc, char **argv) {  
  // feenableexcept(FE_INVALID | FE_OVERFLOW | FE_DIVBYZERO);

  // boost mpi initialization
  mpi::environment env(argc, argv);
  mpi::communicator world;

  /*// wait for gdb to attach
  if(world.rank() == 0) {
    int i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == i)
      ;
  }
  */

  // parse arguments
  if(world.rank() == 0) {
    cerr << "master" << world.rank() << ": parsing arguments...";
  }

  if(world.rank() == 0) {
    cerr << "done." << endl;
  }

  unsigned NUMBER_OF_LABELS = 12;
  unsigned FIRST_LABEL_ID = 4;

  // randomize draws
  int seed = time(NULL);
  if(world.rank() == 0) {
    cerr << "master" << world.rank() << ": executing srand(" << seed << ")" << endl;
  }
  srand(seed);

  // configurations
  if(world.rank() == 0) {
    cerr << "master" << world.rank() << ": setting configurations...";
  }
  LearningInfo learningInfo(&world, GetOutputPrefix(argc, argv));
  // general 
  learningInfo.debugLevel = DebugLevel::MINI_BATCH;
  learningInfo.useMaxIterationsCount = false;
  learningInfo.maxIterationsCount = 50;
  learningInfo.mpiWorld = &world;
  learningInfo.initializeLambdasWithGaussian = false;
  learningInfo.initializeLambdasWithZero = true;
  learningInfo.initializeLambdasWithOne = false;
  //  learningInfo.useMinLikelihoodDiff = true;
  //  learningInfo.minLikelihoodDiff = 10;
  learningInfo.useMinLikelihoodRelativeDiff = true;
  learningInfo.minLikelihoodRelativeDiff = 0.01;
  learningInfo.useSparseVectors = true;
  learningInfo.persistParamsAfterNIteration = 10;
  // block coordinate descent
  learningInfo.optimizationMethod.algorithm = OptAlgorithm::BLOCK_COORD_DESCENT;
  // lbfgs
  learningInfo.optimizationMethod.subOptMethod = new OptMethod();
  learningInfo.optimizationMethod.subOptMethod->algorithm = OptAlgorithm::LBFGS;
  learningInfo.optimizationMethod.subOptMethod->regularizer = Regularizer::L1;
  learningInfo.optimizationMethod.subOptMethod->regularizationStrength = 1.0;
  learningInfo.optimizationMethod.subOptMethod->miniBatchSize = 0;
  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxIterations = 4;
  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxEvalsPerIteration = 5;
  //  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.memoryBuffer = 50;
  //  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.precision = 0.00000000000000000000000001;
  learningInfo.optimizationMethod.subOptMethod->moveAwayPenalty = 0.0;
  learningInfo.retryLbfgsOnRoundingErrors = true;
  learningInfo.supervisedTraining = false;

  // thetas
  learningInfo.thetaOptMethod = new OptMethod();
  learningInfo.thetaOptMethod->algorithm = OptAlgorithm::EXPECTATION_MAXIMIZATION;
  //learningInfo.invokeCallbackFunctionEveryKIterations = 1;

  // parse command line arguments
  string textFilename, outputFilenamePrefix, goldLabelsFilename;
  ParseParameters(argc, argv, textFilename, outputFilenamePrefix, goldLabelsFilename, learningInfo);

  // initialize the model
  LatentCrfModel* model = LatentCrfPosTagger::GetInstance(textFilename, outputFilenamePrefix, learningInfo, NUMBER_OF_LABELS, FIRST_LABEL_ID);
  
  // hmm initialization
  unsigned bestRank = HmmInitialize(world, textFilename, outputFilenamePrefix, NUMBER_OF_LABELS, *((LatentCrfPosTagger*)model), FIRST_LABEL_ID, goldLabelsFilename);
  model->BroadcastTheta(bestRank);
  
  // sync all processes
  bool dummy = true;
  if(learningInfo.mpiWorld->rank() == 0) {
    mpi::gather<bool>(*learningInfo.mpiWorld, dummy, &dummy, 0);
  } else {
    mpi::gather<bool>(*learningInfo.mpiWorld, dummy, 0);
  }

  // use gold labels to do supervised training
  if(learningInfo.supervisedTraining) {
    model->SupervisedTrain(goldLabelsFilename);
    if(learningInfo.mpiWorld->rank() == 0) {
      model->PersistTheta(outputFilenamePrefix + ".supervised.theta");
      model->lambda->PersistParams(outputFilenamePrefix + ".supervised.lambda");
    }
  }

  // unsupervised training of the model
  if(world.rank() == 0) {
    cerr << "master" << world.rank() << ": train the model..." << endl;
  }
  model->Train();
  if(world.rank() == 0) {
    cerr << "training finished!" << endl;
  }
  
  // print best params
  if(learningInfo.mpiWorld->rank() == 0) {
    model->lambda->PersistParams(outputFilenamePrefix + string(".final.lambda"));
    model->PersistTheta(outputFilenamePrefix + string(".final.theta"));
  }

  // we don't need the slaves anymore
  if(world.rank() > 0) {
    return 0;
  }
    
  // compute some statistics on a test set
  cerr << "analyze the data using the trained model..." << endl;
  string analysisFilename = outputFilenamePrefix + ".analysis";
  model->Analyze(textFilename, analysisFilename);
  cerr << "analysis can be found at " << analysisFilename << endl;
  
  // viterbi
  string labelsFilename = outputFilenamePrefix + ".labels";
  model->Label(textFilename, labelsFilename);
  cerr << "automatic labels can be found at " << labelsFilename << endl;

  // compare to gold standard
  if(goldLabelsFilename != "") {
    cerr << "comparing to gold standard tagging..." << endl;
    double vi = model->ComputeVariationOfInformation(labelsFilename, goldLabelsFilename);
    cerr << "done. \nvariation of information = " << vi << endl;
    double manyToOne = model->ComputeManyToOne(labelsFilename, goldLabelsFilename);
    cerr << "many-to-one = " << manyToOne << endl ;
  }
}
