#include <fenv.h>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/thread/thread.hpp>
#include "LatentCrfModel.h"
#include "HmmModel2.h"

using namespace fst;
using namespace std;
namespace mpi = boost::mpi;

typedef ProductArc<FstUtils::LogWeight, FstUtils::LogWeight> ProductLogArc;

void ParseParameters(int argc, char **argv, string &textFilename, string &outputFilenamePrefix, string &goldLabelsFilename) {
  assert(argc >= 3);
  textFilename = argv[1];
  outputFilenamePrefix = argv[2];
  if(argc >= 4) {
    goldLabelsFilename = argv[3];
  } else {
    goldLabelsFilename = "";
  }
}

// returns the rank of the process which have found the best HMM parameters
unsigned HmmInitialize(mpi::communicator world, string textFilename, string outputFilenamePrefix, int NUMBER_OF_LABELS, LatentCrfModel &latentCrfModel, int FIRST_LABEL_ID, string goldLabelsFilename) {

  outputFilenamePrefix += ".hmm";

  // hmm initializer can't initialize the latent crf multinomials when zI dpeends on both y_{i-1} and y_i
  assert(latentCrfModel.learningInfo.zIDependsOnYIM1 == false);

  // configurations
  cerr << "rank #" << world.rank() << ": training the hmm model to initialize latentCrfModel parameters..." << endl;

  bool persistHmmParams = false;

  LearningInfo learningInfo;
  learningInfo.maxIterationsCount = 2;
  learningInfo.useMaxIterationsCount = true;
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

    // now initialize the latentCrfModel's theta parameters
    MultinomialParams::ConditionalMultinomialParam<int> nLogThetaGivenOneLabel;
    for(map<int, MultinomialParams::MultinomialParam>::iterator contextIter = latentCrfModel.nLogThetaGivenOneLabel.params.begin(); 
	contextIter != latentCrfModel.nLogThetaGivenOneLabel.params.end();
	contextIter++) {
      for(map<int, double>::iterator probIter = contextIter->second.begin(); probIter != contextIter->second.end(); probIter++) {
	probIter->second = hmmModel.nlogTheta[contextIter->first][probIter->second];
      }
    }
    
    // then initialize the "transition" latentCrfModel's lambda parameters
    for(map<int, MultinomialParams::MultinomialParam>::const_iterator contextIter = hmmModel.nlogGamma.params.begin();
	contextIter != hmmModel.nlogGamma.params.end();
	contextIter++) {
      const int yIM1 = contextIter->first;
      for(map<int, double>::const_iterator probIter = contextIter->second.begin(); probIter != contextIter->second.end(); probIter++) {
	int yI = probIter->first;
	const double hmmNlogProb = probIter->second;
	stringstream temp;
	temp << "F51:" << yIM1 << ":" << yI;
	if(!latentCrfModel.lambda->ParamExists(temp.str())) {
	  cerr << "parameter " << temp.str() << " exists as a transition feature in the hmm model, but was not found in the latentCrfModel." << endl;
	  cerr << "============================================" << endl;
	  cerr << "hmm params: " << endl;
	  hmmModel.nlogGamma.PrintParams();
	  cerr << "============================================" << endl;
	  cerr << "latentCrfModel params: " << endl;
	  latentCrfModel.lambda->PrintParams();
	  assert(false);
	}
	latentCrfModel.lambda->UpdateParam(temp.str(), hmmNlogProb);
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
  string textFilename, outputFilenamePrefix, goldLabelsFilename;
  ParseParameters(argc, argv, textFilename, outputFilenamePrefix, goldLabelsFilename);
  if(world.rank() == 0) {
    cerr << "done." << endl;
  }

  unsigned NUMBER_OF_LABELS = 2;
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
  LearningInfo learningInfo;
  // general 
  learningInfo.debugLevel = DebugLevel::MINI_BATCH;
  learningInfo.useMaxIterationsCount = false;
  learningInfo.maxIterationsCount = 50;
  learningInfo.mpiWorld = &world;
  //  learningInfo.useMinLikelihoodDiff = true;
  //  learningInfo.minLikelihoodDiff = 10;
  learningInfo.useMinLikelihoodRelativeDiff = true;
  learningInfo.minLikelihoodRelativeDiff = 0.01;
  learningInfo.useSparseVectors = true;
  learningInfo.zIDependsOnYIM1 = false;
  learningInfo.persistParamsAfterNIteration = 10;
  // block coordinate descent
  learningInfo.optimizationMethod.algorithm = OptAlgorithm::BLOCK_COORD_DESCENT;
  // lbfgs
  learningInfo.optimizationMethod.subOptMethod = new OptMethod();
  learningInfo.optimizationMethod.subOptMethod->algorithm = OptAlgorithm::LBFGS;
  learningInfo.optimizationMethod.subOptMethod->regularizer = Regularizer::L2;
  learningInfo.optimizationMethod.subOptMethod->regularizationStrength = 1.0;
  learningInfo.optimizationMethod.subOptMethod->miniBatchSize = 0;
  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxIterations = 20;
  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxEvalsPerIteration = 5;
  //  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.memoryBuffer = 50;
  //  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.precision = 0.00000000000000000000000001;
  learningInfo.optimizationMethod.subOptMethod->moveAwayPenalty = 0.0;
  learningInfo.retryLbfgsOnRoundingErrors = true;
  learningInfo.supervisedTraining = false;

  // add constraints
  learningInfo.constraints.clear();
  if(world.rank() == 0) {
    cerr << "done." << endl;
  }
  
  // initialize the model
  LatentCrfModel& model = LatentCrfModel::GetInstance(textFilename, outputFilenamePrefix, learningInfo, NUMBER_OF_LABELS, FIRST_LABEL_ID);
  
  // hmm initialization
  unsigned bestRank = HmmInitialize(world, textFilename, outputFilenamePrefix, NUMBER_OF_LABELS, model, FIRST_LABEL_ID, goldLabelsFilename);
  model.BroadcastTheta(bestRank);
  model.BroadcastLambdas(bestRank);

  // use gold labels to do supervised training
  if(learningInfo.supervisedTraining) {
    model.SupervisedTrain(goldLabelsFilename);
    if(learningInfo.mpiWorld->rank() == 0) {
      model.PersistTheta(outputFilenamePrefix + ".supervised.theta");
      model.lambda->PersistParams(outputFilenamePrefix + ".supervised.lambda");
    }
  }

  // unsupervised training of the model
  if(world.rank() == 0) {
    cerr << "master" << world.rank() << ": train the model..." << endl;
  }
  model.Train();
  if(world.rank() == 0) {
    cerr << "training finished!" << endl;
  }
  
  // we don't need the slaves anymore
  if(world.rank() > 0) {
    return 0;
  }
    
  // compute some statistics on a test set
  cerr << "analyze the data using the trained model..." << endl;
  string analysisFilename = outputFilenamePrefix + ".analysis";
  model.Analyze(textFilename, analysisFilename);
  cerr << "analysis can be found at " << analysisFilename << endl;
  
  // viterbi
  string labelsFilename = outputFilenamePrefix + ".labels";
  model.Label(textFilename, labelsFilename);
  cerr << "automatic labels can be found at " << labelsFilename << endl;

  // compare to gold standard
  if(goldLabelsFilename != "") {
    cerr << "comparing to gold standard tagging..." << endl;
    double vi = model.ComputeVariationOfInformation(labelsFilename, goldLabelsFilename);
    cerr << "done. \nvariation of information = " << vi << endl;
    double manyToOne = model.ComputeManyToOne(labelsFilename, goldLabelsFilename);
    cerr << "many-to-one = " << manyToOne << endl ;
  }
}
