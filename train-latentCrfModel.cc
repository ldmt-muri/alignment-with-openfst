#include <fenv.h>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/thread/thread.hpp>
#include "LearningInfo.h"
#include "FstUtils.h"
#include "StringUtils.h"
#include "LatentCrfModel.h"

using namespace fst;
using namespace std;
namespace mpi = boost::mpi;

typedef ProductArc<LogWeight, LogWeight> ProductLogArc;

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

void DistributeTasks(int argc, char **argv, LatentCrfModel &model, mpi::communicator &world) {
  cerr << "TASK DISTRIBUTOR" << endl;
  // slave nodes stay here
  if(world.rank() > 0) {
    
    int sentId;
    mpi::request updateThetaRequest = world.irecv(0, LatentCrfModel::MPI_TAG_UPDATE_SLAVE_THETA, model.nLogTheta);
    vector<double> paramWeights;
    mpi::request updateLambdaRequest = world.irecv(0, LatentCrfModel::MPI_TAG_UPDATE_SLAVE_LAMBDA, paramWeights);
    vector<string> paramIds;
    mpi::request updateLambdaIdsRequest = world.irecv(0, LatentCrfModel::MPI_TAG_UPDATE_SLAVE_LAMBDA_IDS, paramIds);
    mpi::request computeThetaMleRequest = world.irecv(0, LatentCrfModel::MPI_TAG_COMPUTE_PARTIAL_THETA_MLE, sentId);     
    mpi::request dieRequest = world.irecv(0, LatentCrfModel::MPI_TAG_DIE);
    bool initialized = false;
    do {
      //      cerr << "slave#" << world.rank() << " is waiting for the master's command..." << endl;

      if(updateThetaRequest.test()) {
	// nLogTheta in the slave's model has been implicitly updated
	cerr << "UPDATE IRECEIVED\nslave#" << world.rank() << " has updated its theta params." << endl;
      	if(initialized) {
	  world.send(0, LatentCrfModel::MPI_TAG_ACK_THETA_UPDATED);
	  cerr << "ACK SENT\nslave#" << world.rank() << " sent a theta ack" << endl;
	} else {
	  initialized = true;
	}
	cerr << "environment::max_tag() = " << mpi::environment::max_tag() << endl;
	updateThetaRequest = world.irecv(0, LatentCrfModel::MPI_TAG_UPDATE_SLAVE_THETA, model.nLogTheta);
      } else if (updateLambdaRequest.test()) {
	// lambda->paramWeights in the slave's model has been implicitly updated
	cerr << "slave#" << world.rank() << " has updated its lambda params." << endl;
	world.send(0, LatentCrfModel::MPI_TAG_ACK_LAMBDA_UPDATED);
	// update the local params
	cerr << "model.lambda->paramWeights.size() = " << model.lambda->paramWeights.size() << endl;
	cerr << "model.lambda->paramIds.size() = " << model.lambda->paramIds.size() << endl;
	cerr << "model.lambda->paramIndexes.size() = " << model.lambda->paramIndexes.size() << endl;
	for(int index = 0; index < paramWeights.size(); index++) {
	  assert(index < model.lambda->GetParamsCount());
	  model.lambda->UpdateParam(index, paramWeights[index]);
	}
	updateLambdaRequest = world.irecv(0, LatentCrfModel::MPI_TAG_UPDATE_SLAVE_LAMBDA, paramWeights);
      } else if (updateLambdaIdsRequest.test()) {
	// lambda->paramIds in the slave's model has been implicitly updated
	cerr << "slave#" << world.rank() << " has updated its lambda ids." << endl;
	world.send(0, LatentCrfModel::MPI_TAG_ACK_LAMBDA_INDEXES_UPDATED);
	// add the local params
	for(int index = 0; index < paramIds.size(); index++) {
	  model.lambda->AddParam(paramIds[index]);
	  assert(index == model.lambda->GetParamIndex(paramIds[index]));
	}
	updateLambdaIdsRequest = world.irecv(0, LatentCrfModel::MPI_TAG_UPDATE_SLAVE_LAMBDA_IDS, paramIds);
      } else if (dieRequest.test()) {
	cerr << "slave#" << world.rank() << " is killing itself." << endl;
	exit(0);

      } else if (computeThetaMleRequest.test()) {
	cerr << "slave#" << world.rank() << " is computing partial mle from sentId " << sentId << endl;
	MultinomialParams::ConditionalMultinomialParam mle;
	map<int, double> mleMarginals;
	model.UpdateThetaMleForSent(sentId, mle, mleMarginals);
	world.send(0, LatentCrfModel::MPI_TAG_RETURN_PARTIAL_THETA_MLE, mle);
	world.send(0, LatentCrfModel::MPI_TAG_RETURN_PARTIAL_THETA_MLE_MARGINAL, mleMarginals);
	computeThetaMleRequest = world.irecv(0, LatentCrfModel::MPI_TAG_COMPUTE_PARTIAL_THETA_MLE, sentId);     
      } else {
	//	cerr << "slave#" << world.rank() << " will nap for a while" << endl;	
	boost::this_thread::sleep( boost::posix_time::seconds(1) );
      }
    } while(true);
  }
  // master node continues...
  std::cout << "I am the master." << std::endl;
}

int main(int argc, char **argv) {  
  // feenableexcept(FE_INVALID | FE_OVERFLOW | FE_DIVBYZERO);

  // boost mpi initialization
  mpi::environment env(argc, argv);
  mpi::communicator world;
  
  // parse arguments
  cerr << "parsing arguments...";
  string textFilename, outputFilenamePrefix, goldLabelsFilename;
  ParseParameters(argc, argv, textFilename, outputFilenamePrefix, goldLabelsFilename);
  cerr << "done." << endl;

  // randomize draws
  int seed = time(NULL);
  cerr << "executing srand(" << seed << ")" << endl;
  srand(seed);

  // configurations
  cerr << "setting configurations...";
  LearningInfo learningInfo;
  // general 
  learningInfo.debugLevel = DebugLevel::MINI_BATCH;
  learningInfo.useMaxIterationsCount = true;
  learningInfo.maxIterationsCount = 50;
  learningInfo.mpiWorld = &world;
  //  learningInfo.useMinLikelihoodDiff = true;
  //  learningInfo.minLikelihoodDiff = 10;
  learningInfo.useMinLikelihoodRelativeDiff = true;
  learningInfo.minLikelihoodRelativeDiff = 0.001;
  learningInfo.useSparseVectors = true;
  // block coordinate descent
  learningInfo.optimizationMethod.algorithm = OptAlgorithm::BLOCK_COORD_DESCENT;
  // lbfgs
  learningInfo.optimizationMethod.subOptMethod = new OptMethod();
  learningInfo.optimizationMethod.subOptMethod->algorithm = OptAlgorithm::LBFGS;
  learningInfo.optimizationMethod.subOptMethod->regularizer = Regularizer::NONE;
  learningInfo.optimizationMethod.subOptMethod->regularizationStrength = 0.1;
  learningInfo.optimizationMethod.subOptMethod->miniBatchSize = 0;
  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxIterations = 9;
  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.maxEvalsPerIteration = 7;
  //  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.memoryBuffer = 50;
  //  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.precision = 0.00000000000000000000000001;
  learningInfo.optimizationMethod.subOptMethod->lbfgsParams.l1 = (learningInfo.optimizationMethod.subOptMethod->regularizer == Regularizer::L1);
  learningInfo.optimizationMethod.subOptMethod->moveAwayPenalty = 0.0;

  // add constraints
  learningInfo.constraints.clear();
  if(world.rank() == 0) {
    cerr << "done." << endl;
  }
  
  // initialize the model
  LatentCrfModel& model = LatentCrfModel::GetInstance(textFilename, outputFilenamePrefix, learningInfo);

  // train the model
  cerr << "train the model..." << endl;
  model.Train();
  cerr << "rank #" << world.rank() << ": training finished!" << endl;
  
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
    cerr << "done. variation of information = " << vi << endl;
  }
}
