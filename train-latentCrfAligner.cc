#include <fenv.h>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/thread/thread.hpp>

#include "LatentCrfAligner.h"
#include "IbmModel1.h"

using namespace fst;
using namespace std;
namespace mpi = boost::mpi;

typedef ProductArc<FstUtils::LogWeight, FstUtils::LogWeight> ProductLogArc;

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
void IbmModel1Initialize(mpi::communicator world, string textFilename, string outputFilenamePrefix, LatentCrfAligner &latentCrfAligner, string &NULL_SRC_TOKEN) {

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
  learningInfo.maxIterationsCount = 5;
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
  
  // now initialize the latentCrfAligner's theta parameters, and also augment the precomputed features with ibm model 1 features
  string ibm1PrecomputedFeatureId = "_ibm1";
  cerr << "rank #" << world.rank() << "now update the multinomial params of the latentCrfALigner model." << endl;
  for(map<int, MultinomialParams::MultinomialParam>::iterator contextIter = latentCrfAligner.nLogThetaGivenOneLabel.params.begin(); 
      contextIter != latentCrfAligner.nLogThetaGivenOneLabel.params.end();
      contextIter++) {
    
    for(map<int, double>::iterator probIter = contextIter->second.begin(); probIter != contextIter->second.end(); probIter++) {

      assert(ibmModel1.params[contextIter->first].count(probIter->first) > 0);
      probIter->second = ibmModel1.params[contextIter->first][probIter->first];
      latentCrfAligner->lambda->AddToPrecomputedFeaturesWith2Inputs(contextIter->first, probIter->first, ibm1PrecomputedFeatureId, probIter->second);
    }
  }
  
  cerr << "rank #" << world.rank() << "ibm model 1 initialization finished." << endl;
}

int main(int argc, char **argv) {  

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
  learningInfo.useMaxIterationsCount = true;
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
  
  // ibm model 1 initialization of theta params. also updates the lambda precomputed features by adding ibm model 1 probs
  IbmModel1Initialize(world, textFilename, outputFilenamePrefix, *((LatentCrfAligner*)model), ((LatentCrfAligner*)model)->NULL_TOKEN_STR);
  model->BroadcastTheta(0);
  model->BroadcastLambdas(0);

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
  ofstream labelsFile(labelsFilename.c_str());
  for(unsigned exampleId = 0; exampleId < model->examplesCount; ++exampleId) {
    std::vector<int> &srcSent = model->GetObservableContext(exampleId);
    std::vector<int> &tgtSent = model->GetObservableSequence(exampleId);
    std::vector<int> labels;
    // run viterbi
    ((LatentCrfAligner*)model)->Label(tgtSent, srcSent, labels);
    // 
    for(unsigned i = 0; i < labels.size(); ++i) {
      // dont write null alignments
      if(labels[i] == ((LatentCrfAligner*)model)->NULL_POSITION) {
	continue;
      }
      // determine the alignment (i.e. src position) for this tgt position (i)
      int alignment = labels[i] - ((LatentCrfAligner*)model)->FIRST_SRC_POSITION;
      assert(alignment >= 0);
      labelsFile << alignment << "-" << i << " ";
    }
    labelsFile << endl;
  }
  labelsFile.close();
  cerr << "alignments can be found at " << labelsFilename << endl;
}
