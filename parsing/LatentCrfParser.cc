#include "LatentCrfParser.h"

using Eigen::MatrixXlogd;
using Eigen::VectorXlogd;

string LatentCrfParser::ROOT_STR = "__ROOT__";
int64_t LatentCrfParser::ROOT_ID = -1000000;
int LatentCrfParser::ROOT_POSITION = -1;

// singleton
LatentCrfModel* LatentCrfParser::GetInstance(const string &textFilename, 
					     const string &outputPrefix, 
					     LearningInfo &learningInfo, 
					     const string &initialLambdaParamsFilename, 
					     const string &initialThetaParamsFilename,
					     const string &wordPairFeaturesFilename) {
  
  if(!instance) {
    instance = new LatentCrfParser(textFilename, 
                                    outputPrefix,
                                    learningInfo, 
                                    initialLambdaParamsFilename, 
                                    initialThetaParamsFilename,
                                    wordPairFeaturesFilename);
  }
  return instance;
}

LatentCrfModel* LatentCrfParser::GetInstance() {
  if(!instance) {
    assert(false);
  }

  return instance;
}

LatentCrfParser::LatentCrfParser(const string &textFilename,
				   const string &outputPrefix,
				   LearningInfo &learningInfo,
				   const string &initialLambdaParamsFilename, 
				   const string &initialThetaParamsFilename,
				   const string &wordPairFeaturesFilename) : LatentCrfModel("",
                                                                    outputPrefix,
                                                                    learningInfo,
                                                                    LatentCrfParser::ROOT_POSITION,
                                                                    LatentCrfParser::Task::DEPENDENCY_PARSING) {
  
  // slaves wait for master
  if(learningInfo.mpiWorld->rank() != 0) {
    bool vocabEncoderIsReady;
    boost::mpi::broadcast<bool>(*learningInfo.mpiWorld, vocabEncoderIsReady, 0);
  }
  // encode the null token which is conventionally added to the beginning of the sentnece. 
  ROOT_STR = "__ROOT__";
  ROOT_ID = vocabEncoder.Encode(ROOT_STR);
  assert(ROOT_ID != vocabEncoder.UnkInt());
  string minusOne = "-1";
  ROOT_DETAILS.details.push_back(0);
  ROOT_DETAILS.details.push_back(vocabEncoder.Encode(ROOT_STR));
  ROOT_DETAILS.details.push_back(vocabEncoder.Encode(ROOT_STR));
  ROOT_DETAILS.details.push_back(vocabEncoder.Encode(ROOT_STR));
  ROOT_DETAILS.details.push_back(vocabEncoder.Encode(ROOT_STR));
  ROOT_DETAILS.details.push_back(vocabEncoder.Encode(ROOT_STR));
  ROOT_DETAILS.details.push_back(vocabEncoder.Encode(minusOne));
  ROOT_DETAILS.details.push_back(vocabEncoder.Encode(ROOT_STR));
  ROOT_DETAILS.details.push_back(vocabEncoder.Encode(minusOne));
  ROOT_DETAILS.details.push_back(vocabEncoder.Encode(ROOT_STR));
  
  // read and encode data
  sents.clear();
  vocabEncoder.ReadConll(textFilename, sents, 1);
  assert(sents.size() > 0);
  examplesCount = sents.size();
  
  // fix learningInfo.firstKExamplesToLabel
  if(learningInfo.firstKExamplesToLabel <= 1) {
    learningInfo.firstKExamplesToLabel = examplesCount;
  }  
  
  if(learningInfo.mpiWorld->rank() == 0 && wordPairFeaturesFilename.size() > 0) {
    lambda->LoadPrecomputedFeaturesWith2Inputs(wordPairFeaturesFilename);
  }

  // master signals to slaves that he's done
  if(learningInfo.mpiWorld->rank() == 0) {
    bool vocabEncoderIsReady;
    boost::mpi::broadcast<bool>(*learningInfo.mpiWorld, vocabEncoderIsReady, 0);
  }

  // initialize (and normalize) the log theta params to gaussians
  // TODO-OPT: only the master process needs to init theta 
  if(learningInfo.initializeThetasWithGaussian || 
     learningInfo.initializeThetasWithUniform || 
     learningInfo.initializeThetasWithKleinManning) {
    InitTheta();
  }

  // TODO-OPT: only the master process should load the params file! this is a waste of time
  if(initialThetaParamsFilename.size() > 0 ) {
    //assert(nLogThetaGivenOneLabel.params.size() == 0);
    if(learningInfo.mpiWorld->rank() == 0) {
      cerr << "initializing theta params from " << initialThetaParamsFilename << endl;
    }
    MultinomialParams::LoadParams(initialThetaParamsFilename, nLogThetaGivenOneLabel, vocabEncoder, true, true);
    //string reloadedParamsFilename = initialThetaParamsFilename + ".reloaded";
    //MultinomialParams::PersistParams(reloadedParamsFilename, nLogThetaGivenOneLabel, vocabEncoder, true, true);
    assert(nLogThetaGivenOneLabel.params.size() > 0);
  } else {
    BroadcastTheta(0);
  }

  // load saved parameters
  if(initialLambdaParamsFilename.size() > 0) {
    lambda->LoadParams(initialLambdaParamsFilename);
    assert(lambda->paramWeightsTemp.size() == lambda->paramIndexes.size());
    assert(lambda->paramIdsTemp.size() == lambda->paramIndexes.size());
  }

  // add all features in this data set to lambda
  InitLambda();

  assert(lambda->paramWeightsTemp.size() == 0 && lambda->paramIdsTemp.size() == 0);
  assert(lambda->paramWeightsPtr->size() == lambda->paramIndexes.size());
  assert(lambda->paramIdsPtr->size() == lambda->paramIndexes.size());

  if(learningInfo.mpiWorld->rank() == 0) {
    vocabEncoder.PersistVocab(outputPrefix + string(".vocab"));
  }
}

void LatentCrfParser::InitTheta() {

  // some initializers have not been implemented yet
  assert(learningInfo.initializeThetasWithUniform || 
         learningInfo.initializeThetasWithGaussian || 
         learningInfo.initializeThetasWithModel1 ||
         learningInfo.initializeThetasWithKleinManning);
  assert(!learningInfo.initializeThetasWithModel1);
  
  if(learningInfo.mpiWorld->rank() == 0 && learningInfo.debugLevel >= DebugLevel::CORPUS) {
    cerr << "master" << learningInfo.mpiWorld->rank() << ": initializing thetas...";
    if(learningInfo.initializeThetasWithUniform) cerr << "with uniform" << endl;
    else if(learningInfo.initializeThetasWithGaussian) cerr << "with gaussian" << endl; 
    else if(learningInfo.initializeThetasWithModel1) cerr << "with model1" << endl;
    else if(learningInfo.initializeThetasWithKleinManning) cerr << "with klein&manning" << endl;
    else cerr << "EXACTLY ONE OF THETA INITIALIZERS MUST BE SELECTED. THIS SHOULD NEVER HAPPEN!" << endl;
  }
  
  assert(sents.size() > 0);

  // first initialize nlogthetas 
  nLogThetaGivenOneLabel.params.clear();
  for(unsigned sentId = 0; sentId < sents.size(); ++sentId) {
    vector<ObservationDetails> &sent = sents[sentId];
    vector<ObservationDetails> &reconstructedSent = sents[sentId];
    
    // klein and manning constant 
    vector<double> kleinManningConstant;
    for(unsigned i = 0; i < sent.size(); ++i) {
      kleinManningConstant.push_back(0.0);
      for(unsigned j = 0; j < sent.size(); ++j) {
        if(i == j) continue;
        // klein and manning initialization
        kleinManningConstant[i] += 1.0 / abs(i-j);
        //kleinManningConstant[i] += abs(i-j);
      }
      // klein and manning initialization
      kleinManningConstant[i] /= (1-1.0/sent.size());
      //kleinManningConstant[i] = (1-1.0/sent.size()) / kleinManningConstant[i];
    }
    for(int i = 0; i < (signed)sent.size(); ++i) {
      // first, initialize root selection theta parameters
      int64_t childReconstructed = sent[i].details[learningInfo.oneBasedConllFieldIdReconstructed-1];
      if(learningInfo.initializeThetasWithKleinManning) { 
        if(nLogThetaGivenOneLabel.params.find(ROOT_ID) == nLogThetaGivenOneLabel.params.end() ||
           nLogThetaGivenOneLabel[ROOT_ID].find(childReconstructed) == nLogThetaGivenOneLabel[ROOT_ID].end()) {
          nLogThetaGivenOneLabel[ROOT_ID][childReconstructed] = 0.0;
        }
        nLogThetaGivenOneLabel[ROOT_ID][childReconstructed] += 1.0 / sent.size();
      } else {
        nLogThetaGivenOneLabel.params[ROOT_ID][childReconstructed] = 
          learningInfo.initializeThetasWithGaussian?
          abs(gaussianSampler.Draw()) : 1.0;
      }
      // now, initialize internal attachments
      auto parentToken = sent[i];
      for(int j = 0; j < (signed)reconstructedSent.size(); ++j) {
        if(i == j) { continue; }
        int64_t parentConditioned = parentToken.details[learningInfo.oneBasedConllFieldIdConditioned-1];
        auto childToken = reconstructedSent[j];
        int64_t decision = childToken.details[learningInfo.oneBasedConllFieldIdReconstructed-1];
        if(i < j && learningInfo.generateChildConditionalOnDirection) {
          parentConditioned *= -1;
        } else if(i < j && learningInfo.generateChildAndDirection) {
          decision *= -1;
        }
        
        if(learningInfo.initializeThetasWithKleinManning) {
          if(nLogThetaGivenOneLabel.params.find(parentConditioned) == nLogThetaGivenOneLabel.params.end() || 
             nLogThetaGivenOneLabel.params[parentConditioned].find(decision) == nLogThetaGivenOneLabel.params[parentConditioned].end()) {
            nLogThetaGivenOneLabel.params[parentConditioned][decision] = 0.0;
          }
          double increment = 1.0 / abs(i-j); // / kleinManningConstant[j];
          nLogThetaGivenOneLabel.params[parentConditioned][decision] += increment;
        } else {
          nLogThetaGivenOneLabel.params[parentConditioned][decision] = 
            learningInfo.initializeThetasWithGaussian? abs(gaussianSampler.Draw()):
            1.0;
        }
      }
    }
  }
  
  // then normalize them
  if(learningInfo.initializeThetasWithKleinManning) {
    MultinomialParams::NormalizeParams(nLogThetaGivenOneLabel, 1.0, false, true, false);
  } else {
    MultinomialParams::NormalizeParams(nLogThetaGivenOneLabel, 1.0, true, true, false);
  }
  
  //stringstream thetaParamsFilename;
  //thetaParamsFilename << outputPrefix << ".init.theta";
  //PersistTheta(thetaParamsFilename.str());

  if(learningInfo.mpiWorld->rank() == 0) {
    cerr << "done" << endl;
  }
}

void LatentCrfParser::PrepareExample(unsigned exampleId) {  
}

vector<ObservationDetails>& LatentCrfParser::GetReconstructedObservableDetailsSequence(int exampleId) {
  if(testingMode) {
    return testSents[exampleId];
  } else {
    // refactor: this following line does not logically belong here
    lambda->learningInfo->currentSentId = exampleId;

    assert((unsigned)exampleId < sents.size());
    return sents[exampleId];
  }
}

void LatentCrfParser::Label(std::vector<int64_t> &tokens, std::vector<int> &labels) { 
  throw std::runtime_error("void LatentCrfParser::Label(std::vector<int64_t> &tokens, std::vector<int> &labels) is not implemented"); 
}

vector<int64_t>& LatentCrfParser::GetObservableSequence(int exampleId) { 
  throw std::runtime_error("vector<int64_t>& LatentCrfParser::GetObservableSequence(int exampleId) is not implemented");
}

vector<int64_t>& LatentCrfParser::GetObservableContext(int exampleId) { 
  throw std::runtime_error("vector<int64_t>& LatentCrfParser::GetObservableContext(int exampleId) is not implemented");
}

vector<int64_t>& LatentCrfParser::GetReconstructedObservableSequence(int exampleId) { 
  throw std::runtime_error("vector<int64_t>& LatentCrfParser::GetReconstructedObservableSequence(int exampleId) is not implemented");
}

vector<ObservationDetails>& LatentCrfParser::GetObservableDetailsSequence(int exampleId) {
  if(testingMode) {
    assert((unsigned)exampleId < testSents.size());
    return testSents[exampleId];
  } else {
    lambda->learningInfo->currentSentId = exampleId;
    assert((unsigned)exampleId < sents.size());
    return sents[exampleId];
  }
}

void LatentCrfParser::SetTestExample(vector<ObservationDetails> &sent) {
  testSents.clear();
  testSents.push_back(sent);
}

void LatentCrfParser::Label(vector<ObservationDetails> &tokens, vector<int> &labels) {

  // set up
  assert(labels.size() == 0); 
  assert(tokens.size() > 0);
  testingMode = true;
  SetTestExample(tokens);

  // do the actual labeling
  double treeLogProb;
  labels = GetViterbiParse(0, !learningInfo.testWithCrfOnly, treeLogProb);

  // set down ;)
  testingMode = false;

  assert(labels.size() == tokens.size());  
}

void LatentCrfParser::Label(const string &labelsFilename) {
  ofstream labelsFile;
  if(learningInfo.mpiWorld->rank() == 0) {
    labelsFile.open(labelsFilename.c_str());
  }
  assert(learningInfo.firstKExamplesToLabel <= examplesCount);
  double llOneBest = 0.0;
  for(unsigned exampleId = 0; exampleId < learningInfo.firstKExamplesToLabel; ++exampleId) {
    lambda->learningInfo->currentSentId = exampleId;
    if(exampleId % learningInfo.mpiWorld->size() != (unsigned)learningInfo.mpiWorld->rank()) {
      if(learningInfo.mpiWorld->rank() == 0) {
        string labelSequence;
        mpi::broadcast<string>(*learningInfo.mpiWorld, labelSequence, exampleId % learningInfo.mpiWorld->size());
        labelsFile << labelSequence;
      }
      continue;
    }
    
    double treeLogProb;
    std::vector<int> labels = GetViterbiParse(exampleId, !learningInfo.testWithCrfOnly, treeLogProb);
    llOneBest += treeLogProb;
    auto tokens = GetObservableDetailsSequence(exampleId);
    assert(labels.size() == tokens.size());
    
    stringstream ss;
    for(unsigned i = 0; i < labels.size(); ++i) {
      // conll token index is one based 
      int parent = labels[i] + 1;
      ss << tokens[i].details[ObservationDetailsHeader::ID] << "\t";
      ss << vocabEncoder.Decode(tokens[i].details[ObservationDetailsHeader::FORM]) << "\t";
      ss << vocabEncoder.Decode(tokens[i].details[ObservationDetailsHeader::LEMMA]) << "\t";
      ss << vocabEncoder.Decode(tokens[i].details[ObservationDetailsHeader::CPOSTAG]) << "\t";
      ss << vocabEncoder.Decode(tokens[i].details[ObservationDetailsHeader::POSTAG]) << "\t";
      ss << vocabEncoder.Decode(tokens[i].details[ObservationDetailsHeader::FEATS]) << "\t";
      ss << parent << "\t";
      ss << "_" << "\t";
      ss << "_" << "\t";                        \
      ss << "_" << endl;
    }
    ss << endl;
    if(learningInfo.mpiWorld->rank() == 0){
      labelsFile << ss.str();
    } else {
      string labelSequence(ss.str());
      mpi::broadcast<string>(*learningInfo.mpiWorld, labelSequence, exampleId % learningInfo.mpiWorld->size());
    }
  }
  if(learningInfo.mpiWorld->rank() == 0) {
    labelsFile.close();
  }
  // all slaves need to consume the broadcasted strings
  for(unsigned exampleId = 0; exampleId < learningInfo.firstKExamplesToLabel; ++exampleId) {
    lambda->learningInfo->currentSentId = exampleId;
    if(exampleId % learningInfo.mpiWorld->size() != (unsigned)learningInfo.mpiWorld->rank() && 
       exampleId % learningInfo.mpiWorld->size() != 0 && 
       learningInfo.mpiWorld->rank() != 0) {
      string dummy;
      mpi::broadcast<string>(*learningInfo.mpiWorld, dummy, exampleId % learningInfo.mpiWorld->size());
    }
  }
  double llOneBestTotal = 0.0;
  mpi::reduce<double>(*learningInfo.mpiWorld, llOneBest, llOneBestTotal, std::plus<double>(), 0);
  if(learningInfo.mpiWorld->rank() == 0) {
    cerr << "labels can be found at " << labelsFilename << endl;
    cerr << "log likelihood of the one best parses = " << llOneBest << endl;
  }
}

// convenience function
void LatentCrfParser::FireFeatures(const unsigned sentId,
                                  FastSparseVector<double> &h) {
  auto tokens = GetObservableDetailsSequence(sentId);
  if(tokens.size() > learningInfo.maxSequenceLength) {
    return;
  }
  MatrixXlogd adjacency, laplacianHat, laplacianHatInverse;
  VectorXlogd rootSelection;
  LogValD laplacianHatDeterminant;
  BuildMatrices(sentId, rootSelection, adjacency, laplacianHat,
                laplacianHatInverse, laplacianHatDeterminant, false);
}

// build the matrixes which can be used to marginalize proper dependency trees as of Koo et al 2007.
void LatentCrfParser::BuildMatrices(const unsigned sentId,
                                    VectorXlogd &rootScores,
                                    MatrixXlogd &adjacency,
                                    MatrixXlogd &laplacianHat,
                                    MatrixXlogd &laplacianHatInverse,
                                    LogValD &laplacianHatDeterminant,
                                    bool conditionOnZ) {
 
  // build A_{y|x} matrix and use matrix tree theoerem to compute Z(x), \sum_{y: (h,m) \in y} p(y|x)
  auto tokens = GetObservableDetailsSequence(sentId);
  auto reconstructedTokens = GetReconstructedObservableDetailsSequence(sentId);
  assert(tokens.size() == reconstructedTokens.size());
  // adjacency matrix A(y|x) in (Koo et al. 2007)
  double exponentSum = 0.0;
  adjacency.resize(tokens.size(), tokens.size());
  for(unsigned headPosition = 0; headPosition < tokens.size(); ++headPosition) {
    for(unsigned childPosition = 0; childPosition < tokens.size(); ++childPosition) {
      int64_t decision = reconstructedTokens[childPosition].details[learningInfo.oneBasedConllFieldIdReconstructed-1];
      if(headPosition < childPosition && learningInfo.generateChildAndDirection) {
        decision *= -1;
      }
      int64_t headConditioned = reconstructedTokens[headPosition].details[learningInfo.oneBasedConllFieldIdConditioned-1];
      if(headPosition < childPosition && learningInfo.generateChildConditionalOnDirection) {
        headConditioned *= -1;
      }
      double nlogTheta = nLogThetaGivenOneLabel[headConditioned][decision];
      double multinomialTerm = conditionOnZ? nlogTheta : 0.0;
      FastSparseVector<double> activeFeatures;
      if(headPosition != childPosition) {
        lambda->FireFeatures(tokens[headPosition], tokens[childPosition], tokens, activeFeatures);
      }
      double exponent = 
        headPosition == childPosition?
        - MultinomialParams::NLOG_ZERO:
        - multinomialTerm + lambda->DotProduct( activeFeatures );
      adjacency(headPosition, childPosition) = LogValD(exponent, false);
      if(headPosition != childPosition) {
        exponentSum += exponent;
      }
      
      if(std::isnan(adjacency(headPosition, childPosition).as_float()) || std::isinf(adjacency(headPosition, childPosition).as_float())) {
        cerr << "ERROR: adjacency(" << headPosition << ", " << childPosition << ") = " << adjacency(headPosition, childPosition) << endl;
        cerr << "       LogValD(- multinomialTerm + lambda->DotProduct( activeFeatures ), false)" << LogValD(- multinomialTerm + lambda->DotProduct( activeFeatures ), false) << endl;
        cerr << "       multinomialTerm = " << multinomialTerm << endl;
        cerr << "       lambda->DotProduct(activeFeatures) = " << lambda->DotProduct(activeFeatures) << endl;
        assert(false);
      }
    }
  }
  // root selection scores r(y|x) in (Koo et al. 2007)
  rootScores.resize(tokens.size());
  int64_t conditionedRoot = LatentCrfParser::ROOT_DETAILS.details[learningInfo.oneBasedConllFieldIdConditioned-1];
  for(unsigned rootPosition = 0; rootPosition < tokens.size(); ++rootPosition) {
    int64_t decision = reconstructedTokens[rootPosition].details[learningInfo.oneBasedConllFieldIdReconstructed-1];
    double multinomialTerm = conditionOnZ? 
      nLogThetaGivenOneLabel[conditionedRoot][decision]: 
      0.0;
    FastSparseVector<double> activeFeatures;
    lambda->FireFeatures(LatentCrfParser::ROOT_DETAILS, tokens[rootPosition], tokens, activeFeatures);
    rootScores(rootPosition) = LogValD(- multinomialTerm + lambda->DotProduct( activeFeatures ), false);
    if(std::isnan(rootScores(rootPosition).as_float())) {
      cerr << "ERROR: rootScores(" << rootPosition << ") = " << rootScores(rootPosition) << endl;
      assert(false);
    }
  }
  // laplacian matrix L(y|x) in (Koo et al. 2007)
  laplacianHat = -1.0 * adjacency;
  for(unsigned colIndex = 0; colIndex < tokens.size(); ++colIndex) {
    laplacianHat(colIndex, colIndex) = adjacency.col(colIndex).array().sum();
  }

  // modified laplacian matrix to allow for O(n^3) inference; \hat{L}(y|x) in (Koo et al. 2007)
  laplacianHat.row(0) = rootScores;

  // now, we divide all elements in laplacianHat by a constant to ensure numerical stability
  // while computing the determinant and inverse
  //double scalingConstant = max(laplacianHat.maxCoeff().as_float(), fabs(laplacianHat.minCoeff().as_float()));
  double scalingConstant = 
    tokens.size() == 1? 1.0 : MultinomialParams::nExp(- exponentSum / tokens.size() / (tokens.size() - 1));
  if(std::isinf(scalingConstant)) {
    cerr << "WARNING: scalingConstant = " << scalingConstant << endl; 
    cerr << "         reset scalingConstant = 1.0" << endl;
    scalingConstant = 1.0;
  } else if(scalingConstant <= 0.0 || std::isnan(scalingConstant)) {
    cerr << "ERROR: scalingConstant = " << scalingConstant << " = exp " << laplacianHat.maxCoeff() << " <= 0.0" << endl;
    assert(false);
  }
  
  //  cerr << endl << "DEBUG: scalingConstant = " << scalingConstant << " = exp " << laplacianHat.maxCoeff();
  assert(scalingConstant > 0.0);
  laplacianHat /= scalingConstant;
  
  // now compute determinant
  laplacianHatDeterminant = laplacianHat.determinant();
  for(unsigned i = 0; i < laplacianHat.rows(); ++i) {
    laplacianHatDeterminant *= LogValD(scalingConstant);
  }
  
  // then compute inverse
  laplacianHatInverse = laplacianHat.inverse();
  laplacianHatInverse /= scalingConstant;

  // now obtain the original unscaled laplacianHat
  laplacianHat *= scalingConstant;

}

void LatentCrfParser::SupervisedTrainTheta() {

  // first, clear all parameters to zeros
  for ( auto & context : nLogThetaGivenOneLabel.params) {
    for (auto & decision : context.second) {
      decision.second = 0.0;
    }
  }
  
  for(unsigned sentId = 0; sentId < examplesCount; ++sentId) {
    auto tokens = GetObservableDetailsSequence(sentId);
    for(int childIndex = 0; childIndex < (int) tokens.size(); ++childIndex) {
      int zeroBasedHeadIndex = (int) tokens[childIndex].details[ObservationDetailsHeader::HEAD] - 1;
      int64_t childThetaKey = tokens[childIndex].details[learningInfo.oneBasedConllFieldIdReconstructed-1];
      if(zeroBasedHeadIndex != -1 && 
         zeroBasedHeadIndex < childIndex && 
         learningInfo.generateChildAndDirection) {
        childThetaKey *= -1;
      }
      int64_t headThetaKey = 
        zeroBasedHeadIndex < 0?
        LatentCrfParser::ROOT_ID:
        tokens[zeroBasedHeadIndex].details[learningInfo.oneBasedConllFieldIdConditioned-1];
      if(zeroBasedHeadIndex != -1 &&
         zeroBasedHeadIndex < childIndex &&
         learningInfo.generateChildConditionalOnDirection) {
        headThetaKey *= -1;
      }
      assert( nLogThetaGivenOneLabel.params.find(headThetaKey) != nLogThetaGivenOneLabel.params.end());
      assert( nLogThetaGivenOneLabel.params[headThetaKey].find(childThetaKey) != nLogThetaGivenOneLabel.params[headThetaKey].end());
      nLogThetaGivenOneLabel.params[headThetaKey][childThetaKey]++;
    }
  }
  
  bool unnormalizedParamsAreInNLog = false;
  bool normalizedParamsAreInNLog = true;
  MultinomialParams::NormalizeParams(nLogThetaGivenOneLabel, learningInfo.multinomialSymmetricDirichletAlpha, unnormalizedParamsAreInNLog, normalizedParamsAreInNLog, learningInfo.variationalInferenceOfMultinomials);
  
}

// returns -log p(z|x)
double LatentCrfParser::UpdateThetaMleForSent(const unsigned sentId, 
  				     MultinomialParams::ConditionalMultinomialParam<int64_t> &mle, 
  				     boost::unordered_map<int64_t, double> &mleMarginals) {

  assert(sentId < examplesCount);

  // prune long sequences
  if( learningInfo.maxSequenceLength > 0 && 
      GetObservableDetailsSequence(sentId).size() > learningInfo.maxSequenceLength ) {
    return 0.0;
  }
  
  // build A_{y|x} matrix and use matrix tree theoerem to compute Z(x), \sum_{y: (h,m) \in y} p(y|x)
  // TODO: WE DON'T REALLY NEED THIS MATRIX; AFTERALL, COMPUTING LIKELIHOOD IN EM IS ONLY USEFUL FOR
  // DEBUGGING PURPOSES
  MatrixXlogd yGivenXAdjacency, yGivenXLaplacianHat, yGivenXLaplacianHatInverse;
  VectorXlogd yGivenXRootSelection;
  LogValD yGivenXLaplacianHatDeterminant;
  BuildMatrices(sentId, yGivenXRootSelection, yGivenXAdjacency, yGivenXLaplacianHat, 
                yGivenXLaplacianHatInverse, yGivenXLaplacianHatDeterminant, false);
  LogValD Z = yGivenXLaplacianHatDeterminant;

  // build A_{y|x,z} matrix and use matrix tree theorem to compute C(x), marginal(h,m;y|z,x)=\sum_{y:(h,m)\in y} p(y|x,z)
  MatrixXlogd yGivenXZAdjacency, yGivenXZLaplacianHat, yGivenXZLaplacianHatInverse;
  VectorXlogd yGivenXZRootSelection;
  LogValD yGivenXZLaplacianHatDeterminant = -1;
  BuildMatrices(sentId, yGivenXZRootSelection, yGivenXZAdjacency, yGivenXZLaplacianHat, 
                yGivenXZLaplacianHatInverse, yGivenXZLaplacianHatDeterminant, true);
  LogValD C = yGivenXZLaplacianHatDeterminant;
  if(C > Z) {
    cerr << "C = " << C << endl;
    cerr << "Z = " << Z << endl;
  }
  assert(C <= Z);

  if(std::isinf(C.v_) || C.as_float() == 0.0 || C.as_float() == -0.0) {
    cerr << "WARNING: C = " << C << ". will just skip this sentence." << endl;
    return 0.0;
  }

  auto reconstructedTokens = GetReconstructedObservableDetailsSequence(sentId);
  
  // for (h,m) \in \cal{T}_{np}^s: mle[h][m] += nLogThetaGivenOneLabel[h][m] * marginal(h,m;y|z,x)
  for(unsigned rootPosition = 0; rootPosition < yGivenXZLaplacianHat.rows(); ++rootPosition) {
    // marginal probability of making this decision; \mu_{0,m} in (Koo et al. 2007)
    double marginal = (yGivenXZLaplacianHat(0,rootPosition) * yGivenXZLaplacianHatInverse(rootPosition,0)).as_float();
    if(marginal > 1.01 || marginal < -0.01) {
      cerr << "ERROR: marginal = " << marginal << endl;
      cerr << "                = exp " << yGivenXZLaplacianHat(0,rootPosition) << " * exp " << yGivenXZLaplacianHatInverse(rootPosition,0) << endl;
      cerr << "       yGivenXZLaplacianHatDeterminant = " << yGivenXZLaplacianHatDeterminant << endl;
      cerr << "       yGivenXLaplacianHatDeterminant = " << yGivenXLaplacianHatDeterminant << endl;
      cerr << "       yGivenXZLaplacianHat = " << endl;
      for(unsigned headPosition = 0; headPosition < yGivenXZLaplacianHat.rows(); ++headPosition) {
        for(unsigned childPosition = 0; childPosition < yGivenXZLaplacianHat.cols(); ++childPosition) {
          cerr << yGivenXZLaplacianHat(headPosition, childPosition) << " ";
        }
        cerr << endl;
      }
      cerr << "       yGivenXZLaplacianHatInverse = " << endl;
      for(unsigned headPosition = 0; headPosition < yGivenXZLaplacianHat.rows(); ++headPosition) {
        for(unsigned childPosition = 0; childPosition < yGivenXZLaplacianHat.cols(); ++childPosition) {
          cerr << yGivenXZLaplacianHatInverse(headPosition, childPosition) << " ";
        }
        cerr << endl;
      }
      assert(false);
    }
    int64_t decision = reconstructedTokens[rootPosition].details[learningInfo.oneBasedConllFieldIdReconstructed-1];
    mle[LatentCrfParser::ROOT_ID][decision] += marginal;
    mleMarginals[LatentCrfParser::ROOT_ID] += marginal;
  }
  for(unsigned headPosition = 0; headPosition < yGivenXZLaplacianHat.rows(); ++headPosition) {
    for(unsigned childPosition = 0; childPosition < yGivenXZLaplacianHat.cols(); ++childPosition) {
      if(headPosition == childPosition) { continue; }
      double marginal = childPosition == 0? 0.0 :
        (yGivenXZAdjacency(headPosition, childPosition) * yGivenXZLaplacianHatInverse(childPosition, childPosition)).as_float();
      marginal -= headPosition == 0? 0.0 :
        (yGivenXZAdjacency(headPosition, childPosition) * yGivenXZLaplacianHatInverse(childPosition, headPosition)).as_float();
      if(marginal > 1.01 || marginal < -0.01) {
        cerr << "WARNING: marginal = " << marginal << endl;
      }
    
      int64_t decision = reconstructedTokens[childPosition].details[learningInfo.oneBasedConllFieldIdReconstructed-1];
      if(headPosition < childPosition && learningInfo.generateChildAndDirection) {
        decision *= -1;
      }
      int64_t parentConditioned = reconstructedTokens[headPosition].details[learningInfo.oneBasedConllFieldIdConditioned-1];
      if(headPosition < childPosition && learningInfo.generateChildConditionalOnDirection) {
        parentConditioned *= -1;
      }
      mle[parentConditioned][decision] += marginal;
      mleMarginals[parentConditioned] += marginal;      
    }
  }

  // nlog p(z|x)
  assert(C.as_float() >= 0 && Z.as_float() >= 0);
  double nLogC = -log(C), nLogZ = -log(Z);
  //cerr << "sent " << sentId << ": -log p(z|x) = " << nLogC << " - " << nLogZ << " = " << nLogC - nLogZ << endl;
  return nLogC - nLogZ;
}

// -loglikelihood is the return value
double LatentCrfParser::ComputeNllZGivenXAndLambdaGradient(
							  vector<double> &derivativeWRTLambda, int fromSentId, int toSentId, double *devSetNll) {
  
  double objective = 0;

  assert(!learningInfo.fixPosteriorExpectationsAccordingToPZGivenXWhileOptimizingLambdas);
  assert(derivativeWRTLambda.size() == lambda->GetParamsCount());
  
  // for each training example
  for(int sentId = fromSentId; sentId < toSentId; sentId++) {
    
    // sentId is assigned to the process with rank = sentId % world.size()
    if(sentId % learningInfo.mpiWorld->size() != learningInfo.mpiWorld->rank()) {
      continue;
    }
    
    // prune long sequences
    if( learningInfo.maxSequenceLength > 0 &&
        GetObservableDetailsSequence(sentId).size() > learningInfo.maxSequenceLength ) {
      continue;
    }

    // build A_{y|x} matrix and use matrix tree theoerem to compute Z(x), \sum_{y: (h,m) \in y} p(y|x)
    MatrixXlogd yGivenXAdjacency, yGivenXLaplacianHat, yGivenXLaplacianHatInverse;
    VectorXlogd yGivenXRootSelection;
    LogValD yGivenXLaplacianHatDeterminant;
    BuildMatrices(sentId, yGivenXRootSelection, yGivenXAdjacency, yGivenXLaplacianHat, 
                  yGivenXLaplacianHatInverse, yGivenXLaplacianHatDeterminant, false);
    LogValD Z = yGivenXLaplacianHatDeterminant;
    
    // build A_{y|x,z} matrix and use matrix tree theorem to compute C(x), marginal(h,m;y|z,x)=\sum_{y:(h,m)\in y} p(y|x,z)
    MatrixXlogd yGivenXZAdjacency, yGivenXZLaplacianHat, yGivenXZLaplacianHatInverse;
    VectorXlogd yGivenXZRootSelection;
    LogValD yGivenXZLaplacianHatDeterminant;
    BuildMatrices(sentId, yGivenXZRootSelection, yGivenXZAdjacency, yGivenXZLaplacianHat, 
                  yGivenXZLaplacianHatInverse, yGivenXZLaplacianHatDeterminant, true);
    if(yGivenXZLaplacianHatDeterminant.as_float() <= 0.0) {
      cerr << "Warning: yGivenXZLaplacianHatDeterminant = " << yGivenXZLaplacianHatDeterminant.as_float() << endl;
    }
    LogValD C = yGivenXZLaplacianHatDeterminant;
    if(C > Z) {
      cerr << "ERROR: C = " << C << " > Z = " << Z << endl;
    }
    assert(C <= Z);
    // update the loglikelihood
    assert(!learningInfo.useEarlyStopping);
    double nLogC = -log(C), nLogZ = -log(Z);
    // keep an eye on bad numbers
    if(std::isinf(nLogZ) || std::isinf(nLogC) || std::isnan(nLogZ) || std::isnan(nLogC)) {
      cerr << "ERROR: nLogZ = " << nLogZ << ", nLogC = " << nLogC << ". ";
      cerr << "       Z = " << Z << ", C = " << C << ". ";
      cerr << "I will just ignore this sentence ..." << endl;
      continue;
    }

    if(nLogC < nLogZ) {
      cerr << "this must be a bug. nLogC always be >= nLogZ. " << endl;
      cerr << "nLogC = " << nLogC << endl;
      cerr << "nLogZ = " << nLogZ << endl;
    }
    
    //cerr << "sent " << sentId << ": -log p(z|x) = " << nLogC << " - " << nLogZ << " = " << nLogC - nLogZ << endl;
    objective += nLogC;
    objective -= nLogZ;
    //cerr << "sent " << sentId << ": nLogC - nLogZ = nLog(" << C << ") - nLog(" << Z << ") = " << nLogC << " - " << nLogZ << " = " << nLogC - nLogZ << endl;
    
    auto tokens = GetObservableDetailsSequence(sentId);
    
    // debug
    stringstream marginalGivenXSS, marginalGivenXZSS;
    
    // for (h,m) in \cal{T}_{np}^s:
    //   for k in f(x,h,m):
    //     dll/d\lambda_k += f_k(x,h,m) * [marginal(h,m;y|z,x)-marginal(h,m;y|x)]
    for(unsigned headPosition = 0; headPosition < tokens.size(); ++headPosition) {
      marginalGivenXZSS << headPosition << "\t";
      marginalGivenXSS << headPosition << "\t";
      for(unsigned childPosition = 0; childPosition < tokens.size(); ++childPosition) {
        FastSparseVector<double> activeFeatures;
        if(headPosition != childPosition) {
          lambda->FireFeatures(tokens[headPosition], tokens[childPosition], tokens, activeFeatures);
        }
        double marginalGivenXZ = childPosition == 0? 0.0 :
          (yGivenXZAdjacency(headPosition, childPosition) * 
           yGivenXZLaplacianHatInverse(childPosition, childPosition)).as_float();
        marginalGivenXZ -= headPosition == 0? 0.0 :
          (yGivenXZAdjacency(headPosition, childPosition) * 
           yGivenXZLaplacianHatInverse(childPosition, headPosition)).as_float();
        double marginalGivenX = childPosition == 0? 0.0 :
          (yGivenXAdjacency(headPosition, childPosition) * 
           yGivenXLaplacianHatInverse(childPosition, childPosition)).as_float();
        marginalGivenX -= headPosition == 0? 0.0 :
          (yGivenXAdjacency(headPosition, childPosition) * 
           yGivenXLaplacianHatInverse(childPosition, headPosition)).as_float();
        
        marginalGivenXZSS << std::setprecision(1) << marginalGivenXZ << "\t";
        
        marginalGivenXSS << std::setprecision(1) << marginalGivenX << "\t";
        
        double marginalDiff = marginalGivenX - marginalGivenXZ;
        for(auto featIter = activeFeatures.begin(); featIter != activeFeatures.end(); ++featIter) {
          derivativeWRTLambda[featIter->first] += featIter->second * marginalDiff;
        }
      }
      marginalGivenXZSS << headPosition << "\n";
      marginalGivenXSS << headPosition << "\n";
    }
    
    // debug
    stringstream headerSS;
    headerSS <<  "sentId = " << sentId << "; tokens = ";
    for(unsigned i = 0; i < tokens.size(); ++i) {
      headerSS << i << "=" << vocabEncoder.Decode(tokens[i].details[learningInfo.oneBasedConllFieldIdReconstructed-1]) << " ";
    }
    headerSS << endl;
    for(unsigned i = 0; i < tokens.size(); ++i) {
      headerSS << "\t" << i;
    }
    headerSS << endl;
    
    if(sentId == 13 || sentId == 1715) { 
      cerr << headerSS.str() << endl << "p(attachment|X,Z) = " << endl << marginalGivenXZSS.str() << endl;
      cerr << headerSS.str() << endl << "p(attachment|X) = "   << endl << marginalGivenXSS.str() << endl;
      cerr << "------------------------------------" << endl;
    }
    
    // don't forget to also update the gradients of root selection features
    for(unsigned rootPosition = 0; rootPosition < tokens.size(); ++rootPosition) {
      FastSparseVector<double> activeFeatures;
      lambda->FireFeatures(ROOT_DETAILS, tokens[rootPosition], tokens, activeFeatures);
      double marginalGivenXZ = (yGivenXZLaplacianHat(0, rootPosition) * 
                                yGivenXZLaplacianHatInverse(rootPosition, 0)).as_float();
      double marginalGivenX = (yGivenXLaplacianHat(0, rootPosition) * 
                               yGivenXLaplacianHatInverse(rootPosition, 0)).as_float();
      double marginalDiff = marginalGivenX - marginalGivenXZ;
      for(auto featIter = activeFeatures.begin(); featIter != activeFeatures.end(); ++featIter) {
        derivativeWRTLambda[featIter->first] += featIter->second * marginalDiff;
      } 
    }

  
    // debug info
    if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH && sentId % learningInfo.nSentsPerDot == 0) {
      cerr << ".";
    }
  } // end of training examples 

  cerr << learningInfo.mpiWorld->rank() << "|";
  
  return objective;
}

// -loglikelihood is the return value
// nll = - log ( exp( \lambda . f(y-gold, x) ) / \sum_y exp( \lambda.f(y,x) )
//     = - \lambda . f(y-gold, x) + log \sum_y exp( \lambda . f(y, x) )
//     = - \lambda . f(y-gold, x) - nLogZ
// nDerivative_k = - f_k(y-gold, x) + [ \sum_y f_k(y,x) * exp (\lambda . f(y,x)) ] / [ \sum_y exp( \lambda . f(y, x)) ]
//               = - f_k(y-gold, x) + E_{p(y|x)}[f_k(y,x)]
//               = - f_k(y-gold, x) + \sum_{h,m} p_marginal(m,h|x) * f_k(m,h)
double LatentCrfParser::ComputeNllYGivenXAndLambdaGradient(
							  vector<double> &derivativeWRTLambda, int fromSentId, int toSentId) {
  
  double objective = 0;

  assert(derivativeWRTLambda.size() == lambda->GetParamsCount());
  
  // for each training example
  for(int sentId = fromSentId; sentId < toSentId; sentId++) {
    
    // sentId is assigned to the process with rank = sentId % world.size()
    if(sentId % learningInfo.mpiWorld->size() != learningInfo.mpiWorld->rank()) {
      continue;
    }

    // prune long sequences
    if( learningInfo.maxSequenceLength > 0 && 
        GetObservableDetailsSequence(sentId).size() > learningInfo.maxSequenceLength ) {
      continue;
    }
    
    // build A_{y|x} matrix and use matrix tree theoerem to compute Z(x), \sum_{y: (h,m) \in y} p(y|x)
    MatrixXlogd yGivenXAdjacency, yGivenXLaplacianHat, yGivenXLaplacianHatInverse;
    VectorXlogd yGivenXRootSelection;
    LogValD yGivenXLaplacianHatDeterminant;
    BuildMatrices(sentId, yGivenXRootSelection, yGivenXAdjacency, yGivenXLaplacianHat, 
                  yGivenXLaplacianHatInverse, yGivenXLaplacianHatDeterminant, false);
    LogValD Z = yGivenXLaplacianHatDeterminant;
    
    // update the loglikelihood
    double nLogZ = -log(Z);
    // keep an eye on bad numbers
    if(std::isinf(nLogZ)) {
      cerr << "WARNING: nLogZ = " << nLogZ << ". ";
      //cerr << "the yGivenXLaplacianHat matrix looks like: " << endl << yGivenXLaplacianHat << endl;
      cerr << "I will just ignore this sentence ..." << endl;
      assert(false);
      continue;
    } else if(std::isnan(nLogZ)) {
      cerr << "ERROR: nLogZ = " << nLogZ << endl;
      assert(false);
    }

    // the denominator
    objective -= nLogZ;
    //cerr << "objective -= nLogZ=" << nLogZ << endl;

    auto tokens = GetObservableDetailsSequence(sentId);
    // for (h,m) in \cal{T}_{np}^s:
    //   for k in f(x,h,m):
    //     see the update formula at the beginning of this function
    for(unsigned headPosition = 0; headPosition < tokens.size(); ++headPosition) {
      for(unsigned childPosition = 0; childPosition < tokens.size(); ++childPosition) {
        FastSparseVector<double> activeFeatures;
        if(headPosition != childPosition) {
          lambda->FireFeatures(tokens[headPosition], tokens[childPosition], tokens, activeFeatures);
        }
        double marginalGivenX = childPosition == 0? 0.0 :
          (yGivenXAdjacency(headPosition, childPosition) * 
           yGivenXLaplacianHatInverse(childPosition, childPosition)).as_float();
        marginalGivenX -= headPosition == 0? 0.0 :
          (yGivenXAdjacency(headPosition, childPosition) * 
           yGivenXLaplacianHatInverse(childPosition, headPosition)).as_float();
        
        bool goldAttachment = 
          (unsigned) tokens[childPosition].details[ObservationDetailsHeader::HEAD] == headPosition + 1;

        if(goldAttachment) {
          objective -= lambda->DotProduct(activeFeatures);
          //cerr << "objective -= lambda.f(h,m)=" << lambda->DotProduct(activeFeatures) << endl;
        }

        for(auto featIter = activeFeatures.begin(); featIter != activeFeatures.end(); ++featIter) {
          derivativeWRTLambda[featIter->first] += featIter->second * marginalGivenX;
          //cerr << "derivativeWRTLambda[" << featIter->first << "] += featIter->second * marginalGivenX = " << featIter->second << " * " << marginalGivenX << " = " << featIter->second * marginalGivenX << endl;
          if(goldAttachment) {
            derivativeWRTLambda[featIter->first] -= featIter->second * 1.0;
            //cerr << "derivativeWRTLambda[" << featIter->first << "] -= featIter->second * 1.0 = " << featIter->second * 1.0 << endl;
          }
        }
      }
    }
    
    // don't forget to also update the gradients of root selection features
    for(unsigned rootPosition = 0; rootPosition < tokens.size(); ++rootPosition) {
      FastSparseVector<double> activeFeatures;
      lambda->FireFeatures(ROOT_DETAILS, tokens[rootPosition], tokens, activeFeatures);
      double marginalGivenX = (yGivenXLaplacianHat(0, rootPosition) * 
                               yGivenXLaplacianHatInverse(rootPosition, 0)).as_float();
      bool goldAttachment = 
        tokens[rootPosition].details[ObservationDetailsHeader::HEAD] == 0;
      if(goldAttachment) {
        objective -= lambda->DotProduct(activeFeatures);
        //cerr << "objective -= lambda->DotProduct(activeFeatures) = " << lambda->DotProduct(activeFeatures) << endl;
      }
      for(auto featIter = activeFeatures.begin(); featIter != activeFeatures.end(); ++featIter) {
        derivativeWRTLambda[featIter->first] += featIter->second * marginalGivenX;
        //cerr << "derivativeWRTLambda[" << featIter->first << "] += featIter->second * marginalGivenX = " << featIter->second << " * " << marginalGivenX << " = " << featIter->second * marginalGivenX << endl;
        if(goldAttachment) {
          derivativeWRTLambda[featIter->first] -= featIter->second * 1.0;
          //cerr << "derivativeWRTLambda[" << featIter->first << "] -= featIter->second * 1.0 = " << featIter->second * 1.0 << endl;
        }
      }
    }

    // debug info
    if(sentId % learningInfo.nSentsPerDot == 0) {
      cerr << ".";
    }
  } // end of training examples 

  cerr << learningInfo.mpiWorld->rank() << "|";
  
  return objective;
}

// run Tarjan's implementation of Chiu-Liu-Edmonds for maximum spanning trees
double LatentCrfParser::GetMaxSpanningTree(VectorXlogd &rootSelection, MatrixXlogd &adjacency, vector<int> &maxSpanningTree) {

  // I will use an explicit ROOT vertix, of index adjacency.rows()
  unsigned n_vertices = adjacency.rows() + 1;
  complete_graph      g(n_vertices);
  multi_array<double, 2> weights(extents[n_vertices][n_vertices]);
  vector<Vertex>      rootVertices = {(int)adjacency.rows()}; // you can use this vector to specify particular root(s)
  
  // set weights (doubles)
  for(unsigned rowId = 0; rowId < n_vertices; ++rowId) {
    for(unsigned columnId = 0; columnId < n_vertices; ++columnId) {
      weights[rowId][columnId] = 
        rowId < adjacency.rows() && columnId < adjacency.rows()?
        -MultinomialParams::nLog( adjacency(rowId, columnId).as_float() ):
        rowId == adjacency.rows() && columnId < adjacency.rows()? 
        -MultinomialParams::nLog( rootSelection(columnId).as_float() ) : 
        -MultinomialParams::NLOG_ZERO; 
    }
  }
  
  // run edmonds algorithm for a few cases
  vector<Edge>        branching;
  edmonds_optimum_branching<true, true, true>
    (g, identity_property_map(), weights.origin(),
     rootVertices.begin(), rootVertices.end(), back_inserter(branching));
  
  // initialize the mst; everyone is a root
  // TODO-OPT: you don't really need to clear before resizing, and you don't need to fill up with -2;
  //           I just do this now for debugging purposes
  maxSpanningTree.clear();
  maxSpanningTree.resize(adjacency.rows(), -2);
  
  // modify parents of nonroot vertices
  double ans = 0.0;
  unsigned edgesCounter = 0;
  BOOST_FOREACH (Edge e, branching)
    {
      edgesCounter++;
      maxSpanningTree[target(e, g)] = source(e, g) == adjacency.rows()? -1 : source(e, g);
      ans += weights[source(e, g)][target(e, g)];
    }
  
  assert(!std::isinf(ans) && !std::isnan(ans));     
  assert(edgesCounter == adjacency.rows());
  
  // TODO-OPT: for debugging only
  bool atLeastOneRootExist = false;
  for(unsigned potentialRootId = 0; potentialRootId < adjacency.rows(); ++potentialRootId) {
    assert(maxSpanningTree[potentialRootId] != -2);
    if(maxSpanningTree[potentialRootId] == -1) { atLeastOneRootExist = true; }
  }
  assert(atLeastOneRootExist);

  return ans;
}

// element i in the returned vector is the zero-based index of the ith token in the sentence.
// the parent of a root word is -1
vector<int> LatentCrfParser::GetViterbiParse(int sentId, bool conditionOnZ, double &treeLogProb) {
  // build A_{y|x,z} or A_{y|x} (depending on the second parameter) matrix and use matrix tree theoerem to compute Z(x), \sum_{y: (h,m) \in y} p(y|x)
  MatrixXlogd adjacency, laplacianHat, laplacianHatInverse;
  VectorXlogd rootSelection;
  LogValD laplacianHatDeterminant;
  vector<int> maxSpanTree;

  auto tokens = GetObservableDetailsSequence(sentId);
  if(tokens.size() > learningInfo.maxSequenceLength) {
    maxSpanTree.resize(tokens.size(), -1);
    treeLogProb = 0.0;
    cerr << "WARNING: skipping Viterbi parsing of sentId=" << sentId << ", length = " << tokens.size() << " > learningInfo.maxSequenceLength = " << learningInfo.maxSequenceLength << endl; 
    return maxSpanTree;
  }

  BuildMatrices(sentId, rootSelection, adjacency, laplacianHat, 
                laplacianHatInverse, laplacianHatDeterminant, conditionOnZ);
  
  if(adjacency.rows() == 1) {
    maxSpanTree.push_back(-1); // the single word in a sentence of length 1 must be root
    return maxSpanTree;
  }
  double treeScoreExponent = GetMaxSpanningTree(rootSelection, adjacency, maxSpanTree);
  LogValD treeScoreLogValD = LogValD(treeScoreExponent, false);
  double treeProb = (treeScoreLogValD / laplacianHatDeterminant).as_float();
  if(treeProb > 1.0) { treeProb = 1.0; }
  else if(treeProb <= 0.0) { treeProb = 0.000001; }
  treeLogProb = log(treeProb);
  
  return maxSpanTree;
}

pair<complete_graph::edge_iterator, complete_graph::edge_iterator> 
boost::edges(const complete_graph &g)
{
  return make_pair(complete_graph::edge_iterator(g.n_vertices, 1),
                   complete_graph::edge_iterator(g.n_vertices, g.n_vertices*g.n_vertices));
}

unsigned
boost::num_edges(const complete_graph &g)
{
  return (g.n_vertices - 1) * (g.n_vertices - 1);
}

int
boost::source(int edge, const complete_graph &g)
{
  return edge / g.n_vertices;
}

int
boost::target(int edge, const complete_graph &g)
{
  return edge % g.n_vertices;
}

void LatentCrfParser::FireFeatures(int yI, int yIM1, unsigned sentId, int i, 
				  FastSparseVector<double> &activeFeatures) { 
    cerr << "this method is not implemented for dependency parsing" << endl;
    assert(false);
}