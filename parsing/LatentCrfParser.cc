#include "LatentCrfParser.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

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
				   const string &wordPairFeaturesFilename) : LatentCrfModel(textFilename,
											    outputPrefix,
											    learningInfo,
											    LatentCrfParser::ROOT_POSITION,
											    LatentCrfParser::Task::DEPENDENCY_PARSING) {

  // unlike POS tagging, yDomain depends on the src sentence length. we will set it on a per-sentence basis.
  this->yDomain.clear();
  
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
  vocabEncoder.ReadConll(textFilename, sents);
  assert(sents.size() > 0);
  examplesCount = sents.size();
  
  // fix learningInfo.firstKExamplesToLabel
  cerr << "firstKExamplesToLabel = " << learningInfo.firstKExamplesToLabel << endl;
  if(learningInfo.firstKExamplesToLabel == 1) {
    learningInfo.firstKExamplesToLabel = examplesCount;
  }  
  cerr << "firstKExamplesToLabel = " << learningInfo.firstKExamplesToLabel << endl;

  if(learningInfo.mpiWorld->rank() == 0 && wordPairFeaturesFilename.size() > 0) {
    lambda->LoadPrecomputedFeaturesWith2Inputs(wordPairFeaturesFilename);
  }

  // master signals to slaves that he's done
  if(learningInfo.mpiWorld->rank() == 0) {
    bool vocabEncoderIsReady;
    boost::mpi::broadcast<bool>(*learningInfo.mpiWorld, vocabEncoderIsReady, 0);
  }

  // initialize (and normalize) the log theta params to gaussians
  if(learningInfo.initializeThetasWithGaussian || learningInfo.initializeThetasWithUniform) {
    cerr << "calling InitTheta() " << endl;
    InitTheta();
  }

  if(initialThetaParamsFilename.size() > 0) {
    //assert(nLogThetaGivenOneLabel.params.size() == 0);
    if(learningInfo.mpiWorld->rank() == 0) {
      cerr << "initializing theta params from " << initialThetaParamsFilename << endl;
    }
    MultinomialParams::LoadParams(initialThetaParamsFilename, nLogThetaGivenOneLabel, vocabEncoder, true, true);
    string reloadedParamsFilename = initialThetaParamsFilename + ".reloaded";
    MultinomialParams::PersistParams(reloadedParamsFilename, nLogThetaGivenOneLabel, vocabEncoder, true, true);
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
         learningInfo.initializeThetasWithModel1);
  assert(!learningInfo.initializeThetasWithModel1);
  
  if(learningInfo.mpiWorld->rank() == 0 && learningInfo.debugLevel >= DebugLevel::CORPUS) {
    cerr << "master" << learningInfo.mpiWorld->rank() << ": initializing thetas...";
  }

  assert(sents.size() > 0);

  // first initialize nlogthetas 
  nLogThetaGivenOneLabel.params.clear();
  for(unsigned sentId = 0; sentId < sents.size(); ++sentId) {
    vector<ObservationDetails> &sent = sents[sentId];
    vector<ObservationDetails> &reconstructedSent = sents[sentId];
    for(unsigned i = 0; i < sent.size(); ++i) {
      nLogThetaGivenOneLabel.params[ROOT_ID][sent[i].details[ObservationDetailsHeader::RECONSTRUCTED]] = 
        learningInfo.initializeThetasWithGaussian?
        abs(gaussianSampler.Draw()) : 1.0;
      auto parentToken = sent[i];
      for(unsigned j = 0; j < reconstructedSent.size(); ++j) {
        if(i == j) { continue; }
        auto childToken = reconstructedSent[j];
        nLogThetaGivenOneLabel.params[parentToken.details[ObservationDetailsHeader::RECONSTRUCTED]][childToken.details[ObservationDetailsHeader::RECONSTRUCTED]] = 
          learningInfo.initializeThetasWithGaussian?
          abs(gaussianSampler.Draw()):1.0;
      }
    }
  }
  
  // then normalize them
  MultinomialParams::NormalizeParams(nLogThetaGivenOneLabel);

  stringstream thetaParamsFilename;
  thetaParamsFilename << outputPrefix << ".init.theta";
  PersistTheta(thetaParamsFilename.str());

  if(learningInfo.mpiWorld->rank() == 0) {
    cerr << "done" << endl;
  }
}

void LatentCrfParser::PrepareExample(unsigned exampleId) {
  yDomain.clear();
  this->yDomain.insert(LatentCrfParser::ROOT_POSITION);
  unsigned sentLength = testingMode? testSents[exampleId].size() : sents[exampleId].size();
  // each position in the src sentence, including null, should have an entry in yDomain
  for(unsigned i = LatentCrfParser::ROOT_POSITION + 1; i < LatentCrfParser::ROOT_POSITION + sentLength + 1; ++i) {
    yDomain.insert(i);
  }
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
  labels = GetViterbiParse(0, true);

  // set down ;)
  testingMode = false;

  assert(labels.size() == tokens.size());  
}

void LatentCrfParser::Label(const string &labelsFilename) {
  ofstream labelsFile(labelsFilename.c_str());
  assert(learningInfo.firstKExamplesToLabel <= examplesCount);
  for(unsigned exampleId = 0; exampleId < learningInfo.firstKExamplesToLabel; ++exampleId) {
    lambda->learningInfo->currentSentId = exampleId;
    if(exampleId % learningInfo.mpiWorld->size() != (unsigned)learningInfo.mpiWorld->rank()) {
      if(learningInfo.mpiWorld->rank() == 0){
        string labelSequence;
        learningInfo.mpiWorld->recv(exampleId % learningInfo.mpiWorld->size(), 0, labelSequence);
        labelsFile << labelSequence;
      }
      continue;
    }

    std::vector<int> labels = GetViterbiParse(exampleId, true);
    auto tokens = GetObservableDetailsSequence(exampleId);
    assert(labels.size() == tokens.size());
    
    stringstream ss;
    for(unsigned i = 0; i < labels.size(); ++i) {
      // conll token index is one based 
      int parent = labels[i] + 1;
      ss << tokens[i].details[ObservationDetailsHeader::ID] << "\t" \
         << vocabEncoder.Decode(tokens[i].details[ObservationDetailsHeader::FORM]) << "\t" \
         << vocabEncoder.Decode(tokens[i].details[ObservationDetailsHeader::LEMMA]) << "\t" \
         << vocabEncoder.Decode(tokens[i].details[ObservationDetailsHeader::CPOSTAG]) << "\t" \
         << vocabEncoder.Decode(tokens[i].details[ObservationDetailsHeader::POSTAG]) << "\t" \
         << vocabEncoder.Decode(tokens[i].details[ObservationDetailsHeader::FEATS]) << "\t" \
         << parent << "\t"                               \
         << "_" << "\t" \
         << "_" << "\t" \
         << "_" << endl;
    }
    ss << endl;
    if(learningInfo.mpiWorld->rank() == 0){
      labelsFile << ss.str();
    } else {
      //cerr << "rank" << learningInfo.mpiWorld->rank() << " will send exampleId " << exampleId << " to master" << endl; 
      learningInfo.mpiWorld->send(0, 0, ss.str());
      //cerr << "rank" << learningInfo.mpiWorld->rank() << " sending done." << endl;
    }
  }
  labelsFile.close();
}

// convenience function
void LatentCrfParser::FireFeatures(const unsigned sentId,
                                  FastSparseVector<double> &h) {
  auto tokens = GetObservableDetailsSequence(sentId);
  if(tokens.size() > learningInfo.maxSequenceLength) {
    return;
  }
  MatrixXd adjacency, laplacianHat, laplacianHatInverse;
  VectorXd rootSelection;
  double laplacianHatDeterminant = -1;
  BuildMatrices(sentId, rootSelection, adjacency, laplacianHat,
                laplacianHatInverse, laplacianHatDeterminant, false);
  assert(laplacianHatDeterminant != -1);
}

// build the matrixes which can be used to marginalize proper dependency trees as of Koo et al 2007.
void LatentCrfParser::BuildMatrices(const unsigned sentId,
                                    VectorXd &rootScores,
                                    MatrixXd &adjacency,
                                    MatrixXd &laplacianHat,
                                    MatrixXd &laplacianHatInverse,
                                    double &laplacianHatDeterminant,
                                    bool conditionOnZ) {
 
  // build A_{y|x} matrix and use matrix tree theoerem to compute Z(x), \sum_{y: (h,m) \in y} p(y|x)
  auto tokens = GetObservableDetailsSequence(sentId);
  auto reconstructedTokens = GetReconstructedObservableDetailsSequence(sentId);
  assert(tokens.size() == reconstructedTokens.size());
  // adjacency matrix A(y|x) in (Koo et al. 2007)
  adjacency.resize(tokens.size(), tokens.size());
  for(unsigned headPosition = 0; headPosition < tokens.size(); ++headPosition) {
    for(unsigned childPosition = 0; childPosition < tokens.size(); ++childPosition) {
      double multinomialTerm = conditionOnZ? nLogThetaGivenOneLabel[reconstructedTokens[headPosition].details[ObservationDetailsHeader::RECONSTRUCTED]][reconstructedTokens[childPosition].details[ObservationDetailsHeader::RECONSTRUCTED]]: 0.0;
      FastSparseVector<double> activeFeatures;
      if(headPosition == childPosition) {
        adjacency(headPosition, childPosition) = 0.0;
      } else {
        lambda->FireFeatures(tokens[headPosition], tokens[childPosition], tokens, activeFeatures);
        adjacency(headPosition, childPosition) = MultinomialParams::nExp( multinomialTerm - lambda->DotProduct( activeFeatures ) );
      }
    }
  }
  //cerr << "adjacency = " << endl;
  //cerr << adjacency << endl << endl;
  // root selection scores r(y|x) in (Koo et al. 2007)
  rootScores.resize(tokens.size());
  //cerr << "rootScores = " << endl;
  //cerr << rootScores << endl << endl;
  int64_t reconstructedRoot = LatentCrfParser::ROOT_DETAILS.details[ObservationDetailsHeader::RECONSTRUCTED];
  for(unsigned rootPosition = 0; rootPosition < tokens.size(); ++rootPosition) {
    double multinomialTerm = conditionOnZ? 
      nLogThetaGivenOneLabel[reconstructedRoot][reconstructedTokens[rootPosition].details[ObservationDetailsHeader::RECONSTRUCTED]]: 
      0.0;
    FastSparseVector<double> activeFeatures;
    lambda->FireFeatures(LatentCrfParser::ROOT_DETAILS, tokens[rootPosition], tokens, activeFeatures);
    rootScores(rootPosition) = MultinomialParams::nExp( multinomialTerm - lambda->DotProduct( activeFeatures ) );
  }
  // laplacian matrix L(y|x) in (Koo et al. 2007)
  laplacianHat = -1.0 * adjacency;
  for(unsigned colIndex = 0; colIndex < tokens.size(); ++colIndex) {
    laplacianHat(colIndex, colIndex) = adjacency.col(colIndex).array().sum();
  }
  // modified laplacian matrix to allow for O(n^3) inference; \hat{L}(y|x) in (Koo et al. 2007)
  laplacianHat.row(0) = rootScores;
  //cerr << "laplacianHat = " << endl;
  //cerr << laplacianHat << endl << endl;

  // now, we divide all elements in laplacianHat by a constant to ensure numerical stability
  // while computing the determinant and inverse
  //cerr << "adjacency is " << adjacency << endl << endl;
  //cerr << "laplacianHat is " << laplacianHat << endl << endl;
  //cerr << "laplacianHat is " << laplacianHat << endl;
  //cerr << "laplacianHat.maxCoeff() = " << laplacianHat.maxCoeff() << " ";
  //cerr << "laplacianHat.minCoeff() = " << laplacianHat.minCoeff() << endl;
  double scalingConstant = max(laplacianHat.maxCoeff(), fabs(laplacianHat.minCoeff()));
  assert(scalingConstant > 0.0);
  //cerr << "scalingConstant = " << scalingConstant << endl;
  laplacianHat /= scalingConstant;
  //cerr << "laplacianHat / scalingConstant = " << endl;
  //cerr << laplacianHat << endl << endl;

  // now compute determinant
  laplacianHatDeterminant = laplacianHat.determinant();
  for(unsigned i = 0; i < laplacianHat.rows(); ++i) {
    laplacianHatDeterminant *= scalingConstant;
  }
  if(std::isinf(laplacianHatDeterminant)) {
    cerr << "laplacianHat.cols() = " << laplacianHat.cols() << endl;
    cerr << "laplacianHatDeterminant = " << laplacianHatDeterminant << " = " << laplacianHat.determinant() << " * " << pow(scalingConstant, (double) laplacianHat.rows()) << endl << endl;
    assert(false);
  }
  
  // then compute inverse
  laplacianHatInverse = laplacianHat.inverse();
  laplacianHatInverse /= scalingConstant;
  //cerr << "laplacianHatInverse = " << endl;
  //cerr << laplacianHatInverse << endl << endl;

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
    for(unsigned childIndex = 0; childIndex < tokens.size(); ++childIndex) {
      int zeroBasedHeadIndex = (int) tokens[childIndex].details[ObservationDetailsHeader::HEAD] - 1;
      int64_t childThetaKey = tokens[childIndex].details[ObservationDetailsHeader::RECONSTRUCTED];
      int64_t headThetaKey = 
        zeroBasedHeadIndex < 0?
        LatentCrfParser::ROOT_ID:
        tokens[zeroBasedHeadIndex].details[ObservationDetailsHeader::RECONSTRUCTED];
      assert( nLogThetaGivenOneLabel.params.find(headThetaKey) != nLogThetaGivenOneLabel.params.end());
      assert( nLogThetaGivenOneLabel.params[headThetaKey].find(childThetaKey) != nLogThetaGivenOneLabel.params[headThetaKey].end());
      nLogThetaGivenOneLabel.params[headThetaKey][childThetaKey]++;
    }
  }
  
  MultinomialParams::NormalizeParams(nLogThetaGivenOneLabel, 1.0, false, true, false);
  
}

// returns -log p(z|x)
double LatentCrfParser::UpdateThetaMleForSent(const unsigned sentId, 
  				     MultinomialParams::ConditionalMultinomialParam<int64_t> &mle, 
  				     boost::unordered_map<int64_t, double> &mleMarginals) {

  //cerr << "LatentCrfParser's impelmentation of LatentCrfModel::UpdateThetaMleForSent" << endl;
  //std::cerr << "sentId = " << sentId << endl;
  assert(sentId < examplesCount);

  // prune long sequences
  if( learningInfo.maxSequenceLength > 0 && 
      GetObservableDetailsSequence(sentId).size() > learningInfo.maxSequenceLength ) {
    return 0.0;
  }
  
  // build A_{y|x} matrix and use matrix tree theoerem to compute Z(x), \sum_{y: (h,m) \in y} p(y|x)
  // TODO: WE DON'T REALLY NEED THIS MATRIX; AFTERALL, COMPUTING LIKELIHOOD IN EM IS ONLY USEFUL FOR
  // DEBUGGING PURPOSES
  MatrixXd yGivenXAdjacency, yGivenXLaplacianHat, yGivenXLaplacianHatInverse;
  VectorXd yGivenXRootSelection;
  double yGivenXLaplacianHatDeterminant = -1;
  BuildMatrices(sentId, yGivenXRootSelection, yGivenXAdjacency, yGivenXLaplacianHat, 
                yGivenXLaplacianHatInverse, yGivenXLaplacianHatDeterminant, false);
  assert(yGivenXLaplacianHatDeterminant != -1);
  double Z = yGivenXLaplacianHatDeterminant;
  
  // build A_{y|x,z} matrix and use matrix tree theorem to compute C(x), marginal(h,m;y|z,x)=\sum_{y:(h,m)\in y} p(y|x,z)
  MatrixXd yGivenXZAdjacency, yGivenXZLaplacianHat, yGivenXZLaplacianHatInverse;
  VectorXd yGivenXZRootSelection;
  double yGivenXZLaplacianHatDeterminant = -1;
  BuildMatrices(sentId, yGivenXZRootSelection, yGivenXZAdjacency, yGivenXZLaplacianHat, 
                yGivenXZLaplacianHatInverse, yGivenXZLaplacianHatDeterminant, true);
  assert(yGivenXZLaplacianHatDeterminant != -1);
  double C = yGivenXZLaplacianHatDeterminant;
  if(C > Z) {
    cerr << "C = " << C << endl;
    cerr << "Z = " << Z << endl;
  }
  assert(C <= Z);

  auto reconstructedTokens = GetReconstructedObservableDetailsSequence(sentId);
  
  // for (h,m) \in \cal{T}_{np}^s: mle[h][m] += nLogThetaGivenOneLabel[h][m] * marginal(h,m;y|z,x)
  for(unsigned rootPosition = 0; rootPosition < yGivenXZLaplacianHat.rows(); ++rootPosition) {
    // marginal probability of making this decision; \mu_{0,m} in (Koo et al. 2007)
    double marginal = yGivenXZLaplacianHat(0,rootPosition) * yGivenXZLaplacianHatInverse(rootPosition,0);
    //if(marginal == 0.0) { cerr << "WARNING: marginal == 0.0 when rootPosition = " << rootPosition << endl; } 
    mle[LatentCrfParser::ROOT_ID][reconstructedTokens[rootPosition].details[ObservationDetailsHeader::RECONSTRUCTED]] += MultinomialParams::nExp(nLogThetaGivenOneLabel[LatentCrfParser::ROOT_ID][reconstructedTokens[rootPosition].details[ObservationDetailsHeader::RECONSTRUCTED]]) * marginal;
    mleMarginals[LatentCrfParser::ROOT_ID] += MultinomialParams::nExp(nLogThetaGivenOneLabel[LatentCrfParser::ROOT_ID][reconstructedTokens[rootPosition].details[ObservationDetailsHeader::RECONSTRUCTED]]) * marginal;
  }
  for(unsigned headPosition = 0; headPosition < yGivenXZLaplacianHat.rows(); ++headPosition) {
    for(unsigned childPosition = 0; childPosition < yGivenXZLaplacianHat.cols(); ++childPosition) {
      double marginal = childPosition == 0? 0.0 :
        yGivenXZAdjacency(headPosition, childPosition) * 
        yGivenXZLaplacianHatInverse(childPosition, childPosition);
      marginal -= headPosition == 0? 0.0 :
        yGivenXZAdjacency(headPosition, childPosition) * 
        yGivenXZLaplacianHatInverse(childPosition, headPosition);
      /*if(marginal == 0.0 && headPosition != childPosition) { 
        cerr << "WARNING: marginal == 0.0, when headPosition = " << headPosition << ", childPosition = " << childPosition << endl; 
        } */
      mle[reconstructedTokens[headPosition].details[ObservationDetailsHeader::RECONSTRUCTED]][reconstructedTokens[childPosition].details[ObservationDetailsHeader::RECONSTRUCTED]] += marginal * MultinomialParams::nExp(nLogThetaGivenOneLabel[reconstructedTokens[headPosition].details[ObservationDetailsHeader::RECONSTRUCTED]][reconstructedTokens[childPosition].details[ObservationDetailsHeader::RECONSTRUCTED]]);
      mleMarginals[reconstructedTokens[headPosition].details[ObservationDetailsHeader::RECONSTRUCTED]] += marginal * MultinomialParams::nExp(nLogThetaGivenOneLabel[reconstructedTokens[headPosition].details[ObservationDetailsHeader::RECONSTRUCTED]][reconstructedTokens[childPosition].details[ObservationDetailsHeader::RECONSTRUCTED]]);
    }
  }

  // nlog p(z|x)
  double nLogC = MultinomialParams::nLog(C), nLogZ = MultinomialParams::nLog(Z);
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
    MatrixXd yGivenXAdjacency, yGivenXLaplacianHat, yGivenXLaplacianHatInverse;
    VectorXd yGivenXRootSelection;
    double yGivenXLaplacianHatDeterminant;
    BuildMatrices(sentId, yGivenXRootSelection, yGivenXAdjacency, yGivenXLaplacianHat, 
                  yGivenXLaplacianHatInverse, yGivenXLaplacianHatDeterminant, false);
    double Z = yGivenXLaplacianHatDeterminant;
    
    // build A_{y|x,z} matrix and use matrix tree theorem to compute C(x), marginal(h,m;y|z,x)=\sum_{y:(h,m)\in y} p(y|x,z)
    MatrixXd yGivenXZAdjacency, yGivenXZLaplacianHat, yGivenXZLaplacianHatInverse;
    VectorXd yGivenXZRootSelection;
    double yGivenXZLaplacianHatDeterminant;
    BuildMatrices(sentId, yGivenXZRootSelection, yGivenXZAdjacency, yGivenXZLaplacianHat, 
                  yGivenXZLaplacianHatInverse, yGivenXZLaplacianHatDeterminant, true);
    if(yGivenXZLaplacianHatDeterminant <= 0.0) {
      cerr << "yGivenXZLaplacianHatDeterminant = " << yGivenXZLaplacianHatDeterminant << endl;
      cerr << "yGivenXZLaplacianHat = " << endl << yGivenXZLaplacianHat << endl << endl;
      assert(false);
    }
    double C = yGivenXZLaplacianHatDeterminant;
    assert(!std::isinf(Z));
    assert(!std::isinf(C));
    if(C > Z) {
      cerr << "ERROR: C = " << C << " > Z = " << Z << endl;
    }
    assert(C <= Z);
    // update the loglikelihood
    assert(!learningInfo.useEarlyStopping);
    double nLogC = MultinomialParams::nLog(C), nLogZ = MultinomialParams::nLog(Z);
    // keep an eye on bad numbers
    if(std::isinf(nLogZ) || std::isinf(nLogC)) {
      cerr << "WARNING: nLogZ = " << nLogZ << ", nLogC = " << nLogC << ". ";
      cerr << "WARNING: Z = " << Z << ", C = " << C << ". ";
      cerr << "I will just ignore this sentence ..." << endl;
      
      //cerr << "yGivenXZAdjacency = " << endl << yGivenXZAdjacency << endl << endl;
      cerr << "yGivenXZLaplacianHat = " << endl << yGivenXZLaplacianHat << endl << endl;
      cerr << "yGivenXLaplacianHat = " << endl << yGivenXLaplacianHat << endl << endl;
      //cerr << "yGivenXZLaplacianHatInverse = " << endl << yGivenXZLaplacianHatInverse << endl << endl;
      //cerr << "yGivenXZRootSelection = " << endl << yGivenXZRootSelection << endl << endl;

      assert(false);
      continue;
    } else if(std::isnan(nLogZ) || std::isnan(nLogC)) {
      cerr << "ERROR: nLogZ = " << nLogZ << ", nLogC = " << nLogC << endl;
      assert(false);
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
    // for (h,m) in \cal{T}_{np}^s:
    //   for k in f(x,h,m):
    //     dll/d\lambda_k += f_k(x,h,m) * [marginal(h,m;y|z,x)-marginal(h,m;y|x)]
    for(unsigned headPosition = 0; headPosition < tokens.size(); ++headPosition) {
      for(unsigned childPosition = 0; childPosition < tokens.size(); ++childPosition) {
        FastSparseVector<double> activeFeatures;
        if(headPosition != childPosition) {
          lambda->FireFeatures(tokens[headPosition], tokens[childPosition], tokens, activeFeatures);
        }
        double marginalGivenXZ = childPosition == 0? 0.0 :
          yGivenXZAdjacency(headPosition, childPosition) * 
          yGivenXZLaplacianHatInverse(childPosition, childPosition);
        marginalGivenXZ -= headPosition == 0? 0.0 :
          yGivenXZAdjacency(headPosition, childPosition) * 
          yGivenXZLaplacianHatInverse(childPosition, headPosition);
        double marginalGivenX = childPosition == 0? 0.0 :
          yGivenXAdjacency(headPosition, childPosition) * 
          yGivenXLaplacianHatInverse(childPosition, childPosition);
        marginalGivenX -= headPosition == 0? 0.0 :
          yGivenXAdjacency(headPosition, childPosition) * 
          yGivenXLaplacianHatInverse(childPosition, headPosition);
        
        double marginalDiff = marginalGivenX - marginalGivenXZ;
        for(auto featIter = activeFeatures.begin(); featIter != activeFeatures.end(); ++featIter) {
          derivativeWRTLambda[featIter->first] += featIter->second * marginalDiff;
        }
      }
    }
    
    // don't forget to also update the gradients of root selection features
    for(unsigned rootPosition = 0; rootPosition < tokens.size(); ++rootPosition) {
      FastSparseVector<double> activeFeatures;
      lambda->FireFeatures(ROOT_DETAILS, tokens[rootPosition], tokens, activeFeatures);
      double marginalGivenXZ = yGivenXZLaplacianHat(0, rootPosition) * yGivenXZLaplacianHatInverse(rootPosition, 0);
      double marginalGivenX =   yGivenXLaplacianHat(0, rootPosition) * yGivenXLaplacianHatInverse(rootPosition, 0);
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
    MatrixXd yGivenXAdjacency, yGivenXLaplacianHat, yGivenXLaplacianHatInverse;
    VectorXd yGivenXRootSelection;
    double yGivenXLaplacianHatDeterminant;
    BuildMatrices(sentId, yGivenXRootSelection, yGivenXAdjacency, yGivenXLaplacianHat, 
                  yGivenXLaplacianHatInverse, yGivenXLaplacianHatDeterminant, false);
    double Z = yGivenXLaplacianHatDeterminant;
    
    // update the loglikelihood
    double nLogZ = MultinomialParams::nLog(Z);
    // keep an eye on bad numbers
    if(std::isinf(nLogZ)) {
      cerr << "WARNING: nLogZ = " << nLogZ << ". ";
      cerr << "the yGivenXLaplacianHat matrix looks like: " << endl << yGivenXLaplacianHat << endl;
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
          yGivenXAdjacency(headPosition, childPosition) * 
          yGivenXLaplacianHatInverse(childPosition, childPosition);
        marginalGivenX -= headPosition == 0? 0.0 :
          yGivenXAdjacency(headPosition, childPosition) * 
          yGivenXLaplacianHatInverse(childPosition, headPosition);
        
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
      double marginalGivenX = yGivenXLaplacianHat(0, rootPosition) * yGivenXLaplacianHatInverse(rootPosition, 0);
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
double LatentCrfParser::GetMaxSpanningTree(VectorXd &rootSelection, MatrixXd &adjacency, vector<int> &maxSpanningTree) {

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
        adjacency(rowId, columnId) = -1.0 * MultinomialParams::nLog( adjacency(rowId, columnId) ):
        rowId == adjacency.rows() && columnId < adjacency.rows()? 
        -1.0 * rootSelection(columnId) : 
        MultinomialParams::NLOG_ZERO; 
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
vector<int> LatentCrfParser::GetViterbiParse(int sentId, bool conditionOnZ) {
  // build A_{y|x,z} or A_{y|x} (depending on the second parameter) matrix and use matrix tree theoerem to compute Z(x), \sum_{y: (h,m) \in y} p(y|x)
  MatrixXd adjacency, laplacianHat, laplacianHatInverse;
  VectorXd rootSelection;
  double laplacianHatDeterminant = -1;
  vector<int> maxSpanTree;

  auto tokens = GetObservableDetailsSequence(sentId);
  if(tokens.size() > learningInfo.maxSequenceLength) {
    maxSpanTree.resize(tokens.size(), -1);
    return maxSpanTree;
  }

  BuildMatrices(sentId, rootSelection, adjacency, laplacianHat, 
                laplacianHatInverse, laplacianHatDeterminant, conditionOnZ);
  
  // we currently select the max spanning tree without taking into consideration the root selection weight
  if(adjacency.rows() == 1) {
    maxSpanTree.push_back(-1); // the single word in the sentence must be root
    return maxSpanTree;
  }
  GetMaxSpanningTree(rootSelection, adjacency, maxSpanTree);
  
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

