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

  // encode the null token which is conventionally added to the beginning of the src sentnece. 
  ROOT_STR = "__ROOT__";
  ROOT_ID = vocabEncoder.Encode(ROOT_STR);
  assert(ROOT_ID != vocabEncoder.UnkInt());
  
  // read and encode data
  sents.clear();
  vocabEncoder.Read(textFilename, sents);
  assert(sents.size() > 0);
  examplesCount = sents.size();

  if(learningInfo.mpiWorld->rank() == 0 && wordPairFeaturesFilename.size() > 0) {
    lambda->LoadPrecomputedFeaturesWith2Inputs(wordPairFeaturesFilename);
  }

  // master signals to slaves that he's done
  if(learningInfo.mpiWorld->rank() == 0) {
    bool vocabEncoderIsReady;
    boost::mpi::broadcast<bool>(*learningInfo.mpiWorld, vocabEncoderIsReady, 0);
  }

  // initialize (and normalize) the log theta params to gaussians
  InitTheta();
  if(initialThetaParamsFilename.size() > 0) {
    //assert(nLogThetaGivenOneLabel.params.size() == 0);
    if(learningInfo.mpiWorld->rank() == 0) {
      cerr << "initializing theta params from " << initialThetaParamsFilename << endl;
    }
    MultinomialParams::LoadParams(initialThetaParamsFilename, nLogThetaGivenOneLabel, vocabEncoder, true, true);
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

  if(learningInfo.mpiWorld->rank() == 0 && learningInfo.debugLevel >= DebugLevel::CORPUS) {
    cerr << "master" << learningInfo.mpiWorld->rank() << ": initializing thetas...";
  }

  assert(sents.size() > 0);

  // first initialize nlogthetas to unnormalized gaussians
  nLogThetaGivenOneLabel.params.clear();
  for(unsigned sentId = 0; sentId < sents.size(); ++sentId) {
    vector<int64_t> &sent = sents[sentId];
    vector<int64_t> &reconstructedSent = sents[sentId];
    for(unsigned i = 0; i < sent.size(); ++i) {
      auto parentToken = sent[i];
      for(unsigned j = 0; j < reconstructedSent.size(); ++j) {
        if(i == j) { continue; }
	auto childToken = reconstructedSent[j];
	if(learningInfo.initializeThetasWithGaussian) {
          nLogThetaGivenOneLabel.params[parentToken][childToken] = abs(gaussianSampler.Draw());
        } else if (learningInfo.initializeThetasWithUniform || learningInfo.initializeThetasWithModel1) {
          nLogThetaGivenOneLabel.params[parentToken][childToken] = 1;
        }
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

vector<int64_t>& LatentCrfParser::GetReconstructedObservableSequence(int exampleId) {
  if(testingMode) {
    return testSents[exampleId];
  } else {
    // refactor: this following line does not logically belong here
    lambda->learningInfo->currentSentId = exampleId;

    assert(exampleId < sents.size());
    return sents[exampleId];
  }
}

vector<int64_t>& LatentCrfParser::GetObservableSequence(int exampleId) {
  if(testingMode) {
    assert(exampleId < testSents.size());
    return testSents[exampleId];
  } else {
    lambda->learningInfo->currentSentId = exampleId;
    assert(exampleId < sents.size());
    return sents[exampleId];
  }
}

vector<int64_t>& LatentCrfParser::GetObservableContext(int exampleId) { 
  if(testingMode) {
    assert(exampleId < testSents.size());
    return testSents[exampleId];
  } else {
    assert(exampleId < sents.size());
    return sents[exampleId];
  }
}

void LatentCrfParser::SetTestExample(vector<int64_t> &sent) {
  testSents.clear();
  testSents.push_back(sent);
}

void LatentCrfParser::Label(vector<int64_t> &tokens, vector<int> &labels) {

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
    if(exampleId % learningInfo.mpiWorld->size() != learningInfo.mpiWorld->rank()) {
      if(learningInfo.mpiWorld->rank() == 0){
        string labelSequence;
        learningInfo.mpiWorld->recv(exampleId % learningInfo.mpiWorld->size(), 0, labelSequence);
        labelsFile << labelSequence;
      }
      continue;
    }

    //std::vector<int64_t> &sent = GetObservableSequence(exampleId);
    std::vector<int> labels = GetViterbiParse(exampleId, true);
    
    // TODO: depparse. fix how the viterbi parses are written to file
    stringstream ss;
    for(unsigned i = 0; i < labels.size(); ++i) {
      // determine the alignment (i.e. src position) for this tgt position (i)
      int parent = labels[i] - ROOT_POSITION;
      ss << parent << " ";
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

int64_t LatentCrfParser::GetContextOfTheta(unsigned sentId, int y) {
  vector<int64_t> &sent = GetObservableContext(sentId);
  if(y == ROOT_POSITION) {
    return LatentCrfParser::ROOT_ID;
  } else {
    assert(y - ROOT_POSITION - 1 < sent.size());
    assert(y - ROOT_POSITION - 1 >= 0);
    return sent[y - ROOT_POSITION - 1];
  }
}

// build the matrixes which can be used to marginalize proper dependency trees as of Koo et al 2007.
void LatentCrfParser::BuildMatrices(const unsigned sentId,
                                    MatrixXd *adjacency,
                                    MatrixXd *laplacianHat,
                                    bool conditionOnZ) {
  assert(adjacency == 0);
  assert(laplacianHat == 0);

  // build A_{y|x} matrix and use matrix tree theoerem to compute Z(x), \sum_{y: (h,m) \in y} p(y|x)
  auto tokens = GetObservableSequence(sentId);
  auto reconstructedTokens = GetReconstructedObservableSequence(sentId);
  assert(tokens.size() == reconstructedTokens.size());
  // adjacency matrix A(y|x) in (Koo et al. 2007)
  adjacency = new MatrixXd(tokens.size(), tokens.size());
  for(unsigned headPosition = 0; headPosition < tokens.size(); ++headPosition) {
    for(unsigned childPosition = 0; childPosition < tokens.size(); ++childPosition) {
      double multinomialTerm = conditionOnZ? nLogThetaGivenOneLabel[reconstructedTokens[headPosition]][reconstructedTokens[childPosition]]: 0.0;
      FastSparseVector<double> activeFeatures;
      lambda->FireFeatures(tokens[headPosition], tokens[childPosition], activeFeatures);
      (*adjacency)(headPosition, childPosition) = 
        (headPosition == childPosition)? 0.0:
        MultinomialParams::nExp( multinomialTerm + lambda->DotProduct( activeFeatures ) );
    }
  }
  // root selection scores r(y|x) in (Koo et al. 2007)
  VectorXd rootScores(tokens.size());
  for(unsigned rootPosition = 0; rootPosition < tokens.size(); ++rootPosition) {
    double multinomialTerm = conditionOnZ? 
      nLogThetaGivenOneLabel[LatentCrfParser::ROOT_ID][reconstructedTokens[rootPosition]]: 
      0.0;
    FastSparseVector<double> activeFeatures;
    lambda->FireFeatures(LatentCrfParser::ROOT_ID, tokens[rootPosition], activeFeatures);
    rootScores(rootPosition) = MultinomialParams::nExp( lambda->DotProduct( activeFeatures ) );
  }
  // laplacian matrix L(y|x) in (Koo et al. 2007)
  MatrixXd laplacian = *adjacency;
  for(auto rowIndex = 0; rowIndex < tokens.size(); ++rowIndex) {
    laplacian(rowIndex, rowIndex) = laplacian.row(rowIndex).array().sum();
  }
  // modified laplacian matrix to allow for O(n^3) inference; \hat{L}(y|x) in (Koo et al. 2007)
  laplacianHat = new MatrixXd(laplacian);
  laplacianHat->row(0) = rootScores;
}                                    

// returns -log p(z|x)
double LatentCrfParser::UpdateThetaMleForSent(const unsigned sentId, 
  				     MultinomialParams::ConditionalMultinomialParam<int64_t> &mle, 
  				     boost::unordered_map<int64_t, double> &mleMarginals) {

  cerr << "LatentCrfParser's impelmentation of LatentCrfModel::UpdateThetaMleForSent" << endl;
  std::cerr << "sentId = " << sentId << endl;
  assert(sentId < examplesCount);
  
  // build A_{y|x} matrix and use matrix tree theoerem to compute Z(x), \sum_{y: (h,m) \in y} p(y|x)
  // TODO: WE DON'T REALLY NEED THIS MATRIX; AFTERALL, WHO CARES ABOUT COMPUTING LIKELIHOOD? 
  MatrixXd *yGivenXAdjacency = 0, *yGivenXLaplacianHat = 0;
  BuildMatrices(sentId, yGivenXAdjacency, yGivenXLaplacianHat, false);
  double Z = yGivenXLaplacianHat->determinant();
  //MatrixXd yGivenXLaplacianHatInverse = yGivenXLaplacianHat->inverse();
  
  // build A_{y|x,z} matrix and use matrix tree theorem to compute C(x), marginal(h,m;y|z,x)=\sum_{y:(h,m)\in y} p(y|x,z)
  MatrixXd *yGivenXZAdjacency = 0, *yGivenXZLaplacianHat = 0;
  BuildMatrices(sentId, yGivenXZAdjacency, yGivenXZLaplacianHat, true);
  double C = yGivenXZLaplacianHat->determinant();
  MatrixXd yGivenXZLaplacianHatInverse = yGivenXZLaplacianHat->inverse();
  assert(C < Z);

  auto reconstructedTokens = GetReconstructedObservableSequence(sentId);
  
  // for (h,m) \in \cal{T}_{np}^s: mle[h][m] += nLogThetaGivenOneLabel[h][m] * marginal(h,m;y|z,x)
  for(unsigned rootPosition = 0; rootPosition < yGivenXZLaplacianHat->rows(); ++rootPosition) {
    // marginal probability of making this decision; \mu_{0,m} in (Koo et al. 2007)
    double marginal = (*yGivenXZLaplacianHat)(0,rootPosition) * yGivenXZLaplacianHatInverse(rootPosition,0);
    mle[LatentCrfParser::ROOT_ID][reconstructedTokens[rootPosition]] += nLogThetaGivenOneLabel[LatentCrfParser::ROOT_ID][reconstructedTokens[rootPosition]] * marginal;
    mleMarginals[LatentCrfParser::ROOT_ID] += nLogThetaGivenOneLabel[LatentCrfParser::ROOT_ID][reconstructedTokens[rootPosition]] * marginal;
  }
  for(unsigned headPosition = 0; headPosition < yGivenXZLaplacianHat->rows(); ++headPosition) {
    for(unsigned childPosition = 0; childPosition < yGivenXZLaplacianHat->cols(); ++childPosition) {
      double marginal = childPosition == 0? 0.0 :
        (*yGivenXZAdjacency)(headPosition, childPosition) * 
        yGivenXZLaplacianHatInverse(childPosition, childPosition);
      marginal -= headPosition == 0? 0.0 :
        (*yGivenXZAdjacency)(headPosition, childPosition) * 
        yGivenXZLaplacianHatInverse(childPosition, headPosition);
      mle[reconstructedTokens[headPosition]][reconstructedTokens[childPosition]] += marginal * nLogThetaGivenOneLabel[reconstructedTokens[headPosition]][reconstructedTokens[childPosition]];
      mleMarginals[reconstructedTokens[headPosition]] += marginal * nLogThetaGivenOneLabel[reconstructedTokens[headPosition]][reconstructedTokens[childPosition]];
    }
  }

  // nlog p(z|x)
  return MultinomialParams::nLog(C / Z); 
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
    if( GetObservableSequence(sentId).size() > learningInfo.maxSequenceLength ) {
      continue;
    }
    
    // build A_{y|x} matrix and use matrix tree theoerem to compute Z(x), \sum_{y: (h,m) \in y} p(y|x)
    MatrixXd *yGivenXAdjacency = 0, *yGivenXLaplacianHat = 0;
    BuildMatrices(sentId, yGivenXAdjacency, yGivenXLaplacianHat, false);
    double Z = yGivenXLaplacianHat->determinant();
    MatrixXd yGivenXLaplacianHatInverse = yGivenXLaplacianHat->inverse();
    
    // build A_{y|x,z} matrix and use matrix tree theorem to compute C(x), marginal(h,m;y|z,x)=\sum_{y:(h,m)\in y} p(y|x,z)
    MatrixXd *yGivenXZAdjacency = 0, *yGivenXZLaplacianHat = 0;
    BuildMatrices(sentId, yGivenXZAdjacency, yGivenXZLaplacianHat, true);
    double C = yGivenXZLaplacianHat->determinant();
    MatrixXd yGivenXZLaplacianHatInverse = yGivenXZLaplacianHat->inverse();
    assert(C < Z);
    
    auto tokens = GetObservableSequence(sentId);
    // for (h,m) in \cal{T}_{np}^s:
    //   for k in f(x,h,m):
    //     dll/d\lambda_k += f_k(x,h,m) * [marginal(h,m;y|z,x)-marginal(h,m;y|x)]
    for(unsigned headPosition = 0; headPosition < tokens.size(); ++headPosition) {
      for(unsigned childPosition = 0; childPosition < tokens.size(); ++childPosition) {
        FastSparseVector<double> activeFeatures;
        lambda->FireFeatures(tokens[headPosition], tokens[childPosition], activeFeatures);
        double marginalGivenXZ = childPosition == 0? 0.0 :
          (*yGivenXZAdjacency)(headPosition, childPosition) * 
          yGivenXZLaplacianHatInverse(childPosition, childPosition);
        marginalGivenXZ -= headPosition == 0? 0.0 :
          (*yGivenXZAdjacency)(headPosition, childPosition) * 
          yGivenXZLaplacianHatInverse(childPosition, headPosition);
        double marginalGivenX = childPosition == 0? 0.0 :
          (*yGivenXAdjacency)(headPosition, childPosition) * 
          yGivenXLaplacianHatInverse(childPosition, childPosition);
        marginalGivenX -= headPosition == 0? 0.0 :
          (*yGivenXAdjacency)(headPosition, childPosition) * 
          yGivenXLaplacianHatInverse(childPosition, headPosition);
        double marginal = marginalGivenXZ - marginalGivenX;
        for(auto featIter = activeFeatures.begin(); featIter != activeFeatures.end(); ++featIter) {
          derivativeWRTLambda[featIter->first] += featIter->second * marginal;
        }
      }
    }
    
    // don't forget to also update the gradients of root selection features
    for(unsigned rootPosition = 0; rootPosition < tokens.size(); ++rootPosition) {
      FastSparseVector<double> activeFeatures;
      lambda->FireFeatures(ROOT_ID, tokens[rootPosition], activeFeatures);
      double marginalGivenXZ = (*yGivenXZLaplacianHat)(0, rootPosition) * yGivenXZLaplacianHatInverse(rootPosition, 0);
      double marginalGivenX =   (*yGivenXLaplacianHat)(0, rootPosition) * yGivenXLaplacianHatInverse(rootPosition, 0);
      double marginal = marginalGivenXZ - marginalGivenX;
      for(auto featIter = activeFeatures.begin(); featIter != activeFeatures.end(); ++featIter) {
        derivativeWRTLambda[featIter->first] += featIter->second * marginal;
      } 
    }

    // update the loglikelihood
    assert(!learningInfo.useEarlyStopping);
    double nLogC = MultinomialParams::nLog(C), nLogZ = MultinomialParams::nLog(Z);
    objective += nLogC;
    objective -= nLogZ;
    
    // keep an eye on bad numbers
    if(std::isnan(nLogZ) || std::isinf(nLogZ)) {
      assert(false);
    } 

    if(nLogC < nLogZ) {
      cerr << "this must be a bug. nLogC always be >= nLogZ. " << endl;
      cerr << "nLogC = " << nLogC << endl;
      cerr << "nLogZ = " << nLogZ << endl;
    }
    
    // debug info
    if(learningInfo.debugLevel >= DebugLevel::MINI_BATCH && sentId % learningInfo.nSentsPerDot == 0) {
      cerr << ".";
    }
  } // end of training examples 

  cerr << learningInfo.mpiWorld->rank() << "|";
  
  //  cerr << "ending LatentCrfModel::ComputeNllZGivenXAndLambdaGradient" << endl;

  return objective;
}

// run Tarjan's implementation of Chiu-Liu-Edmonds for maximum spanning trees
double LatentCrfParser::GetMaxSpanningTree(MatrixXd &adjacency, vector<int> &maxSpanningTree, int &root) {

  int n_vertices = adjacency.rows();
  complete_graph      g(n_vertices);
  multi_array<double, 2> weights(extents[n_vertices][n_vertices]);
  vector<Vertex>      roots; // = {0, 1} you can use this vector to specify particular root(s)
  vector<Edge>        branching;
  double         ans;

  // set weights (doubles)
  for(unsigned rowId = 0; rowId < adjacency.rows(); ++rowId) {
    for(unsigned columnId = 0; columnId < adjacency.cols(); ++columnId) {
      weights[rowId][columnId] = adjacency(rowId, columnId);
    }
  }
  
  // run edmonds algorithm for a few cases. The cases will be
  // the cross product of the following properties:
  // optimum-is-maximum x attempt-to-span x num-specified-roots
  // where num-specified roots is either 0, 1, or 2. Also the
  // specified roots are either none, the vertex 0, or the
  // vertices 0 and 1.
  edmonds_optimum_branching<true, true, true>
    (g, identity_property_map(), weights.origin(),
     roots.begin(), roots.end(), back_inserter(branching));

  // initialize the mst; everyone is a root
  maxSpanningTree.resize(n_vertices);
  BOOST_FOREACH(int &parent, maxSpanningTree)
    {
      parent = -1; // parent = -1 indicates a root vertix
    }

  // modify parents of nonroot vertices
  ans = 0.0;
  unsigned edgesCounter = 0;
  BOOST_FOREACH (Edge e, branching)
    {
      edgesCounter++;
      maxSpanningTree[target(e, g)] = source(e, g);
      ans += weights[source(e, g)][target(e, g)];
    }
  
  assert(edgesCounter == n_vertices - 1);

  // set the root
  root = ( find(maxSpanningTree.begin(), maxSpanningTree.end(), -1) - maxSpanningTree.begin() );
  assert( root < n_vertices );

  return ans;
}

vector<int> LatentCrfParser::GetViterbiParse(int sentId, bool conditionOnZ) {
  // build A_{y|x,z} or A_{y|x} (depending on the second parameter) matrix and use matrix tree theoerem to compute Z(x), \sum_{y: (h,m) \in y} p(y|x)
  MatrixXd *adjacency = 0, *laplacianHat = 0;
  BuildMatrices(sentId, adjacency, laplacianHat, conditionOnZ);
  
  vector<int> maxSpanTree;
  int root;
  double maxSpanTreeWeight = GetMaxSpanningTree(*adjacency, maxSpanTree, root);
  double rootSelectionWeight = (*laplacianHat)(0, root);

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
