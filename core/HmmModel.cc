#include "HmmModel.h"

using namespace std;
using namespace fst;
using namespace MultinomialParams;
using namespace boost;

HmmModel2::HmmModel2(const string &textFilename, 
		     const string &outputPrefix, 
		     LearningInfo &learningInfo,
		     unsigned numberOfLabels,
		     unsigned firstLabelId) : 
  UnsupervisedSequenceTaggingModel(textFilename, learningInfo),
  gaussianSampler(0.0, 1.0),
  START_OF_SENTENCE_Y_VALUE(firstLabelId-1),
  FIRST_ALLOWED_LABEL_VALUE(firstLabelId) {
    
  this->outputPrefix = outputPrefix;
  this->learningInfo = &learningInfo;
  
  assert(numberOfLabels > 1);
  assert(firstLabelId > 1);
  yDomain.insert(START_OF_SENTENCE_Y_VALUE); // the conceptual yValue of word at position -1 in a sentence
  for(unsigned labelId = START_OF_SENTENCE_Y_VALUE + 1; labelId < START_OF_SENTENCE_Y_VALUE + numberOfLabels + 1 ; labelId++) {
    yDomain.insert(labelId);
  }

  // populate the X domain with all types in the vocabEncoder
  for(auto vocabIter = vocabEncoder.intToToken->begin();
      vocabIter != vocabEncoder.intToToken->end();
      vocabIter++) {
    if(vocabIter->second == "_unk_") {
      continue;
    }
    xDomain.insert(vocabIter->first);
  }
  // zero is reserved for FST epsilon
  assert(xDomain.count(0) == 0);
  
  // populate the observations vector with encoded sentences
  vocabEncoder.Read(textFilename, observations);
  
  
    
  if(!learningInfo.neuralRepFilename.empty()) {
      neuralRep.clear();
      LatentCrfModel::readNeuralRep(learningInfo.neuralRepFilename, neuralRep);      
      for(auto sentence:neuralRep) {
          cerr << "sen length:\t" << sentence.size() << endl;
          for(auto emb:sentence) {
              cerr << emb.mean() << " ";
          }
          cerr << endl;
      }
  }

  // initialize theta and gamma parameters
  InitParams();
}

// gaussian initialization of the multinomial params
void HmmModel2::InitParams(){
  for(set<int>::const_iterator toYIter = yDomain.begin(); toYIter != yDomain.end(); toYIter++) {
    if(*toYIter == START_OF_SENTENCE_Y_VALUE) {
      continue;
    }
    for(set<int>::const_iterator fromYIter = yDomain.begin(); fromYIter != yDomain.end(); fromYIter++) {
      nlogGamma[*fromYIter][*toYIter] = fabs(gaussianSampler.Draw());
    }
    for(set<int64_t>::const_iterator xIter = xDomain.begin(); xIter != xDomain.end(); xIter++) {
      nlogTheta[*toYIter][*xIter] = fabs(gaussianSampler.Draw());
    }
  }
  nlogTheta.GaussianInit();
  nlogGamma.GaussianInit();
  //NormalizeParams(nlogTheta, 1.0, false, true);
  //NormalizeParams(nlogGamma, 1.0, false, true);
    
  if(learningInfo->debugLevel >= DebugLevel::REDICULOUS) {
    cerr << "nlogTheta params: " << endl;
    nlogTheta.PrintParams();
    cerr << "nlogGamma params: " << endl;
    nlogGamma.PrintParams();
  }
  
  if(!learningInfo->neuralRepFilename.empty()) {
      assert(neuralMean.size()==0);
      assert(neuralVar.size()==0);
      for(auto y: yDomain) {
          neuralMean[y].setRandom(Eigen::NEURAL_SIZE,1);
          neuralVar[y].setIdentity();
      }
      cerr << "initialized neural means\n";
  }
}

void HmmModel2::PersistParams(string &prefix) {
  if(prefix.size() == 0) {
    prefix = outputPrefix + ".hmm2.final";
  }
  const string thetaFilename(prefix + ".nlogTheta");
  const string gammaFilename(prefix + ".nlogGamma");
  MultinomialParams::PersistParams(thetaFilename, nlogTheta, vocabEncoder, false, true);
  MultinomialParams::PersistParams(gammaFilename, nlogGamma, vocabEncoder, false, false);
}

// builds the lattice of all possible label sequences
void HmmModel2::BuildThetaGammaFst(vector<int64_t> &x, VectorFst<FstUtils::LogArc> &fst, unsigned sentId) {
  // arcs represent a particular choice of y_i at time step i
  // arc weights are - log \theta_{x_i|y_i} - log \gamma_{y_i|y_{i-1}}
    
    if(sentId==55665566) {
        assert(learningInfo->neuralRepFilename.empty());    
    }
    
    vector<Eigen::VectorNeural> neural;
    if(! learningInfo->neuralRepFilename.empty()) {
        neural = GetNeuralSequence(sentId);
    }
    
  assert(fst.NumStates() == 0);
  int startState = fst.AddState();
  fst.SetStart(startState);
  
  // map values of y_{i-1} and y_i to fst states
  map<int, int> yIM1ToState, yIToState;

  yIM1ToState[START_OF_SENTENCE_Y_VALUE] = startState;

  // for each timestep
  for(int i = 0; i < x.size(); i++){

    // timestep i hasn't reached any states yet
    yIToState.clear();
    // from each state reached in the previous timestep
    for(map<int, int>::const_iterator prevStateIter = yIM1ToState.begin();
	prevStateIter != yIM1ToState.end();
	prevStateIter++) {
      
      int fromState = prevStateIter->second;
      int yIM1 = prevStateIter->first;
      // to each possible value of y_i
      for(set<int>::const_iterator yDomainIter = yDomain.begin();
	  yDomainIter != yDomain.end();
	  yDomainIter++) {

	int yI = *yDomainIter;

	// START_OF_SENTENCE_Y_VALUE can only be used for the hypothetical y_{i-1}, so skip it.
	if(yI == START_OF_SENTENCE_Y_VALUE) {
	  continue;
	}

	// compute arc weight
	double arcWeight = nlogGamma[yIM1][yI];
        if(!learningInfo->neuralRepFilename.empty()) {
            arcWeight += getGaussianPDF(yI, neural[i]);
        } else {
            arcWeight += nlogTheta[yI][x[i]];
        }
	if(arcWeight < 0 || std::isinf(arcWeight) || std::isnan(arcWeight)) {
	  cerr << "FATAL ERROR: arcWeight = " << arcWeight << endl << "will terminate." << endl;
	}
	
	// determine whether to add a new state or reuse an existing state which also represent label y_i and timestep i
	int toState;	
	if(yIToState.count(yI) == 0) {
	  toState = fst.AddState();
	  yIToState[yI] = toState;
	  // is it a final state?
	  if(i == x.size() - 1) {
	    fst.SetFinal(toState, FstUtils::LogWeight::One());
	  }
	} else {
	  toState = yIToState[yI];
	}
	// now add the arc
	fst.AddArc(fromState, FstUtils::LogArc(yIM1, yI, arcWeight, toState));	
      }
    }
    // now, that all states reached in step i have already been created, yIM1ToState has become out-of-date. update it
    yIM1ToState = yIToState;
  }
}

// builds the lattice of all possible label sequences, also computes potentials
void HmmModel2::BuildThetaGammaFst(unsigned sentId, VectorFst<FstUtils::LogArc> &fst, vector<FstUtils::LogWeight> &alphas, vector<FstUtils::LogWeight> &betas) {

  // first, build the lattice
  BuildThetaGammaFst(observations[sentId], fst, sentId);


  // then compute forward/backward state potentials
  assert(alphas.size() == 0);
  assert(betas.size() == 0);
  ShortestDistance(fst, &alphas, false);
  ShortestDistance(fst, &betas, true);
}

void HmmModel2::UpdateMle(const unsigned sentId,
			  const VectorFst<FstUtils::LogArc> &fst, 
			  const vector<FstUtils::LogWeight> &alphas, 
			  const vector<FstUtils::LogWeight> &betas, 
			  ConditionalMultinomialParam<int64_t> &thetaMle, 
			  ConditionalMultinomialParam<int64_t> &gammaMle){
  vector<int64_t> &x = observations[sentId];
 
  // schedule for visiting states such that we know the timestep for each arc
  set<int> iStates, iP1States;
  iStates.insert(fst.Start());

  // for each timestep
  for(int i = 0; i < x.size(); i++) {
    int xI = x[i];
    
    // from each state at timestep i
    for(set<int>::const_iterator iStatesIter = iStates.begin(); 
	iStatesIter != iStates.end(); 
	iStatesIter++) {
      int fromState = *iStatesIter;

      // for each arc leaving this state
      for(ArcIterator< VectorFst<FstUtils::LogArc> > aiter(fst, fromState); !aiter.Done(); aiter.Next()) {
	const FstUtils::LogArc &arc = aiter.Value();
	int yIM1 = arc.ilabel;
	int yI = arc.olabel;
	const FstUtils::LogWeight &arcWeight = arc.weight;
	int toState = arc.nextstate;

	// compute marginal weight of passing on this arc
	const FstUtils::LogWeight nlogWeight(fst::Times(arcWeight , fst::Times(betas[toState], alphas[fromState])));
	double nlogProb = fst::Divide(nlogWeight, betas[0]).Value();
	if(nlogProb < -1.0 || std::isinf(nlogProb) || std::isnan(nlogProb)) {
	  cerr << "FATAL ERROR: nlogProb = " << nlogProb << " = alpha + arcWeight + beta - betas[0] = " << alphas[fromState].Value() << " + " << arcWeight.Value() << " + " << betas[toState].Value() << " - " << betas[0].Value() << endl << "will terminate." << endl;
	  assert(false);
	}
	// fix precision problems
	if(nlogProb < 0) {
	  nlogProb = 0;
	}
	double prob = MultinomialParams::nExp(nlogProb);
	assert( !std::isinf(prob) && !std::isnan(prob) );
	thetaMle[yI][xI] += prob;
	gammaMle[yIM1][yI] += prob;
	
	// prepare the schedule for visiting states in the next timestep
	iP1States.insert(toState);
      } 
    }
    
    // prepare for next timestep
    iStates = iP1States;
    iP1States.clear();
  }
}

// EM training of the HMM
void HmmModel2::Train(){
  do {
    
    // expectation
    double nloglikelihood = 0;
    ConditionalMultinomialParam<int64_t> thetaMle, gammaMle;
    boost::unordered_map< int64_t, std::vector<Eigen::VectorNeural> > meanPerLabel;
    boost::unordered_map< int64_t, std::vector<LogVal<double >>> nNormalizingConstant;
    for(unsigned sentId = 0; sentId < observations.size(); sentId++) {
      VectorFst<FstUtils::LogArc> fst;
      vector<FstUtils::LogWeight> alphas, betas; 
      BuildThetaGammaFst(sentId, fst, alphas, betas);
      if(!learningInfo->neuralRepFilename.empty()) {
          // cerr << "training with word embeddings\n";
          UpdateMle(sentId, fst, alphas, betas, meanPerLabel, nNormalizingConstant, gammaMle);
      } else {
          UpdateMle(sentId, fst, alphas, betas, thetaMle, gammaMle);
      }
      double sentNlogProb = betas[0].Value();
      if(sentNlogProb < -0.01) {
	cerr << "FATAL ERROR: sentNlogProb = " << sentNlogProb << " in sent #" << sentId << endl << "will terminate." << endl;
	for(unsigned stateId = 0; stateId < betas.size(); stateId++) {
	  cerr << "statedId = " << stateId << ", alpha = " << alphas[stateId] << ", beta = " << betas[stateId] << endl;
	}
	assert(false);
      }
      if(learningInfo->debugLevel >= DebugLevel::SENTENCE) {
	cerr << "rank#" << learningInfo->mpiWorld->rank() << ": sent #" << sentId << ": nlogProb = " << sentNlogProb << endl;
      }
      nloglikelihood += sentNlogProb;
    }

    if(!learningInfo->neuralRepFilename.empty()) {
        NormalizeMleMeanAndUpdateMean(meanPerLabel,
        nNormalizingConstant);
    } else {
        // maximization
        MultinomialParams::NormalizeParams(thetaMle, learningInfo->multinomialSymmetricDirichletAlpha, 
                                       false, true, 
                                       learningInfo->variationalInferenceOfMultinomials);
        nlogTheta = thetaMle;
    }
    MultinomialParams::NormalizeParams(gammaMle, learningInfo->multinomialSymmetricDirichletAlpha, 
                                       false, true, 
                                       learningInfo->variationalInferenceOfMultinomials);
    nlogGamma = gammaMle;

    // check convergence
    learningInfo->logLikelihood.push_back(-1 * nloglikelihood);
    learningInfo->iterationsCount++;
    
    if(learningInfo->debugLevel >= DebugLevel::CORPUS) {
      cerr << "rank #" << learningInfo->mpiWorld->rank() << ": nloglikelihood of this iteration = " << nloglikelihood << endl; 
    }

  } while(!learningInfo->IsModelConverged());
}

void HmmModel2::Label(vector<int64_t> &tokens, vector<int> &labels) {
  //cerr << "inside HmmModel2::Label(vector<int64_t> &tokens, vector<int> &labels)" << endl;
  VectorFst<FstUtils::LogArc> fst;
  BuildThetaGammaFst(tokens, fst, 55665566);

  VectorFst<FstUtils::StdArc> fst2, shortestPath;
  fst::ArcMap(fst, &fst2, FstUtils::LogToTropicalMapper());
  fst::ShortestPath(fst2, &shortestPath);
  std::vector<int> dummy;
  FstUtils::LinearFstToVector(shortestPath, dummy, labels);
  assert(labels.size() == tokens.size());
}
void HmmModel2::Label(vector<int64_t> &tokens, vector<int> &labels, unsigned sentId) {
  //cerr << "inside HmmModel2::Label(vector<int64_t> &tokens, vector<int> &labels)" << endl;
  VectorFst<FstUtils::LogArc> fst;
  BuildThetaGammaFst(tokens, fst, sentId);

  VectorFst<FstUtils::StdArc> fst2, shortestPath;
  fst::ArcMap(fst, &fst2, FstUtils::LogToTropicalMapper());
  fst::ShortestPath(fst2, &shortestPath);
  std::vector<int> dummy;
  FstUtils::LinearFstToVector(shortestPath, dummy, labels);
  assert(labels.size() == tokens.size());
}


  void HmmModel2::Label(vector<vector<string> > &tokens, vector<vector<int> > &labels) {
    assert(labels.size() == 0);
    labels.resize(tokens.size());
    if(!learningInfo->neuralRepFilename.empty()) {
        cerr << "labeling with word embeddings\n";
        for(unsigned i = 0 ; i <tokens.size(); i++) {
            Label(observations[i], labels[i], i);
        }
    }
    else {
        for(unsigned i = 0 ; i <tokens.size(); i++) {
            UnsupervisedSequenceTaggingModel::Label(tokens[i], labels[i]);
        }
    }
  }

void HmmModel2::Label(string &inputFilename, string &outputFilename) {
    std::vector<std::vector<std::string> > tokens;
    StringUtils::ReadTokens(inputFilename, tokens);
    vector<vector<int> > labels;
    Label(tokens, labels);
    StringUtils::WriteTokens(outputFilename, labels);
  }

double HmmModel2::getGaussianPDF(int64_t yi, const Eigen::VectorNeural& zi) {
    if(zi.isConstant(Eigen::NONE)) {
        return 0;
    }
    
    const auto c = -0.5 * Eigen::NEURAL_SIZE * log(2 * M_PI);
    const auto& mean = neuralMean[yi];
    const auto& diff = zi - mean;

    //double inner_product = diff.transpose() *  var_inverse * diff;
    double inner_product = diff.squaredNorm();
    if(std::isinf(inner_product)) {
        cerr << "inner product inf!\n";
        assert(false);
        return 2.0e100;
    }
    double log_pdf = c - 0.5 * inner_product;
    
    return -log_pdf;
}

const vector<Eigen::VectorNeural>& HmmModel2::GetNeuralSequence(int exampleId) {
    assert(exampleId < neuralRep.size());
    return neuralRep[exampleId];
}

void HmmModel2::UpdateMle(const unsigned sentId,
			  const VectorFst<FstUtils::LogArc> &fst, 
			  const vector<FstUtils::LogWeight> &alphas, 
			  const vector<FstUtils::LogWeight> &betas, 
        boost::unordered_map< int64_t, std::vector<Eigen::VectorNeural> > &meanPerLabel,
        boost::unordered_map< int64_t, std::vector<LogVal<double >>> &nNormalizingConstant,
			  ConditionalMultinomialParam<int64_t> &gammaMle){
 
  // schedule for visiting states such that we know the timestep for each arc
  set<int> iStates, iP1States;
  iStates.insert(fst.Start());
    const vector<Eigen::VectorNeural>&x = GetNeuralSequence(sentId);
    const auto zeros = Eigen::VectorNeural::Zero(Eigen::NEURAL_SIZE, 1);
  // for each timestep
  for(int i = 0; i < x.size(); i++) {
    auto xI = x[i];
    
    // from each state at timestep i
    for(set<int>::const_iterator iStatesIter = iStates.begin(); 
	iStatesIter != iStates.end(); 
	iStatesIter++) {
      int fromState = *iStatesIter;

      // for each arc leaving this state
      for(ArcIterator< VectorFst<FstUtils::LogArc> > aiter(fst, fromState); !aiter.Done(); aiter.Next()) {
	const FstUtils::LogArc &arc = aiter.Value();
	int yIM1 = arc.ilabel;
	int yI = arc.olabel;
	const FstUtils::LogWeight &arcWeight = arc.weight;
	int toState = arc.nextstate;

	// compute marginal weight of passing on this arc
	const FstUtils::LogWeight nlogWeight(fst::Times(arcWeight , fst::Times(betas[toState], alphas[fromState])));
	double nlogProb = fst::Divide(nlogWeight, betas[0]).Value();
	if(nlogProb < -1.0 || std::isinf(nlogProb) || std::isnan(nlogProb)) {
	  cerr << "FATAL ERROR: nlogProb = " << nlogProb << " = alpha + arcWeight + beta - betas[0] = " << alphas[fromState].Value() << " + " << arcWeight.Value() << " + " << betas[toState].Value() << " - " << betas[0].Value() << endl << "will terminate." << endl;
	  assert(false);
	}
	// fix precision problems
	if(nlogProb < 0) {
	  nlogProb = 0;
	}
	double prob = MultinomialParams::nExp(nlogProb);
                    meanPerLabel[yI].push_back(xI);
                    nNormalizingConstant[yI].push_back(LogVal<double>(-nlogProb, init_lnx()));
	gammaMle[yIM1][yI] += prob;
	
	// prepare the schedule for visiting states in the next timestep
	iP1States.insert(toState);
      } 
    }
    
    // prepare for next timestep
    iStates = iP1States;
    iP1States.clear();
  }
}

void HmmModel2::NormalizeMleMeanAndUpdateMean(boost::unordered_map< int64_t, std::vector<Eigen::VectorNeural> >& means,
        boost::unordered_map< int64_t, std::vector<LogVal<double>>>& nNormalizingConstant) {

    boost::unordered_map<int64_t, LogVal<double>> sum;
    // init
    for(auto y:yDomain) {
        sum[y] = LogVal<double>::Zero();
    }
    
    // sum
    for (const auto& t : nNormalizingConstant) {
        sum[t.first] += std::accumulate(t.second.begin(), t.second.end(), LogVal<double>::Zero());
    }

    // clear
    for(auto y:yDomain) {
        neuralMean[y].setZero(Eigen::NEURAL_SIZE,1);
     
    }
      
    for (auto y : yDomain) {
        for (auto i = 0; i < means[y].size(); i++) {
            const auto weight = (nNormalizingConstant[y][i] / sum[y]).as_float();
            neuralMean[y] += weight * means[y][i];
        }
    }
}