#include "LatentCrfPosTagger.h"

using namespace std;

vector<int>& LatentCrfPosTagger::GetObservableSequence(int exampleId) {
  if(testingMode) {
    assert(exampleId < testData.size());
    return testData[exampleId];
  } else {
    assert(exampleId < data.size());
    return data[exampleId];
  }
}

// singleton
LatentCrfModel* LatentCrfPosTagger::GetInstance(const string &textFilename, 
						const string &outputPrefix, 
						LearningInfo &learningInfo, 
						unsigned NUMBER_OF_LABELS, 
						unsigned FIRST_LABEL_ID) {
  if(!instance) {
    instance = new LatentCrfPosTagger(textFilename, outputPrefix, learningInfo, NUMBER_OF_LABELS, FIRST_LABEL_ID);
  } else {
    cerr << "A LatentCrfPosTagger object has already been initialized" << endl;
  }
  return instance;
}

LatentCrfModel* LatentCrfPosTagger::GetInstance() {
  if(!instance) {
    assert(false);
  }
  return instance;
}

LatentCrfPosTagger::LatentCrfPosTagger(const string &textFilename, 
				       const string &outputPrefix, 
				       LearningInfo &learningInfo, 
				       unsigned NUMBER_OF_LABELS, 
				       unsigned FIRST_LABEL_ID) : LatentCrfModel(textFilename, 
										 outputPrefix, 
										 learningInfo, 
										 FIRST_LABEL_ID,
										 LatentCrfModel::Task::POS_TAGGING) {
  // set constants
  LatentCrfModel::START_OF_SENTENCE_Y_VALUE = FIRST_LABEL_ID - 1;
  this->FIRST_ALLOWED_LABEL_VALUE = FIRST_LABEL_ID;
  assert(START_OF_SENTENCE_Y_VALUE > 0);


  // POS tag yDomain
  unsigned latentClasses = NUMBER_OF_LABELS;
  assert(latentClasses > 1);
  this->yDomain.insert(LatentCrfModel::START_OF_SENTENCE_Y_VALUE); // the conceptual yValue of word at position -1 in a sentence
  for(unsigned i = 0; i < latentClasses; i++) {
    this->yDomain.insert(LatentCrfModel::START_OF_SENTENCE_Y_VALUE + i + 1);
  }
  // zero is reserved for FST epsilon
  assert(this->yDomain.count(0) == 0);

  // words zDomain
  for(map<int,string>::const_iterator vocabIter = vocabEncoder.intToToken.begin();
      vocabIter != vocabEncoder.intToToken.end();
      vocabIter++) {
    if(vocabIter->second == "_unk_") {
      continue;
    }
    this->zDomain.insert(vocabIter->first);
  }
  // zero is reserved for FST epsilon
  assert(this->zDomain.count(0) == 0);
  
  // read and encode data
  data.clear();
  vocabEncoder.Read(textFilename, data);
  examplesCount = data.size();
  
  // bool vectors indicating which feature types to use
  assert(enabledFeatureTypes.size() == 0);
  // features 1-50 are reserved for wordalignment
  for(int i = 0; i <= 50; i++) {
    enabledFeatureTypes.push_back(false);
  }
  // features 51-100 are reserved for latentCrf model
  for(int i = 51; i < 100; i++) {
    enabledFeatureTypes.push_back(false);
  }
  enabledFeatureTypes[51] = true;   // y_i:y_{i-1}
  //  enabledFeatureTypes[52] = true; // y_i:x_{i-2}
  enabledFeatureTypes[53] = true; // y_i:x_{i-1}
  enabledFeatureTypes[54] = true;   // y_i:x_i
  enabledFeatureTypes[55] = true; // y_i:x_{i+1}
  //enabledFeatureTypes[56] = true; // y_i:x_{i+2}
  enabledFeatureTypes[57] = true; // y_i:i
  //  enabledFeatureTypes[58] = true;
  //  enabledFeatureTypes[59] = true;
  //  enabledFeatureTypes[60] = true;
  //  enabledFeatureTypes[61] = true;
  //  enabledFeatureTypes[62] = true;
  //  enabledFeatureTypes[63] = true;
  //  enabledFeatureTypes[64] = true;
  //  enabledFeatureTypes[65] = true;
  enabledFeatureTypes[66] = true; // y_i:(|x|-i)
  enabledFeatureTypes[67] = true; // capital and i != 0
  //enabledFeatureTypes[68] = true;
  enabledFeatureTypes[69] = true; // coarse hash functions
  //enabledFeatureTypes[70] = true;
  //enabledFeatureTypes[71] = true; // y_i:x_{i-1} where x_{i-1} is closed vocab
  //enabledFeatureTypes[72] = true;
  //enabledFeatureTypes[73] = true; // y_i:x_{i+1} where x_{i+1} is closed vocab
  //enabledFeatureTypes[74] = true;
  //enabledFeatureTypes[75] = true; // y_i

  // initialize (and normalize) the log theta params to gaussians
  InitTheta();

  // make sure all slaves have the same theta values
  BroadcastTheta(0);

  // persist initial parameters
  assert(learningInfo.iterationsCount == 0);
  if(learningInfo.iterationsCount % learningInfo.persistParamsAfterNIteration == 0 && learningInfo.mpiWorld->rank() == 0) {
    stringstream thetaParamsFilename;
    thetaParamsFilename << outputPrefix << ".initial.theta";
    if(learningInfo.debugLevel >= DebugLevel::REDICULOUS) {
      cerr << "persisting theta parameters after iteration " << learningInfo.iterationsCount << " at " << thetaParamsFilename.str() << endl;
    }
    PersistTheta(thetaParamsFilename.str());
  }
  
  // initialize the lambda parameters
  // add all features in this data set to lambda.params
  InitLambda();
}

LatentCrfPosTagger::~LatentCrfPosTagger() {}

void LatentCrfPosTagger::InitTheta() {
  if(learningInfo.mpiWorld->rank() == 0 && learningInfo.debugLevel >= DebugLevel::ESSENTIAL) {
    cerr << "master" << learningInfo.mpiWorld->rank() << ": initializing thetas...";
  }

  // first initialize nlogthetas to unnormalized gaussians
  nLogThetaGivenOneLabel.params.clear();
  for(set<int>::const_iterator yDomainIter = yDomain.begin(); 
      yDomainIter != yDomain.end(); yDomainIter++) {
    for(set<int>::const_iterator zDomainIter = zDomain.begin(); 
        zDomainIter != zDomain.end(); zDomainIter++) {
      nLogThetaGivenOneLabel.params[*yDomainIter][*zDomainIter] = abs(gaussianSampler.Draw());
    }
  }
  
  // then normalize them
  MultinomialParams::NormalizeParams(nLogThetaGivenOneLabel);
  if(learningInfo.mpiWorld->rank() == 0) {
    cerr << "done" << endl;
  }
}

// add constrained features here and set their weights by hand. those weights will not be optimized.
void LatentCrfPosTagger::AddConstrainedFeatures() {
  if(learningInfo.debugLevel >= DebugLevel::CORPUS) {
    cerr << "adding constrained lambda features..." << endl;
  }
  FastSparseVector<double> activeFeatures;
  int yI, xI;
  int yIM1_dummy, index; // we don't really care
  vector<int> x;
  string xIString;
  vector<bool> constrainedFeatureTypes(lambda->COUNT_OF_FEATURE_TYPES, false);
  for(int i = 0; i < learningInfo.constraints.size(); i++) {
    switch(learningInfo.constraints[i].type) {
      // constrains the latent variable corresponding to certain types
    case ConstraintType::yIExclusive_xIString:
      // we only want to constrain one specific feature type
      std::fill(constrainedFeatureTypes.begin(), constrainedFeatureTypes.end(), false);
      constrainedFeatureTypes[54] = true;
      // parse the constraint
      xIString.clear();
      learningInfo.constraints[i].GetFieldsOfConstraintType_yIExclusive_xIString(yI, xIString);
      xI = vocabEncoder.Encode(xIString);
      // fire positively constrained features
      x.clear();
      x.push_back(xI);
      yIM1_dummy = yI; // we don't really care
      index = 0; // we don't really care
      activeFeatures.clear();
      // start hack 
      SetTestExample(x);
      testingMode = true;
      FireFeatures(yI, yIM1_dummy, 0, index, constrainedFeatureTypes, activeFeatures);
      testingMode = false;
      // end hack
      // set appropriate weights to favor those parameters
      for(FastSparseVector<double>::iterator featureIter = activeFeatures.begin(); featureIter != activeFeatures.end(); ++featureIter) {
	lambda->UpdateParam(featureIter->first, REWARD_FOR_CONSTRAINED_FEATURES);
      }
      // negatively constrained features (i.e. since xI is constrained to get the label yI, any other label should be penalized)
      for(set<int>::const_iterator yDomainIter = yDomain.begin(); yDomainIter != yDomain.end(); yDomainIter++) {
	if(*yDomainIter == yI) {
	  continue;
	}
	// fire the negatively constrained features
	activeFeatures.clear();
	// start hack 
	SetTestExample(x);
	testingMode = true;
	FireFeatures(*yDomainIter, yIM1_dummy, 0, index, constrainedFeatureTypes, activeFeatures);
	testingMode = false;
	// end hack
	// set appropriate weights to penalize those parameters
	for(FastSparseVector<double>::iterator featureIter = activeFeatures.begin(); featureIter != activeFeatures.end(); ++featureIter) {
	  lambda->UpdateParam(featureIter->first, PENALTY_FOR_CONSTRAINED_FEATURES);
	}   
      }
      break;
    case ConstraintType::yI_xIString:
      // we only want to constrain one specific feature type
      std::fill(constrainedFeatureTypes.begin(), constrainedFeatureTypes.end(), false);
      constrainedFeatureTypes[54] = true;
      // parse the constraint
      xIString.clear();
      learningInfo.constraints[i].GetFieldsOfConstraintType_yI_xIString(yI, xIString);
      xI = vocabEncoder.Encode(xIString);
      // fire positively constrained features
      x.clear();
      x.push_back(xI);
      yIM1_dummy = yI; // we don't really care
      index = 0; // we don't really care
      activeFeatures.clear();
      // start hack
      SetTestExample(x);
      testingMode = true;
      FireFeatures(yI, yIM1_dummy, 0, index, constrainedFeatureTypes, activeFeatures);
      testingMode = false;
      // end hack
      // set appropriate weights to favor those parameters
      for(FastSparseVector<double>::iterator featureIter = activeFeatures.begin(); featureIter != activeFeatures.end(); ++featureIter) {
	lambda->UpdateParam(featureIter->first, REWARD_FOR_CONSTRAINED_FEATURES);
      }
      break;
    default:
      assert(false);
      break;
    }
  }
  // take note of the number of constrained lambda parameters. use this to limit optimization to non-constrained params
  countOfConstrainedLambdaParameters = lambda->GetParamsCount();
  if(learningInfo.debugLevel >= DebugLevel::CORPUS) {
    cerr << "done adding constrainted lambda features. Count:" << lambda->GetParamsCount() << endl;
  }
}

void LatentCrfPosTagger::SetTestExample(vector<int> &tokens) {
  testData.clear();
  testData.push_back(tokens);
}

void LatentCrfPosTagger::Label(vector<int> &tokens, vector<int> &labels) {
  assert(labels.size() == 0); 
  assert(tokens.size() > 0);
  testingMode = true;

  // hack to reuse the code that manipulates the fst
  SetTestExample(tokens);
  unsigned sentId = 0;
  
  fst::VectorFst<FstUtils::LogArc> fst;
  vector<FstUtils::LogWeight> alphas, betas;
  BuildThetaLambdaFst(sentId, tokens, fst, alphas, betas);
  fst::VectorFst<FstUtils::StdArc> fst2, shortestPath;
  fst::ArcMap(fst, &fst2, FstUtils::LogToTropicalMapper());
  fst::ShortestPath(fst2, &shortestPath);
  std::vector<int> dummy;
  FstUtils::LinearFstToVector(shortestPath, dummy, labels);
  assert(labels.size() == tokens.size());
}
