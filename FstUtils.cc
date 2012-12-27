#include "FstUtils.h"

using namespace fst;

const float FstUtils::LOG_PROBS_MUST_BE_GREATER_THAN_ME = -0.1;

LogPairWeight FstUtils::EncodePairInfinity() {
  return EncodePair(numeric_limits<float>::infinity(), numeric_limits<float>::infinity());
}

LogTripleWeight FstUtils::EncodeTripleInfinity() {
  return EncodeTriple(numeric_limits<float>::infinity(), numeric_limits<float>::infinity(), numeric_limits<float>::infinity());
}

LogQuadWeight FstUtils::EncodeQuadInfinity() {
  return EncodeQuad(numeric_limits<float>::infinity(), 
		    numeric_limits<float>::infinity(), 
		    numeric_limits<float>::infinity(), 
		    numeric_limits<float>::infinity());
}

LogPairWeight FstUtils::EncodePair(float val1, float val2) {
  LogWeight v1, v2;
  v1 = val1;
  v2 = val2;
  return ProductWeight<LogWeight, LogWeight>(v1, v2);
}

LogTripleWeight FstUtils::EncodeTriple(float val1, float val2, float val3) {
  LogWeight v3;
  v3 = val3;
  return LogTripleWeight(EncodePair(val1, val2), v3);
}

LogQuadWeight FstUtils::EncodeQuad(float val1, float val2, float val3, float val4) {
  LogWeight v4;
  v4 = val4;
  return LogQuadWeight(EncodeTriple(val1, val2, val3), v4);
}

string FstUtils::PrintWeight(const TropicalWeightTpl<float>& w) {
  stringstream ss;
  ss << w.Value();
  return ss.str();
}

string FstUtils::PrintWeight(const LogWeight& w) {
  stringstream ss;
  ss << w.Value();
  return ss.str();
}

string FstUtils::PrintWeight(const ProductWeight<LogWeight, LogWeight>& w) {
  stringstream ss;
  ss << "(" << w.Value1().Value() << "," << w.Value2().Value() << ")";
  return ss.str();
}

string FstUtils::PrintWeight(const LogTripleWeight& w) {
  stringstream ss;
  ss << "(" << w.Value1().Value1().Value() 
     << "," << w.Value1().Value2().Value() 
     << "," << w.Value2().Value() 
     << ")";
  return ss.str();
}

string FstUtils::PrintWeight(const LogQuadWeight& w) {
  stringstream ss;
  ss << "(" << w.Value1().Value1().Value1().Value() 
     << "," << w.Value1().Value1().Value2().Value()
     << "," << w.Value1().Value2().Value()
     << "," << w.Value2().Value()
     << ")";
  return ss.str();
}

void FstUtils::DecodePair(const LogPairWeight& w, float& v1, float& v2) {
  v1 = w.Value1().Value();
  v2 = w.Value2().Value();
}

void FstUtils::DecodeTriple(const LogTripleWeight& w, float& v1, float& v2, float& v3) {
  DecodePair(w.Value1(), v1, v2);
  v3 = w.Value2().Value();
}

void FstUtils::DecodeQuad(const LogQuadWeight& w, float& v1, float& v2, float& v3, float& v4) {
  DecodeTriple(w.Value1(), v1, v2, v3);
  v4 = w.Value2().Value();
}

// assumption: the fst has one or more final state
// output: all states which used to be final in the original fst, are not final now, but they go to a new final state with epsion input/output labels and the transition's weight = the original stopping weight of the corresponding state.
void FstUtils::MakeOneFinalState(fst::VectorFst<LogQuadArc>& fst) {
  int finalStateId = fst.AddState();

  for(StateIterator< VectorFst<LogQuadArc> > siter(fst); !siter.Done(); siter.Next()) {
    LogQuadWeight stoppingWeight = fst.Final(siter.Value());
    if(stoppingWeight != LogQuadWeight::Zero()) {
      fst.SetFinal(siter.Value(), LogQuadWeight::Zero());
      fst.AddArc(siter.Value(), LogQuadArc(FstUtils::EPSILON, FstUtils::EPSILON, stoppingWeight, finalStateId));
    }
  }

  fst.SetFinal(finalStateId, LogQuadWeight::One());
}

// assumption: the fst has one or more final state
// output: all states which used to be final in the original fst, are not final now, but they go to a new final state with epsion input/output labels and the transition's weight = the original stopping weight of the corresponding state.
void FstUtils::MakeOneFinalState(fst::VectorFst<LogArc>& fst) {
  int finalStateId = fst.AddState();

  for(StateIterator< VectorFst<LogArc> > siter(fst); !siter.Done(); siter.Next()) {
    LogWeight stoppingWeight = fst.Final(siter.Value());
    if(stoppingWeight != LogWeight::Zero()) {
      fst.SetFinal(siter.Value(), LogWeight::Zero());
      fst.AddArc(siter.Value(), LogArc(FstUtils::EPSILON, FstUtils::EPSILON, stoppingWeight, finalStateId));
    }
  }

  fst.SetFinal(finalStateId, LogWeight::One());
}

// return the id of a final states in this fst. if no final state found, returns -1.
int FstUtils::FindFinalState(const fst::VectorFst<LogQuadArc>& fst) {
  for(StateIterator< VectorFst<LogQuadArc> > siter(fst); !siter.Done(); siter.Next()) {
    const LogQuadArc::StateId &stateId = siter.Value();
    if(fst.Final(stateId) != LogQuadWeight::Zero()) {
      return (int) stateId;
    }
  }
  return -1;
}

// return the id of a final states in this fst. if no final state found, returns -1.
int FstUtils::FindFinalState(const fst::VectorFst<LogArc>& fst) {
  for(StateIterator< VectorFst<LogArc> > siter(fst); !siter.Done(); siter.Next()) {
    const LogArc::StateId &stateId = siter.Value();
    if(fst.Final(stateId) != LogWeight::Zero()) {
      return (int) stateId;
    }
  }
  return -1;
}

// make sure that these two fsts have the same structure, including state ids, input/output labels, and nextstates, but not the weights.
bool FstUtils::AreShadowFsts(const fst::VectorFst<LogQuadArc>& fst1, const fst::VectorFst<LogArc>& fst2) {
  // verify number of states
  if(fst1.NumStates() != fst2.NumStates()) {
    cerr << "different state count" << endl;
    return false;
  }

  StateIterator< VectorFst<LogQuadArc> > siter1(fst1);
  StateIterator< VectorFst<LogArc> > siter2(fst2);
  while(!siter1.Done() || !siter2.Done()) {
    // verify state ids
    int from1 = siter1.Value(), from2 = siter2.Value();
    if(from1 != from2) {
      cerr << "different state ids" << endl;
      return false;
    }
    
    ArcIterator< VectorFst<LogQuadArc> > aiter1(fst1, from1);
    ArcIterator< VectorFst<LogArc> > aiter2(fst2, from2);
    while(!aiter1.Done() || !aiter2.Done()) {
      // verify number of arcs leaving this state
      if(aiter1.Done() || aiter2.Done()) {
	cerr << "different number of arcs leaving " << from1 << endl;
	return false;
      }

      // verify the arc input/output labels
      if(aiter1.Value().ilabel != aiter2.Value().ilabel) {
	cerr << "different input label" << endl;
	return false;
      } else if(aiter1.Value().olabel != aiter2.Value().olabel) {
	cerr << "different output label" << endl;
	return false;
      }
     
      // verify the arc next state
      if(aiter1.Value().nextstate != aiter2.Value().nextstate) {
	cerr << "different next states" << endl;
	return false;
      }
      
      // advance the iterators
      aiter1.Next();
      aiter2.Next();
    }

    // advance hte iterators
    siter1.Next();
    siter2.Next();
  }

  return true;
}

// assumption: 
// - acyclic fst
// - the last element in a LogQuadWeight represent the logprob of using this arc, not conditioned on anything.
void FstUtils::SampleFst(const fst::VectorFst<LogQuadArc>& fst, fst::VectorFst<LogQuadArc>& sampledFst, int sampleSize) {
  assert(sampledFst.NumStates() == 0 && fst.NumStates() > 0);
  
  int dumbSamplingClocks = 0;

  // for debugging only
  //cerr << "sampling" << endl;
  
  // create start and final states of sampledFst
  int sampledFstStartState = sampledFst.AddState();
  sampledFst.SetStart(sampledFstStartState);
  int sampledFstFinalState = sampledFst.AddState();

  // create a LogArc shadow fst of 'fst' which can be later used to compute potentials (LogQuadArc is too complex to compute potentials with)
  clock_t timestamp = clock();
  fst::VectorFst<fst::LogArc> probFst;
  fst::ArcMap(fst, &probFst, LogQuadToLogMapper());
  assert(AreShadowFsts(fst, probFst));
  cerr << "ArcMap took " << 1.0 * (clock() - timestamp) / CLOCKS_PER_SEC << " sec." << endl;

  // compute the potential of each state in fst towards the final state (i.e. unnormalized p(state->final))
  timestamp = clock();
  std::vector<fst::LogWeight> betas;
  fst::ShortestDistance(probFst, &betas, true);
  cerr << "ShortestDistance took " << 1.0  * (clock() - timestamp) / CLOCKS_PER_SEC << " sec." << endl;

  // set the stopping weight of the sampledFst's final state to the inverse of beta[0] = \sum_{path \in fst} weight(path), 
  // effectively making each complete path in the sampledFst has a proper probability according to the path distribution defined
  // as weights on the fst
  sampledFst.SetFinal(sampledFstFinalState, EncodeQuad(0.0, 0.0, 0.0, 0.0 - betas[probFst.Start()].Value()));

  // storage for the alias samplers
  std::map<int,AliasSampler> stateToSampler;

  // now we have all the necessary ingredients to start sampling. lets go!
  int samplesCounter = 0; 

  // Note: seed with time(0) if you don't care about reproducibility
  srand(1234);
  assert(sampleSize > 0);
  timestamp = clock();
  while(samplesCounter++ < sampleSize) {
    clock_t currentFstState = fst.Start();
    clock_t currentSampledFstState = sampledFstStartState;
    
    // for debugging only
    //cerr << endl << "sample #" << samplesCounter << endl;

    // keep adding arcs until we hit a final fst state
    while(fst.Final(currentFstState) == LogQuadWeight::Zero()) {

      // for debugging only
      //cerr << "currentFstState = " << currentFstState << endl;

      // we want to sample the index of the sampled arc
      unsigned chosenArcIndex = -1;

      clock_t timestamp = clock();
      // if this is the first time to visit currentFstState, create an alias sampler for it
      if(stateToSampler.count(currentFstState) == 0) {
	// enumerate the arcs leaving the current fst state, and calculate their respective scores
	// which define the likelihood of taking that arc now. The arc score is 
	// Times(arc's unconditional prob, toState's beta potential)
      	std::vector<double> arcScores;
	double totalScores = 0;
	for(ArcIterator< VectorFst<LogQuadArc> > aiter(fst, currentFstState); !aiter.Done(); aiter.Next()) {
	  float dummy, arcProb;
	  DecodeQuad(aiter.Value().weight, dummy, dummy, dummy, arcProb);
	  double score = nExp(Times(arcProb, betas[aiter.Value().nextstate]).Value());
	  arcScores.push_back(score);
	  totalScores += score;
	}
	
	stateToSampler[currentFstState].Init(arcScores);
      }

      // now we can sample the arc index
      chosenArcIndex = stateToSampler[currentFstState].Draw();
      assert(chosenArcIndex >= 0);
      
      // choose the arc based on the generated arc index
      LogQuadArc chosenArc;
      timestamp = clock();
      ArcIterator< VectorFst<LogQuadArc> > aiter(fst, currentFstState);
      while(chosenArcIndex > 0) {
	aiter.Next();
	chosenArcIndex--;
      }
      chosenArc = aiter.Value();

      // for debugging only
      //cerr << "chosen arc->" << chosenArc.nextstate << " with stopping weight " << PrintQuad(fst.Final(chosenArc.nextstate)) << " " << chosenArc.ilabel << ":" << chosenArc.olabel << " / " << PrintQuad(chosenArc.weight) << endl;

      // move currentFstState
      currentFstState = chosenArc.nextstate;
      
      // are we done with this sample?
      if(fst.Final(chosenArc.nextstate) == LogQuadWeight::Zero()) {
	// not yet
	chosenArc.nextstate = sampledFst.AddState();
      } else {
	// done, yay!
	chosenArc.nextstate = sampledFstFinalState;
      }
      
      // add the chosen arc to sampledFst
      sampledFst.AddArc(currentSampledFstState, chosenArc);

      // move currentSampledFstState
      currentSampledFstState = chosenArc.nextstate;      
    }
  }
  cerr << "sampling loop took " << (float) (clock() - timestamp) / CLOCKS_PER_SEC << " sec." << endl;
}

// returns a moses-style alignment string compatible with the alignment represented in the transducer bestAlignment
// assumption:
// - bestAlignment is a linear chain transducer. 
// - the input labels are tgt positions
// - the output labels are the corresponding src positions according to the alignment
string FstUtils::PrintAlignment(const VectorFst< StdArc > &bestAlignment) {
  stringstream output;
  
  cerr << "best alignment FST summary: " << endl;
  cerr << PrintFstSummary<StdArc>(bestAlignment) << endl;

  // traverse the transducer beginning with the start state
  int startState = bestAlignment.Start();
  int currentState = startState;
  int tgtPos = 0;
  while(bestAlignment.Final(currentState) == LogWeight::Zero()) {
    
    // get hold of the arc
    ArcIterator< VectorFst< StdArc > > aiter(bestAlignment, currentState);

    // identify the next state
    int nextState = aiter.Value().nextstate;

    // skip epsilon arcs
    if(aiter.Value().ilabel == EPSILON && aiter.Value().olabel == EPSILON) {
      currentState = nextState;
      continue;
    }

    // check the tgt position (shouldn't be a surprise)
    tgtPos++;
    if(aiter.Value().ilabel != tgtPos) {
      cerr << "aiter.Value().ilabel = " << aiter.Value().ilabel << ", whereas tgtPos = " << tgtPos << endl;
    }
    assert(aiter.Value().ilabel == tgtPos);

    // check the src position (should be >= 0)
    assert(aiter.Value().olabel >= 0);

    // print the alignment giza-style
    int srcPos = aiter.Value().olabel;
    // giza++ does not write null alignments
    if(srcPos != 0) {
      // giza++ uses zero-based src and tgt positions, and writes the src position first
      output << (srcPos - 1) << "-" << (tgtPos - 1) << " ";
    }

    // this state shouldn't have other arcs!
    aiter.Next();
    assert(aiter.Done());

    // move forward to the next state
    currentState = nextState;
  }

  return output.str();
}

// assumptions:
// - this fst is linear. no state has more than one outgoing/incoming arc
void FstUtils::LinearFstToVector(const fst::VectorFst<StdArc> &fst, std::vector<int> &ilabels, std::vector<int> &olabels, bool keepEpsilons) {
  assert(olabels.size() == 0);
  assert(ilabels.size() == 0);
  int currentState = fst.Start();
  do {
    ArcIterator< VectorFst<StdArc> > aiter(fst, currentState);
    if(keepEpsilons || aiter.Value().ilabel != EPSILON) {
      ilabels.push_back(aiter.Value().ilabel);
    }
    if(keepEpsilons || aiter.Value().olabel != EPSILON) {
      olabels.push_back(aiter.Value().olabel);
    }
    currentState = aiter.Value().nextstate;
    // assert it's a linear FST
    aiter.Next();
    assert(aiter.Done()); 
  } while(fst.Final(currentState) == fst::TropicalWeight::Zero());
}
