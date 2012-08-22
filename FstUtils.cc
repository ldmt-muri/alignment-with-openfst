#include "FstUtils.h"

using namespace fst;

void FstUtils::PrintFstSummary(VectorFst<LogArc>& fst) {
  cout << "states:" << endl;
  for(StateIterator< VectorFst<LogArc> > siter(fst); !siter.Done(); siter.Next()) {
    const LogArc::StateId &stateId = siter.Value();
    string final = fst.Final(stateId) == 0? " FINAL": "";
    string initial = fst.Start() == stateId? " START" : "";
    cout << "state:" << stateId << initial << final << endl;
    cout << "arcs:" << endl;
    for(ArcIterator< VectorFst<LogArc> > aiter(fst, stateId); !aiter.Done(); aiter.Next()) {
      const LogArc &arc = aiter.Value();
      cout << arc.ilabel << ":" << arc.olabel << " " <<  stateId << "-->" << arc.nextstate << " " << arc.weight << endl;
    } 
    cout << endl;
  }
}
