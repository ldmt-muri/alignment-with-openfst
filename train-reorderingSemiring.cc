#include "ReorderingWeight.h"

using namespace fst;
using namespace std;

//ReorderingArc::type = "ReorderingArc";

void Experimental() {
  ReorderingWeight x;

  x = ReorderingWeight(3.0, 3);
  x = Times(x, x);

  VectorFst< ReorderingArc > fst1;
  int state0 = fst1.AddState();
  int state1 = fst1.AddState();
  int state2 = fst1.AddState();
  int state3 = fst1.AddState();
  fst1.SetStart(state0);
  fst1.SetFinal(state3, x.One());
  fst1.AddArc(state0, ReorderingArc(11, 11, ReorderingWeight(1.0, 1), state1));
  fst1.AddArc(state1, ReorderingArc(22, 22, ReorderingWeight(2.0, 2), state2));
  fst1.AddArc(state2, ReorderingArc(22, 22, ReorderingWeight(3.0, 3), state3));
  
  VectorFst< ReorderingArc > fst2;
  state0 = fst2.AddState();
  state1 = fst2.AddState();
  state2 = fst2.AddState();
  fst2.SetStart(state0);
  fst2.SetFinal(state1, x.One());
  fst2.SetFinal(state2, x.One());
  fst2.AddArc(state0, ReorderingArc(22, 22, ReorderingWeight(3.0, 3), state1));
  fst2.AddArc(state0, ReorderingArc(11, 222, ReorderingWeight(0,0), state2));
  fst2.AddArc(state0, ReorderingArc(22, 111, ReorderingWeight(0,0), state1));
  fst2.AddArc(state0, ReorderingArc(22, 222, ReorderingWeight(0.0, 0), state2));
  fst2.AddArc(state1, ReorderingArc(11, 111, ReorderingWeight(0.0, 0), state1));
  fst2.AddArc(state1, ReorderingArc(11, 222, ReorderingWeight(0.0, 0), state2));
  fst2.AddArc(state1, ReorderingArc(22, 111, ReorderingWeight(0.0, 0), state1));
  fst2.AddArc(state1, ReorderingArc(22, 222, ReorderingWeight(0.0, 0), state2));
  fst2.AddArc(state2, ReorderingArc(11, 111, ReorderingWeight(0.0, 0), state1));
  fst2.AddArc(state2, ReorderingArc(11, 222, ReorderingWeight(0.0, 0), state2));
  fst2.AddArc(state2, ReorderingArc(22, 111, ReorderingWeight(0.0, 0), state1));
  fst2.AddArc(state2, ReorderingArc(22, 222, ReorderingWeight(0.0, 0), state2));

  VectorFst< ReorderingArc > fst3;
  state0 = fst3.AddState();
  fst3.SetStart(state0);
  fst3.SetFinal(state0, x.One());
  fst3.AddArc(state0, ReorderingArc(111, 111, ReorderingWeight(0,1), state0));
  fst3.AddArc(state0, ReorderingArc(222, 222, ReorderingWeight(0,2), state0));
  fst3.AddArc(state0, ReorderingArc(111, 111, ReorderingWeight(0,3), state0));
  
  VectorFst< ReorderingArc > temp, final;
  Compose(fst1, fst2, &temp);
  Compose(temp, fst3, &final);

}

int main(int argc, char **argv) {
    Experimental();
    return 0;
}
