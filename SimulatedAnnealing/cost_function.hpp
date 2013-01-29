#ifndef OPTIMIZE_COST_FUNCTION 
#define OPTIMIZE_COST_FUNCTION 1

#include "optimize.hpp"

//
// here define your own cost function on the vecto
//

namespace Optimization{

  //
  // The cost function for the Optimization Problem

  struct CostFun : public std::unary_function<Vector, Value>{
  
    Value operator() (Vector vec) const
    {    
      Value ret = 0;

      BOOST_FOREACH(Vector::value_type v, vec){
	ret += abs(v);                         // |v|_1 just an example
      }
      return ret;
    };    
  };


};
#endif
