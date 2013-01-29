#ifndef OPTIMIZE 
#define OPTIMIZE 1
#include <vector>
#include <iostream>
#include <boost/foreach.hpp>

namespace Optimization{

  typedef double Value;
  typedef std::pair<Value, Value> Interval;
  typedef std::vector<Interval> Domain;
  typedef std::vector<Value> Vector;

  std::ostream& operator<<(std::ostream& os, const Domain& dom)
  {
    BOOST_FOREACH(Domain::value_type d, dom){
      os << "(" << d.first << "," << d.second << ") ";
    }
    return os;
  }

  std::ostream& operator<<(std::ostream& os, const Vector& vec)
  {
    os << "(";
    BOOST_FOREACH(Vector::value_type v, vec){
      os << v << " " ;
    }
    os << ")";
    return os;
  }

};

#endif
