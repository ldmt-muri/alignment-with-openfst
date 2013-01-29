#include "optimize.hpp"
#include "simulatedAnnealing.hpp"
#include "cost_function.hpp"

#include <boost/foreach.hpp>

using namespace Optimization;

int main() {
  
  Domain domain;
  domain.push_back(std::make_pair(-10,20));
  domain.push_back(std::make_pair(-30,50));
  domain.push_back(std::make_pair(-70,80));
  domain.push_back(std::make_pair(-10,100));
  
  std::cout << "Domain" << domain << std::endl;

  SimulatedAnnealing<CostFun> sa(domain);
  Vector result = sa.optimize();

}
