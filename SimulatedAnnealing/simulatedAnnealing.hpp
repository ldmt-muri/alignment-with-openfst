#ifndef SIMULATED_ANNEALING
#define SIMULATED_ANNEALING 1

#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <boost/foreach.hpp>

#include "optimize.hpp"

namespace Optimization{

  template<typename CostFunction>
  class SimulatedAnnealing{

  public: 

    //
    //  SimulatedAnnealing
    //
    //   @param Domain to optimize
    //   @param step (the step for moving an index)

    SimulatedAnnealing(Domain & domain, 
		       Value temperature=100000,
		       Value cool = 0.98,
		       unsigned int step=1) : 
      domain_(domain), size_domain_(domain.size()),
      temperature_(temperature), cool_(cool), step_(step){};
    
    Vector optimize(){

      unsigned int iteration, i;;
      Value step, cost_vec, cost_vecb, sa, r;      

      Vector vec(size_domain_);    

      if (size_domain_ == 0){
	std::cout << "Size domain is 0, exiting" << std::endl;
	return vec;
      }

      srand( time(NULL) );        // init the random generator

      //
      // Random init the vector within the domain
      //
      i=0;
      BOOST_FOREACH(Vector::value_type& v, vec){
	
	v = domain_[i].first +
	  (fabs(domain_[i].first - domain_[i].second) * 
	   (rand() / (RAND_MAX + 1.0)));
	i++;
      }
      
      std::cout << "Initial vector" << vec << std::endl;

      // 
      // // temperature is good enough
      //
      iteration = 0;
      while (temperature_ > 0.1){  
	
	i = rand() % size_domain_;// choose one index to change
	step =                    // chose the step
	  (2 * step_ * rand() / (RAND_MAX + 1.0)) - step_;     
	
	std::cout << "Iteration=" << iteration 
		  << " Temperature=" << temperature_
		  << " changing index i=" << i
		  << " step=" << step << std::endl;	

	Vector vecb(vec);             // copy vector
	vecb[i] = (1+step) * vecb[i];              // with the local change

	if (vecb[i]<domain_[i].first) // within the bounds of var
	  vecb[i]=domain_[i].first;

	if (vecb[i]>domain_[i].second)// within the bounds of var
	  vecb[i]=domain_[i].second;

	std::cout << " vector=" << vec << std::endl;
	std::cout << " vectorb=" << vecb << std::endl;

	cost_vec = CostFunction()(vec);  // original cost
	cost_vecb = CostFunction()(vecb);// cost of permuted vector
	  
	sa = exp((-cost_vec-cost_vecb)/temperature_);

	if (cost_vecb<cost_vec){ 
	  std::cout << "\tcost reduction, changing vector"<<std::endl;
	  vec=vecb;
	} // simulated annealing
	else if ((r = (rand()/(RAND_MAX + 1.0))) < sa){
	  std::cout << "\tsimulated annealing (r=" << r
		    << "<sa=" << sa << "), changing vector"
		    <<std::endl;
	  vec=vecb;
	}

	
	temperature_ *= cool_;          // reduce the temperature
	iteration++;
      }// while
      
      return vec;
    };

  private:
    Domain domain_;                 // range for variables
    std::size_t size_domain_;       // number of variable
    Value temperature_;             // temperature of the system
    Value cool_;                    // coolness of the system
    unsigned int step_;
  };

};

#endif
