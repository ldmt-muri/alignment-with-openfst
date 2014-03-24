#ifndef _LATENT_CRF_PARSER_H_
#define _LATENT_CRF_PARSER_H_

#include <iostream>
#include <vector>
#include <utility>
#include <iterator>
#include <cstdlib>
#include <limits>
#include <cerrno>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graph_concepts.hpp>
#include <boost/concept_check.hpp>
#include <boost/operators.hpp>
#include <boost/iterator.hpp>
#include <boost/multi_array.hpp>
#include <boost/random.hpp>

#include <Eigen/Dense>

#include "../edmonds-alg-1.1.2/src/edmonds_optimum_branching.hpp"
#include "../core/LatentCrfModel.h"

using namespace std;
using namespace boost;

// definitions of a complete graph that implements the EdgeListGraph
// concept of Boost's graph library.
namespace boost {
  struct complete_graph {
  complete_graph(int n_vertices) : n_vertices(n_vertices) {}
    int n_vertices;
    
    struct edge_iterator : public input_iterator_helper<edge_iterator, int, std::ptrdiff_t, int const *, int>
    {
      int edge_idx, n_vertices;
      
    edge_iterator() : edge_idx(0), n_vertices(-1) {}
    edge_iterator(int n_vertices, int edge_idx) : edge_idx(edge_idx), n_vertices(n_vertices) {}
    edge_iterator &operator++() {
      if (edge_idx >= n_vertices * n_vertices)
        return *this;
      ++edge_idx;
      if (edge_idx / n_vertices == edge_idx % n_vertices)
        ++edge_idx;
      return *this;
    }
    int operator*() const {return edge_idx;}
    bool operator==(const edge_iterator &iter) const
    {
      return edge_idx == iter.edge_idx;
    }
    };
  };
  
  template<>
    struct graph_traits<complete_graph> {
    typedef int                             vertex_descriptor;
    typedef int                             edge_descriptor;
    typedef directed_tag                    directed_category;
    typedef disallow_parallel_edge_tag      edge_parallel_category;
    typedef edge_list_graph_tag             traversal_category;
    typedef complete_graph::edge_iterator   edge_iterator;
    typedef unsigned                        edges_size_type;
    
    static vertex_descriptor null_vertex() {return -1;}
  };

  pair<complete_graph::edge_iterator, complete_graph::edge_iterator> edges(const complete_graph &g);
  
  unsigned num_edges(const complete_graph &g);
  
  int source(int edge, const complete_graph &g); 
  
  int target(int edge, const complete_graph &g);
}

typedef graph_traits<complete_graph>::edge_descriptor Edge;
typedef graph_traits<complete_graph>::vertex_descriptor Vertex;

class LatentCrfParser : public LatentCrfModel {

 protected:
  LatentCrfParser(const std::string &textFilename, 
		  const std::string &outputPrefix, 
		  LearningInfo &learningInfo,
		  const std::string &initialLambdaParamsFilename, 
		  const std::string &initialThetaParamsFilename,
		  const std::string &wordPairFeaturesFilename);
  
  ~LatentCrfParser();

  std::vector<int64_t>& GetObservableSequence(int exampleId);

  std::vector<int64_t>& GetObservableContext(int exampleId);

  std::vector<int64_t>& GetReconstructedObservableSequence(int exampleId);

  void InitTheta();

  void PrepareExample(unsigned exampleId);

  int64_t GetContextOfTheta(unsigned sentId, int y);

  void BuildMatrices(const unsigned sentId,
                     Eigen::MatrixXd *adjacency,
                     Eigen::MatrixXd *laplacianHat,
                     bool conditionOnZ);
 public:

  static LatentCrfModel* GetInstance();

  static LatentCrfModel* GetInstance(const std::string &textFilename, 
                                     const std::string &outputPrefix, 
                                     LearningInfo &learningInfo, 
                                     const std::string &initialLambdaParamsFilename, 
                                     const std::string &initialThetaParamsFilename,
                                     const std::string &wordPairFeaturesFilename);
  
  void Label(std::vector<int64_t> &tokens, std::vector<int> &labels);

  void Label(std::vector<int64_t> &tokens, std::vector<int64_t> &context, std::vector<int> &labels);

  void Label(const string &labelsFilename);

  void SetTestExample(std::vector<int64_t> &sent);

  double UpdateThetaMleForSent(const unsigned sentId, 
                               MultinomialParams::ConditionalMultinomialParam<int64_t> &mle, 
                               boost::unordered_map<int64_t, double> &mleMarginals) override;

  double ComputeNllZGivenXAndLambdaGradient(vector<double> &derivativeWRTLambda, 
                                            int fromSentId, 
                                            int toSentId, 
                                            double *devSetNll) override;

  double GetMaxSpanningTree(Eigen::MatrixXd &adjacency, vector<int> &maxSpanningTree, int &root);
  
  vector<int> GetViterbiParse(int sentId, bool conditionOnZ);
  
 private:
  // vocabulary of src language
  //std::set<int64_t> x_sDomain;


  // data
  std::vector< std::vector<int64_t> > sents, contexts, testSents, testContexts;

 public:
  // the value of y_i that indicates the sentence HEAD 
  static int64_t ROOT_ID;
  // the string representing the null token at the source sentence which produce "spurious" tgt words.
  static string ROOT_STR;
  static int ROOT_POSITION;
};

#endif
