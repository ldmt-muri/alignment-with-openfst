#ifndef _LBFGS_UTILS_H_
#define _LBFGS_UTILS_H_

#include <string>
#include <vector>
#include <sstream>
#include <set>

#include "liblbfgs/include/lbfgs.h"

typedef std::string string;
typedef std::stringstream stringstream;

class LbfgsUtils {
 public:
  // returns a string indicating the LBFGS status code
  static string LbfgsStatusIntToString(int status) {
    switch(status) {
    case LBFGS_SUCCESS:
      return "LBFGS_SUCCESS";
      break;
    case LBFGS_ALREADY_MINIMIZED:
      return "LBFGS_ALREADY_MINIMIZED";
      break;
    case LBFGSERR_UNKNOWNERROR:
      return "LBFGSERR_UNKNOWNERROR";
      break;
    case LBFGSERR_LOGICERROR:
      return "LBFGSERR_LOGICERROR";
      break;
    case LBFGSERR_OUTOFMEMORY:
      return "LBFGSERR_OUTOFMEMORY";
      break;
    case LBFGSERR_CANCELED:
      return "LBFGSERR_CANCELED";
      break;
    case LBFGSERR_INVALID_N:
      return "LBFGSERR_INVALID_N";
      break;
    case LBFGSERR_INVALID_N_SSE:
      return "LBFGSERR_INVALID_N_SSE";
      break;
    case LBFGSERR_INVALID_X_SSE:
      return "LBFGSERR_INVALID_X_SSE";
      break;
    case LBFGSERR_INVALID_EPSILON:
      return "LBFGSERR_INVALID_EPSILON";
      break;
    case LBFGSERR_INVALID_TESTPERIOD:
      return "LBFGSERR_INVALID_TESTPERIOD";
      break;
    case LBFGSERR_INVALID_DELTA:
      return "LBFGSERR_INVALID_DELTA";
      break;
    case LBFGSERR_INVALID_LINESEARCH:
      return "LBFGSERR_INVALID_LINESEARCH";
      break;
    case LBFGSERR_INVALID_MINSTEP:
      return "LBFGSERR_INVALID_MINSTEP";
      break;
    case LBFGSERR_INVALID_MAXSTEP:
      return "LBFGSERR_INVALID_MAXSTEP";
      break;
    case LBFGSERR_INVALID_FTOL:
      return "LBFGSERR_INVALID_FTOL";
      break;
    case LBFGSERR_INVALID_WOLFE:
      return "LBFGSERR_INVALID_WOLFE";
      break;
    case LBFGSERR_INVALID_GTOL:
      return "LBFGSERR_INVALID_GTOL";
      break;
    case LBFGSERR_INVALID_XTOL:
      return "LBFGSERR_INVALID_XTOL";
      break;
    case LBFGSERR_INVALID_MAXLINESEARCH:
      return "LBFGSERR_INVALID_MAXLINESEARCH";
      break;
    case LBFGSERR_INVALID_ORTHANTWISE:
      return "LBFGSERR_INVALID_ORTHANTWISE";
      break;
    case LBFGSERR_INVALID_ORTHANTWISE_START:
      return "LBFGSERR_INVALID_ORTHANTWISE_START";
    break;
    case LBFGSERR_INVALID_ORTHANTWISE_END:
      return "LBFGSERR_INVALID_ORTHANTWISE_END";
      break;
    case LBFGSERR_OUTOFINTERVAL:
      return "LBFGSERR_OUTOFINTERVAL";
      break;
    case LBFGSERR_INCORRECT_TMINMAX:
      return "LBFGSERR_INCORRECT_TMINMAX";
      break;
    case LBFGSERR_ROUNDING_ERROR:
      return "LBFGSERR_ROUNDING_ERROR";
      break;
    case LBFGSERR_MINIMUMSTEP:
      return "LBFGSERR_MINIMUMSTEP";
      break;
    case LBFGSERR_MAXIMUMSTEP:
      return "LBFGSERR_MAXIMUMSTEP";
      break;
    case LBFGSERR_MAXIMUMLINESEARCH:
      return "LBFGSERR_MAXIMUMLINESEARCH";
      break;
    case LBFGSERR_MAXIMUMITERATION:
      return "LBFGSERR_MAXIMUMITERATION";
      break;
    case LBFGSERR_WIDTHTOOSMALL:
      return "LBFGSERR_WIDTHTOOSMALL";
      break;
    case LBFGSERR_INVALIDPARAMETERS:
      return "LBFGSERR_INVALIDPARAMETERS";
    break;
    case LBFGSERR_INCREASEGRADIENT:
      return "LBFGSERR_INCREASEGRADIENT";
      break;
    default:
      return "THIS IS NOT A VALID LBFGS STATUS CODE";
      break;
    }
  }

};

#endif
