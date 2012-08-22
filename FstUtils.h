#ifndef _FST_UTILS_H_
#define _FST_UTILS_H_

#include <fst/fstlib.h>

class FstUtils {
 public:
  inline static float nLog(float prob) {
    return -1.0 * log(prob);
  }

  static void PrintFstSummary(fst::VectorFst<fst::LogArc>& fst);

  static const int LOG_ZERO = 30;

};

#endif
