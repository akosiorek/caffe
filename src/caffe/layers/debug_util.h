//
// Created by Adam Kosiorek on 8/30/15.
//

#ifndef CAFFE_DEBUG_UTIL_H
#define CAFFE_DEBUG_UTIL_H

#include <sstream>

namespace caffe {

template<typename Dtype>
std::string vec2str(const Dtype *v, int n = 9) {
  std::stringstream ss;
  for (int i = 0; i < n; ++i) {
    ss << v[i] << " ";
  }
  return ss.str();
}

}  // namespace caffe

#endif //CAFFE_DEBUG_UTIL_H
