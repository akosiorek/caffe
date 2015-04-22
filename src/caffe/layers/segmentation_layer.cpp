#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include "segmentation_energy.h"

namespace caffe {


#include <exception>
#include <fstream>
template<typename Dtype>
void printVec(const Dtype* v) {
  for(int i = 0; i < 9; ++i) {
    std::cout << v[i] << " ";
  }
  std::cout << std::endl;
}

template<typename Dtype>
void SegmentationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {

  for (int i = 0; i < bottom.size(); ++i) {
    CHECK_EQ(bottom[i]->channels(), 1)<< "Segmentation layer requiers a single channel input";
  }

  const SegmentationParameter param = this->layer_param_.segmentation_param();

  energy = std::unique_ptr<EnergyType>(new EnergyType(param));
  this->Reshape(bottom, top);
}

template<typename Dtype>
void SegmentationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  N_ = height_ * width_;

  energy->reshape(width_, height_);
  top[0]->Reshape(num_, 1, height_, width_);
  for(auto& buffer : bufferBackwardProp_) {
    buffer.Reshape(1, 1, height_, width_);
  }
}


template<typename Dtype>
void SegmentationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {

  Dtype* indicator = top[0]->mutable_cpu_data();
  this->energy->setData(bottom[0], bottom[1], bottom[2]);

  //initialize segmentation
  caffe_rng_uniform<Dtype>(N_, 0.1, 0.9, indicator);
//  caffe_set<Dtype>(N_, 0.5, indicator);

  printVec(indicator);
  energy->minimize_cpu(indicator);
  printVec(indicator);
}

template<typename Dtype>
void SegmentationLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  Dtype* bu = bufferBackwardProp_[0].mutable_cpu_data();
  Dtype* au = bufferBackwardProp_[0].mutable_cpu_diff();
  Dtype* term1 = bufferBackwardProp_[1].mutable_cpu_data();
  Dtype* term2 = bufferBackwardProp_[1].mutable_cpu_diff();

  Dtype* indicator = top[0]->mutable_cpu_data();
  Dtype* grad = top[0]->mutable_cpu_diff();

  Dtype* unitGrad = bottom[0]->mutable_cpu_diff();
  Dtype* horizontalGrad = bottom[1]->mutable_cpu_diff();
  Dtype* verticalGrad = bottom[2]->mutable_cpu_diff();

  const Dtype* horizontal = bottom[1]->cpu_data();
  const Dtype* vertical = bottom[2]->cpu_data();

//uniatary
    energy->invHessianVector_cpu(indicator, grad, unitGrad);
    printVec(unitGrad);

//horizontal
//TODO possible errors due to 0 padding; charbonnier introduces non-zero elements in padding
    energy->timesHorizontalB_cpu(indicator, bu);
    caffe_mul<Dtype>(N_, bu, horizontal, au);

    energy->charbonnierD1_cpu(au, term1);
    energy->charbonnierD2_cpu(au, term2);
    energy->zeroLastColumn_cpu(term2);
    caffe_mul<Dtype>(N_, bu, term2, term2);
    caffe_mul<Dtype>(N_, horizontal, term2, term2);
    caffe_axpy<Dtype>(N_, 1, term1, term2);

// Bh * (H^-1 * lossGrad)
    energy->timesHorizontalB_cpu(unitGrad, horizontalGrad);
    caffe_mul<Dtype>(N_, term2, horizontalGrad, horizontalGrad);

//vertical
//TODO possible errors due to 0 padding; charbonnier introduces non-zero elements in padding
    energy->timesVerticalB_cpu(indicator, bu);
    caffe_mul<Dtype>(N_, bu, vertical, au);

    energy->charbonnierD1_cpu(au, term1);
    energy->charbonnierD2_cpu(au, term2);
    energy->zeroLastRow_cpu(term2);
    caffe_mul<Dtype>(N_, bu, term2, term2);
    caffe_mul<Dtype>(N_, vertical, term2, term2);
    caffe_axpy<Dtype>(N_, 1, term1, term2);

// Bv * (H^-1 * lossGrad)
    energy->timesVerticalB_cpu(unitGrad, verticalGrad);
    caffe_mul<Dtype>(N_, term2, verticalGrad, verticalGrad);
}


INSTANTIATE_CLASS(SegmentationLayer);
REGISTER_LAYER_CLASS(Segmentation);

}  // namespace caffe
