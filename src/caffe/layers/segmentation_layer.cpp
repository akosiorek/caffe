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

template<typename Dtype>
void SegmentationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {

  for (int i = 0; i < bottom.size(); ++i) {
    CHECK_EQ(bottom[i]->channels(), 1) << "Segmentation layer requiers a single channel input";
  }

  const SegmentationParameter param = this->layer_param_.segmentation_param();
  indicatorFiller_ = shared_ptr<Filler<Dtype>>(GetFiller<Dtype>(param.indicator_filler()));

  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    // Intialize the data-term weight
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, 1));
    this->blobs_[0]->mutable_cpu_data()[0] = param.init_data_weight();
  }

  energy = std::unique_ptr<EnergyType>(new EnergyType(param, this->blobs_[0]));

  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
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
  indicatorFiller_->Fill(top[0]);
//  energy->minimizeNAG_cpu(indicator);
  energy->minimizeNCOBF_cpu(indicator);
//  energy->minimizeNewton(indicator);
//  energy->minimizeDualGradientMethod_cpu(indicator);
}


template<typename Dtype>
void SegmentationLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  Dtype* bu = bufferBackwardProp_[0].mutable_cpu_data();
  Dtype* au = bufferBackwardProp_[0].mutable_cpu_diff();
  Dtype* term1 = bufferBackwardProp_[1].mutable_cpu_data();
  Dtype* term2 = bufferBackwardProp_[1].mutable_cpu_diff();

//  Forward_cpu(bottom, top);

  const Dtype* indicator = top[0]->cpu_data();
  const Dtype* grad = top[0]->cpu_diff();

  Dtype* unitGrad = bottom[0]->mutable_cpu_diff();
  Dtype* horizontalGrad = bottom[1]->mutable_cpu_diff();
  Dtype* verticalGrad = bottom[2]->mutable_cpu_diff();

  const Dtype* unit = bottom[0]->cpu_data();
  const Dtype* horizontal = bottom[1]->cpu_data();
  const Dtype* vertical = bottom[2]->cpu_data();

//uniatary (1), needed anyway for gradient w.r.t. data weight
  energy->invHessianVector_cpu(indicator, grad, unitGrad);
  caffe_scal<Dtype>(N_, -1, unitGrad);

  //TODO not checked in tests
  // data-weight update
  this->blobs_[0]->mutable_cpu_diff()[0] = caffe_cpu_dot<Dtype>(N_, unitGrad,
                                                                unit);

  //horizontal
  if (propagate_down[1]) {

    //TODO possible errors due to 0 padding; charbonnier introduces non-zero elements in padding
    energy->timesHorizontalB_cpu(indicator, bu);
    caffe_mul<Dtype>(N_, bu, horizontal, au);

    energy->charbonnierD1_cpu(au, term1);
    energy->charbonnierD2_cpu(au, term2);
    energy->zeroLastColumn_cpu(term2);
    caffe_mul<Dtype>(N_, bu, term2, term2);
    caffe_mul<Dtype>(N_, horizontal, term2, term2);
    caffe_axpy<Dtype>(N_, 1, term1, term2);
//
//     caffe_mul<Dtype>(N_, horizontal, term2, term2);

// Bh * (H^-1 * lossGrad)
    energy->timesHorizontalB_cpu(unitGrad, horizontalGrad);
    caffe_mul<Dtype>(N_, term2, horizontalGrad, horizontalGrad);
  }

  //vertical
  if (propagate_down[2]) {
    //TODO possible errors due to 0 padding; charbonnier introduces non-zero elements in padding
    energy->timesVerticalB_cpu(indicator, bu);
    caffe_mul<Dtype>(N_, bu, vertical, au);

    energy->charbonnierD1_cpu(au, term1);
    energy->charbonnierD2_cpu(au, term2);
    energy->zeroLastRow_cpu(term2);
    caffe_mul<Dtype>(N_, bu, term2, term2);
    caffe_mul<Dtype>(N_, vertical, term2, term2);
    caffe_axpy<Dtype>(N_, 1, term1, term2);
//
//      caffe_mul<Dtype>(N_, vertical, term2, term2);

// Bv * (H^-1 * lossGrad)
    energy->timesVerticalB_cpu(unitGrad, verticalGrad);
    caffe_mul<Dtype>(N_, term2, verticalGrad, verticalGrad);
  }

  // multiply by data weight from (1)
  if (propagate_down[0]) {
    caffe_scal<Dtype>(N_, this->blobs_[0]->cpu_data()[0], unitGrad);
    // else zero it
  } else {
    caffe_set<Dtype>(N_, 0, unitGrad);
  }

  LOG(ERROR) << "grad: " << vec2str(grad);
  LOG(ERROR) << "ind:  " << vec2str(indicator);
  LOG(ERROR) << "unit: " << vec2str(unit);
  LOG(ERROR) << "hori: " << vec2str(horizontal);
  LOG(ERROR) << "vert: " << vec2str(vertical);

  LOG(ERROR) << "unitGrad: " << vec2str(unitGrad);
  LOG(ERROR) << "horiGrad: " << vec2str(horizontalGrad);
  LOG(ERROR) << "vertGrad: " << vec2str(verticalGrad);
}


INSTANTIATE_CLASS(SegmentationLayer);
REGISTER_LAYER_CLASS(Segmentation);

}  // namespace caffe
