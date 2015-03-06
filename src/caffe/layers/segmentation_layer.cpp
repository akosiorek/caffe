#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SegmentationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	for(int i = 0; i < bottom.size(); ++i) {
		CHECK_EQ(bottom[i]->channels(), 1) << "Segmentation layer requiers a single channel input";
	}

	SegmentationParameter pool_param = this->layer_param_.segmentation_param();
	this->learningRate_ = pool_param.learning_rate();
	this->gradientTolerance_ = pool_param.gradient_norm_tolerance();

	this->Reshape(bottom, top);
}

template <typename Dtype>
void SegmentationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  N_ = height_ * width_;

  top[0]->Reshape(num_, 1, height_, width_);
  indicatorGradient_.Reshape(1, 1, height_, width_);
}

template <typename Dtype>
void SegmentationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	size_t offset = N_;
	top[0]->CopyFrom(*bottom[0]);
	Dtype* indicatorValue = top[0]->mutable_cpu_data();
	Dtype* indicatorGrad = top[0]->mutable_cpu_diff();
	const Dtype* unitPotential = bottom[0]->cpu_data();
	const Dtype* horizontalPotential = bottom[1]->cpu_data();
	const Dtype* verticalPotential = bottom[2]->cpu_data();

	for(int i = 0; i < num_; ++i) {
		minimize_cpu(indicatorValue, indicatorGrad, unitPotential, horizontalPotential, verticalPotential);
		indicatorValue += offset;
		indicatorGrad += offset;
		unitPotential += offset;
		horizontalPotential += offset;
		verticalPotential += offset;
	}
}

template <typename Dtype>
void SegmentationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

template<typename Dtype>
void SegmentationLayer<Dtype>::minimize_cpu(Dtype* indicatorValue,
		Dtype* indicatorGrad, const Dtype* unit, const Dtype* horizontal,
		const Dtype* vertical) {

	computeEnergyGradient_cpu(indicatorValue, indicatorGrad, unit, horizontal, vertical);
	Dtype gradientNorm = caffe_cpu_dot<Dtype>(N_, indicatorGrad, indicatorGrad);
	Dtype toleranceSquare = gradientTolerance_ * gradientTolerance_;
	while(gradientNorm > toleranceSquare) {
		caffe_axpy<Dtype>(N_, -learningRate_, indicatorGrad, indicatorValue);
		computeEnergyGradient_cpu(indicatorValue, indicatorGrad, unit, horizontal, vertical);
		gradientNorm = caffe_cpu_dot<Dtype>(N_, indicatorGrad, indicatorGrad);
	}
}

template<typename Dtype>
void SegmentationLayer<Dtype>::computeEnergyGradient_cpu(Dtype* indicatorValue,
		Dtype* indicatorGrad, const Dtype* unit, const Dtype* horizontal,
		const Dtype* vertical) {

	Dtype* ux = indicatorGradient_.mutable_cpu_data();
	Dtype* uy = indicatorGradient_.mutable_cpu_diff();
}

INSTANTIATE_CLASS(SegmentationLayer);
REGISTER_LAYER_CLASS(Segmentation);

}  // namespace caffe
