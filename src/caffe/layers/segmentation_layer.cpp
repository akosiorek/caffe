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
	this->stepSize_ = pool_param.step_size();
	this->dataWeight_ = pool_param.data_weight();
	this->logBarrierWeight_ = pool_param.log_barrier_weight();
	this->gradientTolerance_ = pool_param.gradient_norm_tolerance();
	this->smoothnesEps_ = pool_param.smoothnes_eps();
	this->minimizationIters_ = pool_param.minimization_iters();
	this->invHessIters_ = pool_param.inv_hess_iters();
	this->invHessTolerance_ = pool_param.inv_hess_tolerance();

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

  bufferEnergyGrad_.Reshape(1, 1, height_, width_);
  bufferResidualDirection_.Reshape(1, 1, height_, width_);
  bufferMatVecStorage_.Reshape(1, 1, height_, width_);
}

template <typename Dtype>
void SegmentationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	size_t offset = N_;
	top[0]->CopyFrom(*bottom[0]);
	Dtype* indicatorValue = top[0]->mutable_cpu_data();
	Dtype* indicatorGrad = top[0]->mutable_cpu_diff();

	unit_ = bottom[0]->cpu_data();
	horizontal_ = bottom[1]->cpu_data();
	vertical_ = bottom[2]->cpu_data();

	for(int i = 0; i < num_; ++i) {
		minimize_cpu(indicatorValue, indicatorGrad);
		indicatorValue += offset;
		indicatorGrad += offset;
		unit_ += offset;
		horizontal_ += offset;
		vertical_ += offset;
	}
	unit_ = nullptr;
	horizontal_ = nullptr;
	vertical_ = nullptr;
}

template <typename Dtype>
void SegmentationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	Dtype* indicatorValue = top[0]->mutable_cpu_data();
	Dtype* indicatorGrad = top[0]->mutable_cpu_diff();

	Dtype* unitGrad = bottom[0]->mutable_cpu_diff();
//	Dtype* horizontalGrad = bottom[1]->mutable_cpu_diff();
//	Dtype* verticalGrad = bottom[2]->mutable_cpu_diff();

	size_t offset = N_;
	for(int i = 0; i < num_; ++i) {
		//uniatary
		invHessianVector_cpu(indicatorValue, indicatorGrad, unitGrad);

		//horizontal

		//vertical


		indicatorValue += offset;
		indicatorGrad += offset;
		unit_ += offset;
		horizontal_ += offset;
		vertical_ += offset;
	}


	unit_ = nullptr;
	horizontal_ = nullptr;
	vertical_ = nullptr;
}

template<typename Dtype>
void SegmentationLayer<Dtype>::minimize_cpu(Dtype* indicatorValue,
		Dtype* indicatorGrad) {

	computeEnergyGradient_cpu(indicatorValue, indicatorGrad);
	Dtype gradientNorm = caffe_cpu_dot<Dtype>(N_, indicatorGrad, indicatorGrad);
	Dtype toleranceSquare = gradientTolerance_ * gradientTolerance_;
	int iter = 0;
	while(gradientNorm > toleranceSquare && iter++ < minimizationIters_) {
		caffe_axpy<Dtype>(N_, -stepSize_, indicatorGrad, indicatorValue);
		computeEnergyGradient_cpu(indicatorValue, indicatorGrad);
		gradientNorm = caffe_cpu_dot<Dtype>(N_, indicatorGrad, indicatorGrad);
	}
}

template<typename Dtype>
void SegmentationLayer<Dtype>::computeEnergyGradient_cpu(Dtype* indicatorValue,
		Dtype* indicatorGrad) {

	Dtype* du = bufferEnergyGrad_.mutable_cpu_data();
	Dtype* temp = bufferEnergyGrad_.mutable_cpu_diff();

	caffe_set<Dtype>(N_, 0, indicatorGrad);
	// dudx
	forwardDiffDx_cpu(indicatorValue, du);
	caffe_mul<Dtype>(N_, du, horizontal_, du);
	gradInterim_cpu(du, horizontal_, du);
	gradDiffDx_cpu(du, temp);
	caffe_axpy<Dtype>(N_, 1, temp, indicatorGrad);

	// dudy
	forwardDiffDy_cpu(indicatorValue, du);
	caffe_mul<Dtype>(N_, temp, vertical_, du);
	gradInterim_cpu(du, vertical_, du);
	gradDiffDy_cpu(du, temp);
	caffe_axpy<Dtype>(N_, 1, temp, indicatorGrad);

	//gradUnary
	caffe_axpy(N_, dataWeight_, unit_, indicatorGrad);


	//gradLog
	for(int i = 0; i < N_; ++i) {
		du[i] = 1/indicatorValue[i] - 1/(1 - indicatorValue[i]);
	}
	caffe_axpy<Dtype>(N_, -logBarrierWeight_, du, indicatorGrad);

}

template<typename Dtype>
void SegmentationLayer<Dtype>::forwardDiffDx_cpu(const Dtype* f, Dtype* dx) {
	for(int i = 0; i < height_; ++i) {
		caffe_copy<Dtype>(width_ - 1, f + 1, dx);
		caffe_axpy<Dtype>(width_ - 1, -1, f, dx);
		*(dx + width_ - 1) = 0;

		f += width_;
		dx += width_;
	}
}

template<typename Dtype>
void SegmentationLayer<Dtype>::forwardDiffDy_cpu(const Dtype* f, Dtype* dy) {
	caffe_copy<Dtype>(N_ - width_, f + width_, dy);
	caffe_axpy<Dtype>(N_ - width_, -1, f, dy);
	caffe_set<Dtype>(width_, 0, dy + N_ - width_);
}

template<typename Dtype>
void SegmentationLayer<Dtype>::gradInterim_cpu(const Dtype* grad, const Dtype* potential, Dtype* interim) {

	for(int i = 0; i < N_; ++i) {
		interim[i] = -potential[i] * grad[i] / sqrt(grad[i] * grad[i] + smoothnesEps_ * smoothnesEps_);
	}
}

template<typename Dtype>
void SegmentationLayer<Dtype>::gradDiffDx_cpu(const Dtype* grad, Dtype* diffDx) {
	caffe_copy<Dtype>(N_, grad, diffDx);
	for(int i = 0; i < height_; ++i) {
		caffe_axpy<Dtype>(width_ - 2, -1, grad, diffDx + 1);
		*(diffDx + width_ -1) = 0;
	}
}

template<typename Dtype>
void SegmentationLayer<Dtype>::gradDiffDy_cpu(const Dtype* grad, Dtype* diffDy) {
	caffe_copy<Dtype>(N_ - width_, grad, diffDy);
	caffe_axpy<Dtype>(N_ - 2 * width_, -1, grad, diffDy + width_);
	caffe_set<Dtype>(width_, 0, diffDy + N_ - width_);
}

template<typename Dtype>
void SegmentationLayer<Dtype>::hessianVector_cpu(const Dtype* indicator, const Dtype* vec, Dtype* Hv) {

	Dtype eps = smoothnesEps_;
	Dtype* point = bufferHessVec_.mutable_cpu_data();
	Dtype* grad = bufferHessVec_.mutable_cpu_diff();

	caffe_set<Dtype>(N_, 0, Hv);
	//Forward Gradient
	caffe_copy<Dtype>(N_, indicator, point);
	caffe_axpy<Dtype>(N_, 1, vec, point);
	computeEnergyGradient_cpu(point, grad);
	//Save the result
	caffe_axpy<Dtype>(N_, 0.5 / eps, grad, Hv);;

	//Backward Gradient
	caffe_copy<Dtype>(N_, indicator, point);
	caffe_axpy<Dtype>(N_, -1, vec, point);
	computeEnergyGradient_cpu(point, grad);

	//Subtract
	caffe_axpy<Dtype>(N_, -0.5 / eps, grad, Hv);
}

/**
 * Conjugate Gradient
 */
template<typename Dtype>
void SegmentationLayer<Dtype>::invHessianVector_cpu(const Dtype* indicator, const Dtype* vec, Dtype* iHv) {

	Dtype* residual = bufferResidualDirection_.mutable_cpu_data();
	Dtype* direction = bufferResidualDirection_.mutable_cpu_diff();

	Dtype* matVecProd = bufferMatVecStorage_.mutable_cpu_data();

	Dtype residualNormNew = 0;
	Dtype residualNormOld = 0;

	//initialize iHv
	caffe_set<Dtype>(N_, 1, iHv);

	// residual = b - Ax
	caffe_copy<Dtype>(N_, indicator, residual);
	hessianVector_cpu(indicator, indicator, matVecProd);
	caffe_axpy<Dtype>(N_, -1, matVecProd, residual);

	//direction
	caffe_copy<Dtype>(N_, residual, direction);

	int iter = 0;
	while(iter++ < invHessIters_) {
		residualNormNew = caffe_cpu_dot<Dtype>(N_, residual, residual);
		hessianVector_cpu(indicator, direction, matVecProd);
		Dtype alpha = residualNormNew / caffe_cpu_dot<Dtype>(N_, direction, matVecProd);

		//update iHv
		caffe_axpy<Dtype>(N_, alpha, direction, iHv);

		residualNormOld = residualNormNew;
		// new residual
		caffe_axpy<Dtype>(N_, -alpha, matVecProd, residual);
		residualNormNew = caffe_cpu_dot<Dtype>(N_, residual, residual);
		if(sqrt(residualNormNew) < invHessTolerance_) {
			break;
		}

		// new direction
		double beta = residualNormNew / residualNormOld;
		caffe_axpy<Dtype>(N_, -(1 + beta), direction, direction);
		caffe_axpy<Dtype>(N_, 1, residual, direction);
	}
}

INSTANTIATE_CLASS(SegmentationLayer);
REGISTER_LAYER_CLASS(Segmentation);

}  // namespace caffe
