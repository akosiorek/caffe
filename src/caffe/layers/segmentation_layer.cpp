#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

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

template<typename Dtype>
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
  bufferHessVec_.Reshape(1, 1, height_, width_);
  bufferHessian_.Reshape(5, 1, height_, width_);
}

#include <exception>
#include <fstream>
template<typename Dtype>
void SegmentationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {

  size_t offset = N_;
//  top[0]->CopyFrom(*bottom[0]);
  Dtype* indicatorValue = top[0]->mutable_cpu_data();
  Dtype* indicatorGrad = top[0]->mutable_cpu_diff();


  unit_ = bottom[0]->cpu_data();
  horizontal_ = bottom[1]->cpu_data();
  vertical_ = bottom[2]->cpu_data();

  //initialize segmentation
//  caffe_rng_uniform<Dtype>(N_, 0.1, 0.9, indicatorValue);
  caffe_set<Dtype>(N_, 0.5, indicatorValue);

  for (int i = 0; i < num_; ++i) {
    LOG(INFO) << "Forwarding " << i << "/" << num_;
    minimize_cpu(indicatorValue, indicatorGrad);


//    std::ofstream os("dump.txt");
//
//    LOG(INFO) << "Bottom:";
//    for(auto blob : bottom) {
//      for(int i = 0; i < blob->height(); ++i) {
//        for(int j = 0; j < blob->width(); ++j) {
//          os << blob->cpu_data()[i * blob->width() + j] << ",";
//        }
//        os << "\n";
//      }
//      os << "\n\n\n";
//    }
//    LOG(INFO) << "Top:";
//    for(auto blob : top) {
//      for(int i = 0; i < blob->height(); ++i) {
//         for(int j = 0; j < blob->width(); ++j) {
//           os << blob->cpu_data()[i * blob->width() + j] << ",";
//         }
//         os << "\n";
//       }
//       os << "\n\n\n";
//    }
//    os.close();
//    std::terminate();

//    LOG(INFO) << "indicator:";
//    printVec(indicatorValue);

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
void SegmentationLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  Dtype* bu = bufferEnergyGrad_.mutable_cpu_data();
  Dtype* au = bufferEnergyGrad_.mutable_cpu_diff();
  Dtype* term1 = bufferHessVec_.mutable_cpu_data();
  Dtype* term2 = bufferHessVec_.mutable_cpu_diff();

  Dtype* indicatorValue = top[0]->mutable_cpu_data();
  Dtype* indicatorGrad = top[0]->mutable_cpu_diff();

  Dtype* unitGrad = bottom[0]->mutable_cpu_diff();
  Dtype* horizontalGrad = bottom[1]->mutable_cpu_diff();
  Dtype* verticalGrad = bottom[2]->mutable_cpu_diff();

  unit_ = bottom[0]->cpu_data();
  horizontal_ = bottom[1]->cpu_data();
  vertical_ = bottom[2]->cpu_data();

  size_t offset = N_;
  for (int i = 0; i < num_; ++i) {
    //uniatary
    //H^-1 * lossGrad
//    LOG(INFO) << "indicator:";
//    printVec(indicatorValue);
//    LOG(INFO) << "Energy = " << this->energy_cpu(indicatorValue);
    invHessianVector_cpu(indicatorValue, indicatorGrad, unitGrad);
    printVec(unitGrad);

    //horizontal
    timesHorizontalB_cpu(indicatorValue, bu);
    caffe_mul<Dtype>(N_, bu, horizontal_, au);
    //TODO possible errors due to 0 padding; charbonnier introduces non-zero elements in padding
    charbonnierD1_cpu(au, term1);
    charbonnierD2_cpu(au, term2);
    caffe_mul<Dtype>(N_, bu, term2, term2);
    caffe_mul<Dtype>(N_, horizontal_, term2, term2);
    caffe_axpy<Dtype>(N_, 1, term1, term2);

    // Bh * (H^-1 * lossGrad)
    this->timesHorizontalB_cpu(unitGrad, horizontalGrad);
    caffe_mul<Dtype>(N_, term2, horizontalGrad, horizontalGrad);

    //vertical
    timesVerticalB_cpu(indicatorValue, bu);
    caffe_mul<Dtype>(N_, bu, vertical_, au);
    //TODO possible errors due to 0 padding; charbonnier introduces non-zero elements in padding
    charbonnierD1_cpu(au, term1);
    charbonnierD2_cpu(au, term2);
    caffe_mul<Dtype>(N_, bu, term2, term2);
    caffe_mul<Dtype>(N_, vertical_, term2, term2);
    caffe_axpy<Dtype>(N_, 1, term1, term2);

    // Bv * (H^-1 * lossGrad)
    this->timesVerticalB_cpu(unitGrad, verticalGrad);
    caffe_mul<Dtype>(N_, term2, verticalGrad, verticalGrad);

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
Dtype SegmentationLayer<Dtype>::energy_cpu(const Dtype* indicatorValue) {

  Dtype energySmooth = 0;
  Dtype energyData = 0;
  Dtype energyLogBarrier = 0;

  Dtype* eHoriz = bufferEnergyGrad_.mutable_cpu_data();
  this->timesHorizontalB_cpu(indicatorValue, eHoriz);
  caffe_mul<Dtype>(N_, eHoriz, this->horizontal_, eHoriz);

  Dtype* eVert = bufferEnergyGrad_.mutable_cpu_diff();
  this->timesVerticalB_cpu(indicatorValue, eVert);
  caffe_mul<Dtype>(N_, eVert, this->vertical_, eVert);

  Dtype eps = this->smoothnesEps_ * this->smoothnesEps_;
  for(int i = 0; i < N_; ++i) {
    energySmooth += sqrt(eHoriz[i] * eHoriz[i] + eps);
    energySmooth += sqrt(eVert[i] * eVert[i] + eps);
    energyData += this->unit_[i] * indicatorValue[i];
    energyLogBarrier += log(indicatorValue[i]) + log(1 - indicatorValue[i]);
  }

  return energySmooth + this->dataWeight_ * energyData - this->logBarrierWeight_ * energyLogBarrier;
}


template<typename Dtype>
void SegmentationLayer<Dtype>::minimize_cpu(Dtype* indicatorValue,
                                            Dtype* indicatorGrad) {

  computeEnergyGradient_cpu(indicatorValue, indicatorGrad);
  Dtype gradientNorm = caffe_cpu_dot<Dtype>(N_, indicatorGrad, indicatorGrad);
  Dtype toleranceSquare = gradientTolerance_ * gradientTolerance_;
  int iter = 0;
  LOG(INFO) << "Energy at iter #" << iter << " = " << energy_cpu(indicatorValue);
  while (gradientNorm > toleranceSquare && iter++ < minimizationIters_) {
//    printVec(indicatorValue);
//    LOG(INFO) << "Energy at #iter: " << iter << " = " << energy_cpu(indicatorValue) << "\tGradientNorm = " << gradientNorm;
    caffe_axpy<Dtype>(N_, -stepSize_, indicatorGrad, indicatorValue);
    computeEnergyGradient_cpu(indicatorValue, indicatorGrad);
    gradientNorm = caffe_cpu_dot<Dtype>(N_, indicatorGrad, indicatorGrad);
    if(iter % 100 == 0) {
       LOG(INFO) << "Energy at #iter: " << iter << " = " << energy_cpu(indicatorValue) << "\tGradientNorm = " << gradientNorm;
    }
  }
  LOG(INFO) << "Energy at #iter: " << iter << " = " << energy_cpu(indicatorValue) << "\tGradientNorm = " << gradientNorm;
}

template<typename Dtype>
void SegmentationLayer<Dtype>::computeEnergyGradient_cpu(Dtype* indicatorValue,
                                                         Dtype* indicatorGrad) {
  Dtype* du = bufferEnergyGrad_.mutable_cpu_data();
  Dtype* temp = bufferEnergyGrad_.mutable_cpu_diff();
  // dudx
  timesHorizontalB_cpu(indicatorValue, du);
  caffe_mul<Dtype>(N_, du, horizontal_, du);
  charbonnierD1_cpu(du, du);
  caffe_mul<Dtype>(N_, du, horizontal_, du);
  timesHorizontalBt_cpu(du, temp);
  caffe_copy<Dtype>(N_, temp, indicatorGrad);

  // dudy
  timesVerticalB_cpu(indicatorValue, du);
  caffe_mul<Dtype>(N_, du, vertical_, du);
  charbonnierD1_cpu(du, du);
  caffe_mul<Dtype>(N_, du, vertical_, du);
  timesVerticalBt_cpu(du, temp);
  caffe_axpy<Dtype>(N_, 1, temp, indicatorGrad);

  //gradUnary
  caffe_axpy(N_, dataWeight_, unit_, indicatorGrad);

  //gradLog
  for (int i = 0; i < N_; ++i) {
    du[i] = 1 / indicatorValue[i] - 1 / (1 - indicatorValue[i]);
  }
  caffe_axpy<Dtype>(N_, -logBarrierWeight_, du, indicatorGrad);
}

template<typename Dtype>
void SegmentationLayer<Dtype>::timesHorizontalB_cpu(const Dtype* f, Dtype* dx) {
  // dx has an effective width of (width_ - 1) padded with a zero at the end
  for (int i = 0; i < height_; ++i) {
    caffe_copy<Dtype>(width_ - 1, f, dx);
    caffe_axpy<Dtype>(width_ - 1, -1, f + 1, dx);
    *(dx + width_ - 1) = 0;

    f += width_;
    dx += width_;
  }
}

template<typename Dtype>
void SegmentationLayer<Dtype>::timesVerticalB_cpu(const Dtype* f, Dtype* dy) {
  caffe_copy<Dtype>(N_ - width_, f, dy);
  caffe_axpy<Dtype>(N_ - width_, -1, f + width_, dy);
  // pad with zeros
  caffe_set<Dtype>(width_, 0, dy + N_ - width_);
}

/**
 * diffDx has an effective width of (width_ - 1) padded with a zero at the end
 * after the routine it should be:
 *
 *    diffDx = [grad(:, 1) (grad(:, 2:end) - grad(:, 1:end-1)) -grad(:, end)
 *
 * where: end == width_ -1 and effective width becomes width_
 */
template<typename Dtype>
void SegmentationLayer<Dtype>::timesHorizontalBt_cpu(const Dtype* grad,
                                                     Dtype* diffDx) {
  // diffDx = grad(:, :)
  caffe_copy<Dtype>(N_, grad, diffDx);
  for (int i = 0; i < height_; ++i) {
    // ensure that it is in fact padded with a zero
    // or in matlab notation: diffDx[end+1==width_] = 0
    *(diffDx + width_ - 1) = 0;
    // subtract grad(:, 1:end) from diffDx(:, 2:end+1)
    caffe_axpy<Dtype>(width_ - 1, -1, grad, diffDx + 1);
    grad += width_;
    diffDx += width_;
  }
}

template<typename Dtype>
void SegmentationLayer<Dtype>::timesVerticalBt_cpu(const Dtype* grad,
                                                   Dtype* diffDy) {
  caffe_copy<Dtype>(N_ - width_, grad, diffDy);
  caffe_set<Dtype>(width_, 0, diffDy + N_ - width_);
  caffe_axpy<Dtype>(N_ - width_, -1, grad, diffDy + width_);
}


template<typename Dtype>
struct double_converter {
  static void to(int N, const Dtype* X, double* Y) {
      for(int i = 0; i < N; ++i) {
        Y[i] = static_cast<double>(X[i]);
      }
    }

  static void from(int N, const double* X, Dtype* Y)  {
    for(int i = 0; i < N; ++i) {
      Y[i] = static_cast<Dtype>(X[i]);
    }
  }
};


template<typename Dtype>
void SegmentationLayer<Dtype>::computeSparseHessian_cpu(const Dtype* indicator) {

  Dtype* du = bufferEnergyGrad_.mutable_cpu_data();

  // super diagonals
  Dtype* diagP2 = bufferHessian_.mutable_cpu_data();
  Dtype* diagP1 = diagP2 + N_;

  // diag
  Dtype* diag = diagP1 + N_;

  // sub diagonals
  Dtype* diagM1 = diag + N_;
  Dtype* diagM2 = diagM1 + N_;

  caffe_set<Dtype>(5 * N_, 0, diagP2);

// horizontal =========================
  caffe_set<Dtype>(N_, 0, du);
  timesHorizontalB_cpu(indicator, du);
  caffe_mul<Dtype>(N_, du, horizontal_, du);
  charbonnierD2_cpu(du, du);
  // we're using padded du by 1 column; charbonnier puts there non-zero values which have to be taken care of
  zeroLastColumn_cpu(du);
  caffe_mul<Dtype>(N_, du, horizontal_, du);
  caffe_mul<Dtype>(N_, du, horizontal_, du);

  // diag
  caffe_copy<Dtype>(N_, du, diag);
  for (int i = 0; i < height_; ++i) {
    size_t offset = i * width_;
    diag[offset + width_ - 1] = 0;
    // add g(:, 1:end) to g(:, 2:end+1)
    caffe_axpy<Dtype>(width_ - 1, 1, du + offset, diag + offset + 1);
  }

  //superdiag
  caffe_axpy<Dtype>(N_ - 1, -1, du, diagP1);

  //subdiag
  caffe_axpy<Dtype>(N_ - 1, -1, du, diagM1 + 1);

//  vertical =======================================================
  caffe_set<Dtype>(N_, 0, du);
  timesVerticalB_cpu(indicator, du);
  caffe_mul<Dtype>(N_, du, vertical_, du);
  charbonnierD2_cpu(du, du);
  zeroLastRow_cpu(du);
  caffe_mul<Dtype>(N_, du, vertical_, du);
  caffe_mul<Dtype>(N_, du, vertical_, du);

  // diag
  caffe_axpy<Dtype>(N_ - width_, 1, du, diag);
  caffe_axpy<Dtype>(N_ - width_, 1, du, diag + width_);

  //superdiag
  caffe_axpy<Dtype>(N_ - width_, -1, du, diagP2);
//
  //subdiag
  caffe_axpy<Dtype>(N_ - width_, -1, du, diagM2 + width_);

// log barrier =======================================================
  for(int i = 0; i < N_; ++i) {
    Dtype u = indicator[i];
    u = 1 / (u * u) + 1 / ((1 - u) * (1 - u));
    diag[i] += this->logBarrierWeight_ * u;
  }
}

template<typename Dtype>
void SegmentationLayer<Dtype>::approxHessVec_cpu(const Dtype* indicator,
                                                 const Dtype* vec, Dtype* Hv) {
  Dtype maxElem = fabs(vec[0]);
  for(int i = 1; i < N_; ++i) {
    if(fabs(vec[i]) > maxElem) {
      maxElem = fabs(vec[i]);
    }
  }

  Dtype eps = fmax(1e-8, fmin(1, 1e-4 / maxElem));
  Dtype* point = bufferHessVec_.mutable_cpu_data();
  Dtype* grad = bufferHessVec_.mutable_cpu_diff();

  caffe_set<Dtype>(N_, 0, Hv);

//  //  2nd order
//  //f(x + h)
//  caffe_copy<Dtype>(N_, indicator, point);
//  caffe_axpy<Dtype>(N_, eps, vec, point);
//  computeEnergyGradient_cpu(point, grad);
//  caffe_axpy<Dtype>(N_, 0.5 / eps, grad, Hv);
//
//  //f(x - h)
//  caffe_copy<Dtype>(N_, indicator, point);
//  caffe_axpy<Dtype>(N_, -eps, vec, point);
//  computeEnergyGradient_cpu(point, grad);
//
//  //Subtract
//  caffe_axpy<Dtype>(N_, -0.5 / eps, grad, Hv);



  //  4th order
  //f(x + 2h)
  caffe_copy<Dtype>(N_, indicator, point);
  caffe_axpy<Dtype>(N_, 2*eps, vec, point);
  computeEnergyGradient_cpu(point, grad);
  caffe_axpy<Dtype>(N_, -1 / (12*eps), grad, Hv);

  //f(x - 2h)
  caffe_copy<Dtype>(N_, indicator, point);
  caffe_axpy<Dtype>(N_, -2*eps, vec, point);
  computeEnergyGradient_cpu(point, grad);
  caffe_axpy<Dtype>(N_, 1 / (12*eps), grad, Hv);

  //f(x + h)
  caffe_copy<Dtype>(N_, indicator, point);
  caffe_axpy<Dtype>(N_, eps, vec, point);
  computeEnergyGradient_cpu(point, grad);
  caffe_axpy<Dtype>(N_, 2 / (3*eps), grad, Hv);

  //f(x - 2h)
  caffe_copy<Dtype>(N_, indicator, point);
  caffe_axpy<Dtype>(N_, -eps, vec, point);
  computeEnergyGradient_cpu(point, grad);
  caffe_axpy<Dtype>(N_, -2 / (3*eps), grad, Hv);
}

template<typename Dtype>
void SegmentationLayer<Dtype>::sparseHessianMultiply_cpu(const Dtype* vec, Dtype* out) {

  // super diagonals
  const Dtype* diagP2 = bufferHessian_.cpu_data();
  const Dtype* diagP1 = diagP2 + N_;

  // diag
  const Dtype* diag = diagP1+ N_;

  // sub diagonals
  const Dtype* diagM1 = diag + N_;
  const Dtype* diagM2 = diagM1 + N_;

  out[0] = diag[0] * vec[0] + diagP1[0] * vec[1] + diagP2[0] * vec[width_];
  for(int i = 1; i < width_; ++i) {
    out[i] = diagM1[i] * vec[i - 1] + diag[i] * vec[i] + diagP1[i] * vec[i + 1] + diagP2[i] * vec[i + width_];
  }
  for(int i = width_; i < N_ - width_; ++i) {
    out[i] = diagM2[i] * vec[i - width_] + diagM1[i] * vec[i - 1] + diag[i] * vec[i] + diagP1[i] * vec[i + 1] + diagP2[i] * vec[i + width_];
  }
  for(int i = N_ - width_; i < N_ - 1; ++i) {
    out[i] = diagM2[i] * vec[i - width_] + diagM1[i] * vec[i - 1] + diag[i] * vec[i] + diagP1[i] * vec[i + 1];
  }

  out[N_ - 1] = diagM2[N_ - 1] * vec[N_ - width_ - 1] + diagM1[N_ - 1] * vec[N_ - 2] + diag[N_ - 1] * vec[N_ - 1];
}

/**
 * Conjugate Gradient
 */
template<typename Dtype>
void SegmentationLayer<Dtype>::invHessianVector_cpu(const Dtype* indicator,
                                                    const Dtype* vec,
                                                    Dtype* iHv) {

  Dtype* residual = bufferResidualDirection_.mutable_cpu_data();
  Dtype* direction = bufferResidualDirection_.mutable_cpu_diff();
  Dtype* matVecProd = bufferMatVecStorage_.mutable_cpu_data();
  Dtype* newDirectionBuffer = bufferMatVecStorage_.mutable_cpu_diff();

  computeSparseHessian_cpu(indicator);

  //initialize iHv
  caffe_set<Dtype>(N_, 1.0, iHv);



//  approxHessVec_cpu(indicator, iHv, matVecProd);

// residual = b - Ax
    caffe_set<Dtype>(N_, 0, residual);
    caffe_axpy<Dtype>(N_, -1, vec, residual);
    sparseHessianMultiply_cpu(iHv, matVecProd);
    caffe_axpy<Dtype>(N_, 1, matVecProd, residual);

//  caffe_copy<Dtype>(N_, vec, residual);
//  sparseHessianMultiply_cpu(iHv, matVecProd);
//  caffe_axpy<Dtype>(N_, -1, matVecProd, residual);

  //direction
  caffe_copy<Dtype>(N_, residual, direction);

  Dtype residualNorm = caffe_cpu_nrm2<Dtype>(N_, residual);
  Dtype residualNormOld = 0;
  int iter = 0;
  while (iter++ < invHessIters_) {
    LOG(INFO) << "ResidualNorm at #iter = " << iter-1 << " = " << residualNorm;

//    approxHessVec_cpu(indicator, direction, matVecProd);
    sparseHessianMultiply_cpu(direction, matVecProd);

    Dtype alpha = residualNorm * residualNorm
        / caffe_cpu_dot<Dtype>(N_, direction, matVecProd);

    //update iHv
    caffe_axpy<Dtype>(N_, alpha, direction, iHv);

    // new residual
    residualNormOld = residualNorm;
    caffe_axpy<Dtype>(N_, -alpha, matVecProd, residual);
    residualNorm = caffe_cpu_nrm2<Dtype>(N_, residual);

    if (residualNorm < invHessTolerance_) {
      LOG(INFO) << "Terminated at #iter = " << iter << " with ResidualNorm = " << residualNorm;
      break;
    }

    // new direction
    Dtype beta = -residualNorm * residualNorm / (residualNormOld * residualNormOld);
    caffe_copy<Dtype>(N_, residual, newDirectionBuffer);
    caffe_axpy<Dtype>(N_, -beta, direction, newDirectionBuffer);
    caffe_copy<Dtype>(N_, newDirectionBuffer, direction);
  }
}

template<typename Dtype>
void SegmentationLayer<Dtype>::charbonnierD1_cpu(const Dtype* source,
                                                 Dtype* dest) {

  Dtype eps = this->smoothnesEps_ * this->smoothnesEps_;
  for (int i = 0; i < N_; ++i) {
    dest[i] = source[i] / sqrt(source[i] * source[i] + eps);
  }
}

template<typename Dtype>
void SegmentationLayer<Dtype>::charbonnierD2_cpu(const Dtype* source,
                                                 Dtype* dest) {

  Dtype eps = this->smoothnesEps_ * this->smoothnesEps_;
  for (int i = 0; i < N_; ++i) {
    Dtype denom = source[i] * source[i] + eps;
    dest[i] = eps / (denom * sqrt(denom));
  }
}

template<typename Dtype>
void SegmentationLayer<Dtype>::zeroLastRow_cpu(Dtype* v) {
  caffe_set<Dtype>(width_, 0, v + N_ - width_);
}

template<typename Dtype>
void SegmentationLayer<Dtype>::zeroLastColumn_cpu(Dtype* v) {

  for (int i = width_ - 1; i < N_; i += width_) {
    v[i] = 0;
  }
}

INSTANTIATE_CLASS(SegmentationLayer);
REGISTER_LAYER_CLASS(Segmentation);

}  // namespace caffe
