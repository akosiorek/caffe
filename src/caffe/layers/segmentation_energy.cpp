#include "segmentation_energy.h"

using namespace Eigen;

namespace caffe {

template<typename Dtype>
SegmentationEnergy<Dtype>::SegmentationEnergy(const SegmentationParameter& param, shared_ptr<Blob<Dtype>> dataWeight)
    : dataWeight_(dataWeight) {

    this->logBarrierWeight_ = param.log_barrier_weight();
    this->smoothnesEps_ = param.smoothnes_eps();
    this->convexParam_ = param.convex_param();
    this->gradientNormTolerance_ = param.min_update_norm();
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::reshape(int width, int height) {

    this->width_ = width;
    this->height_ = height;
    this->N_ = width * height;

    bufferEnergyGrad_.Reshape(1, 1, height_, width_);
    bufferMinimization_.Reshape(1, 1, height_, width_);
    bufferHessian_.Reshape(5, 1, height_, width_);
    bufferArgMinGrapMap_.Reshape(4, 1, height_, width_);
    bufferArgMinEstFuns_.Reshape(4, 1, height_, width_);
    bufferNCOBF1_.Reshape(1, 1, height_, width_);
    bufferNCOBF2_.Reshape(1, 1, height_, width_);
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::setData(const Blob <Dtype> *unit, const Blob <Dtype> *horizontal, const Blob <Dtype> *vertical) {

    this->unitaryPotential_ = unit;
    this->horizontalPotential_ = horizontal;
    this->verticalPotential_ = vertical;
}

template<typename Dtype>
Dtype SegmentationEnergy<Dtype>::energy_cpu(const Dtype*indicator) const {

    Dtype energySmooth = 0;
    Dtype energyData = 0;
    Dtype energyLogBarrier = 0;

    Dtype* eHoriz = bufferEnergyGrad_.mutable_cpu_data();
    Dtype* eVert = bufferEnergyGrad_.mutable_cpu_diff();


    this->timesHorizontalB_cpu(indicator, eHoriz);
    caffe_mul<Dtype>(N_, eHoriz, this->horizontalPotential_->cpu_data(), eHoriz);

    this->timesVerticalB_cpu(indicator, eVert);
    caffe_mul<Dtype>(N_, eVert, this->verticalPotential_->cpu_data(), eVert);

    Dtype eps = this->smoothnesEps_ * this->smoothnesEps_;
    for(int i = 0; i < N_; ++i) {
        energySmooth += sqrt(eHoriz[i] * eHoriz[i] + eps);
        energySmooth += sqrt(eVert[i] * eVert[i] + eps);
        energyData += this->unitaryPotential_->cpu_data()[i] * indicator[i];
        energyLogBarrier += log(indicator[i]) + log(1 - indicator[i]);
    }

    return energySmooth + this->dataWeight_->cpu_data()[0] * energyData - this->logBarrierWeight_ * energyLogBarrier;
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::computeEnergyGradientPiecewise_cpu(const Dtype* indicator,
                                                                   Dtype* grad) const {

    Dtype* du = bufferEnergyGrad_.mutable_cpu_data();
    Dtype* temp = bufferEnergyGrad_.mutable_cpu_diff();

    // dudx
    timesHorizontalB_cpu(indicator, du);
    caffe_mul<Dtype>(N_, du, this->horizontalPotential_->cpu_data(), du);
    charbonnierD1_cpu(du, du);
    caffe_mul<Dtype>(N_, du, this->horizontalPotential_->cpu_data(), du);
    timesHorizontalBt_cpu(du, temp);
    caffe_copy<Dtype>(N_, temp, grad);

    // dudy
    timesVerticalB_cpu(indicator, du);
    caffe_mul<Dtype>(N_, du, this->verticalPotential_->cpu_data(), du);
    charbonnierD1_cpu(du, du);
    caffe_mul<Dtype>(N_, du, this->verticalPotential_->cpu_data(), du);
    timesVerticalBt_cpu(du, temp);
    caffe_axpy<Dtype>(N_, 1, temp, grad);
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::computeEnergyGradient_cpu(const Dtype* indicator,
                                                         Dtype* grad) const {
    computeEnergyGradientPiecewise_cpu(indicator, grad);

    //gradUnary
    caffe_axpy(N_, this->dataWeight_->cpu_data()[0], this->unitaryPotential_->cpu_data(), grad);

    //gradLog
    for (int i = 0; i < N_; ++i) {
        grad[i] -= logBarrierWeight_ * (1 / indicator[i] - 1 / (1 - indicator[i]));
    }
}

template<typename Dtype>
bool SegmentationEnergy<Dtype>::getValidRoots(int N, Dtype* a, Dtype* b, Dtype* c, Dtype* d, Dtype* result) const {

  bool isValid = true;
  for(int i = 0; i < N; ++i) {

    LOG_IF(ERROR, a[i] == 0) << "a == 0";
    LOG_IF(ERROR, b[i] == 0) << "b == 0";
    LOG_IF(ERROR, c[i] == 0) << "c == 0";
    LOG_IF(ERROR, d[i] == 0) << "d == 0";

    Dtype bi = b[i] / a[i];
    Dtype ci = c[i] / a[i];
    Dtype di = d[i] / a[i];

    Dtype q = -(3 * ci - bi * bi) / 9;
    Dtype r = -(27 * di) + bi * (9 * ci - 2 * bi * bi);
    r /= 54;

    Dtype r13 = 2 * std::sqrt(q);

    q = q * q * q;
    q = std::acos(r / std::sqrt(q));
    result[i] = -(bi / 3) + r13 * std::cos((q + 4 * (Dtype)M_PI) / 3);

    isValid = isValid && (result[i] > 0) && (result[i] < 1);

    if(!isValid) {
      LOG_FIRST_N(ERROR, 1) << "iter: " << i << "\na = " << a[i] << "\nb = " << b[i] << "\nc = " <<
                                                                   c[i] << "\nd = " << d[i] << "\nroot = " << result[i];
    }
  }
  return isValid;
}

template<typename Dtype>
bool SegmentationEnergy<Dtype>::argMinGradMap(Dtype L, const Dtype *y,
                                              Dtype *argMin) const {
    Dtype* a = bufferArgMinGrapMap_.mutable_cpu_data();
    Dtype* b = a + N_;
    Dtype* c = b + N_;
    Dtype* d = c + N_;

    // save piecewiese energy gradient in b
    computeEnergyGradientPiecewise_cpu(y, b);
    caffe_axpy<Dtype>(N_, -L, y, b);
    caffe_axpy<Dtype>(N_, dataWeight_->cpu_data()[0], unitaryPotential_->cpu_data(), b);

    // a
    caffe_set<Dtype>(N_, L, a);

    // b
    caffe_add_scalar<Dtype>(N_, -L, b);

    // c
    caffe_set<Dtype>(N_, -2 * logBarrierWeight_ -L, c);
    caffe_axpy<Dtype>(N_, -1, b, c);

    // d
    caffe_set<Dtype>(N_, logBarrierWeight_, d);

    return getValidRoots(N_, a, b, c, d, argMin);
};

template<typename Dtype>
bool SegmentationEnergy<Dtype>::argMinEstFun(Dtype ak, const Dtype* v, Dtype* argMin) const {

    Dtype* a = bufferArgMinEstFuns_.mutable_cpu_data();
    Dtype* b = a + N_;
    Dtype* c = b + N_;
    Dtype* d = c + N_;

  // update coefficients
  // a == 1 all the time, thus no update
  // b
  computeEnergyGradientPiecewise_cpu(v, argMin);
  caffe_axpy<Dtype>(N_, this->dataWeight_->cpu_data()[0], unitaryPotential_->cpu_data(), argMin);
  caffe_axpy<Dtype>(N_, ak, argMin, b);

  // c
  caffe_add_scalar<Dtype>(N_, 2 * logBarrierWeight_, argMin);
  caffe_axpy<Dtype>(N_, -ak, argMin, c);

  // d
  caffe_add_scalar<Dtype>(N_, + logBarrierWeight_ * ak, d);

  return getValidRoots(N_, a, b, c, d, argMin);
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::minimizeNCOBF_cpu(Dtype* x) const {

    static const Dtype gammaU = 2;  // Lipschitz constant scaling parameters
    static const Dtype gammaD = 2;  // Author mentioned that gammaU = gammaD is a reasonable choice

    Dtype* v = bufferNCOBF1_.mutable_cpu_data();
    Dtype* y = bufferNCOBF1_.mutable_cpu_diff();
    Dtype* t = bufferNCOBF2_.mutable_cpu_data();
    Dtype* grad = bufferNCOBF2_.mutable_cpu_diff();
    Dtype* oldX = bufferMinimization_.mutable_cpu_data();


    // compute L
    Dtype maxHorizontal = *std::max_element(horizontalPotential_->cpu_data(), horizontalPotential_->cpu_data() + N_);
    Dtype maxVertical = *std::max_element(verticalPotential_->cpu_data(), verticalPotential_->cpu_data() + N_);
    Dtype maxPiecewise = std::max(maxHorizontal, maxVertical);
    Dtype L = maxPiecewise * 8 / smoothnesEps_;
    LOG(ERROR) << "max: " << maxPiecewise << " L: " << L;

    // init stuff
    Dtype A = 0;
    Dtype ak = 0;
    caffe_copy<Dtype>(N_, x, v);

    // intit coeffs for computing v
    {
      Dtype* estFunCoeffs = bufferArgMinEstFuns_.mutable_cpu_data();

      // initialize est fun coeffs
      caffe_set<Dtype>(N_, 1, estFunCoeffs);
      caffe_set<Dtype>(N_, -1, estFunCoeffs + N_);
      caffe_axpy<Dtype>(N_, -1, x, estFunCoeffs + N_);
      caffe_set<Dtype>(2 * N_, 0, estFunCoeffs + 2 * N_);
    }

    int iter = 0;
    LOG(INFO) << "Energy at iter #" << iter << " = " << energy_cpu(x);

  computeEnergyGradient_cpu(x, grad);
  Dtype gradNorm = caffe_cpu_nrm2<Dtype>(N_, grad);
  LOG(ERROR) << "iter: " << iter << " grad norm: " << gradNorm;
//      while(iter < minimizationIters_) {
  while(gradNorm > gradientNormTolerance_) {
      caffe_copy<Dtype>(N_, x, oldX);

      Dtype gradDiff = 0;
      Dtype gradConvexEst = std::numeric_limits<Dtype>::max();
      L /= gammaU;
      while(gradDiff < gradConvexEst) {
        L *= gammaU;
        Dtype k = (1 + convexParam_ * A) / L;
        ak = k + std::sqrt(k * (1/L + A));

        // compute y
        caffe_cpu_scale<Dtype>(N_, A / (A + ak), x, y);
        caffe_axpy<Dtype>(N_, ak / (A + ak), v, y);

        // grad at gradMapMinimizer
        if(!argMinGradMap(L, y, t)) {
          LOG(FATAL) << "invalid argMinGradMap - inner loop\n L = " << L;
        }
        computeEnergyGradient_cpu(t, grad);

        caffe_axpy<Dtype>(N_, -1, y, t);
        gradDiff = -caffe_cpu_dot<Dtype>(N_, t, grad);
        gradConvexEst = caffe_cpu_nrm2<Dtype>(N_, grad);
        gradConvexEst *= gradConvexEst  / L;
      }

      A += ak;
      if(!argMinGradMap(L, y, x)) {
        LOG(FATAL) << "invalid argMinGradMap - outer loop";
      }
      L /= gammaD;

      if(iter % 50 == 0) {
        computeEnergyGradient_cpu(x, grad);
        gradNorm = caffe_cpu_nrm2<Dtype>(N_, grad);
      }

      // compute new v
      if(!argMinEstFun(ak, x, v)) {
        LOG(FATAL) << "invalid argMinEstFun - outer loop";
      }
      ++iter;
//      LOG(INFO) << "Energy at iter #" << iter << " = " << energy_cpu(x);
    }

    LOG(INFO) << "Iter #" << iter << " Minimized energy = " << energy_cpu(x);
  LOG(ERROR) << "iter: " << iter << " grad norm: " << gradNorm;
}


template<typename Dtype>
void SegmentationEnergy<Dtype>::timesHorizontalB_cpu(const Dtype* in, Dtype* out) const {
    // dx has an effective width of (width_ - 1) padded with a zero at the end
    for (int i = 0; i < height_; ++i) {
        caffe_copy<Dtype>(width_ - 1, in, out);
        caffe_axpy<Dtype>(width_ - 1, -1, in + 1, out);
        *(out + width_ - 1) = 0;

        in += width_;
        out += width_;
    }
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::timesVerticalB_cpu(const Dtype* in, Dtype* out) const {
    caffe_copy<Dtype>(N_ - width_, in, out);
    caffe_axpy<Dtype>(N_ - width_, -1, in + width_, out);
    // pad with zeros
    caffe_set<Dtype>(width_, 0, out + N_ - width_);
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
void SegmentationEnergy<Dtype>::timesHorizontalBt_cpu(const Dtype* in,
                                                     Dtype* out) const {
    // diffDx = grad(:, :)
    caffe_copy<Dtype>(N_, in, out);
    for (int i = 0; i < height_; ++i) {
        // ensure that it is in fact padded with a zero
        // or in matlab notation: diffDx[end+1==width_] = 0
        *(out + width_ - 1) = 0;
        // subtract grad(:, 1:end) from diffDx(:, 2:end+1)
        caffe_axpy<Dtype>(width_ - 1, -1, in, out + 1);
        in += width_;
        out += width_;
    }
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::timesVerticalBt_cpu(const Dtype* in,
                                                   Dtype* out) const {
    caffe_copy<Dtype>(N_ - width_, in, out);
    caffe_set<Dtype>(width_, 0, out + N_ - width_);
    caffe_axpy<Dtype>(N_ - width_, -1, in, out + width_);
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::computeSparseHessian_cpu(const Dtype* indicator) const {

  Dtype *du = bufferHessian_.mutable_cpu_diff();

  // super diagonals
  Dtype *diagP2 = bufferHessian_.mutable_cpu_data();
  Dtype *diagP1 = diagP2 + N_;

  // diag
  Dtype *diag = diagP1 + N_;

  // sub diagonals
  Dtype *diagM1 = diag + N_;
  Dtype *diagM2 = diagM1 + N_;

  caffe_set<Dtype>(5 * N_, 0, diagP2);

// horizontal =========================
  caffe_set<Dtype>(N_, 0, du);
  timesHorizontalB_cpu(indicator, du);
  caffe_mul<Dtype>(N_, du, this->horizontalPotential_->cpu_data(), du);
  charbonnierD2_cpu(du, du);
  // we're using padded du by 1 column; charbonnier puts there non-zero values which have to be taken care of
  zeroLastColumn_cpu(du);
  caffe_mul<Dtype>(N_, du, this->horizontalPotential_->cpu_data(), du);
  caffe_mul<Dtype>(N_, du, this->horizontalPotential_->cpu_data(), du);

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
  caffe_mul<Dtype>(N_, du, this->verticalPotential_->cpu_data(), du);
  charbonnierD2_cpu(du, du);
  zeroLastRow_cpu(du);
  caffe_mul<Dtype>(N_, du, this->verticalPotential_->cpu_data(), du);
  caffe_mul<Dtype>(N_, du, this->verticalPotential_->cpu_data(), du);

  // diag
  caffe_axpy<Dtype>(N_ - width_, 1, du, diag);
  caffe_axpy<Dtype>(N_ - width_, 1, du, diag + width_);

  //superdiag
  caffe_axpy<Dtype>(N_ - width_, -1, du, diagP2);
//
  //subdiag
  caffe_axpy<Dtype>(N_ - width_, -1, du, diagM2 + width_);

// log barrier =======================================================
  for (int i = 0; i < N_; ++i) {
    Dtype u = indicator[i];
    u = 1 / (u * u) + 1 / ((1 - u) * (1 - u));
    diag[i] += this->logBarrierWeight_ * u;
  }
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::invHessianVector_cpu(const Dtype* indicator,
                                                    const Dtype* vec,
                                                    Dtype* iHv) const {
    typedef Matrix<Dtype, Dynamic, 1> VectorType;

    this->computeSparseHessian_cpu(indicator);
    auto hessian = convertHessianToEigenSparse();

    SparseLU<SparseMatrixT> solver;
    solver.analyzePattern(hessian);
    solver.factorize(hessian);

    const Map<const VectorType> vector(vec, N_, 1);
    Map<VectorType> result(iHv, N_, 1);
    result = solver.solve(vector);
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::charbonnierD1_cpu(const Dtype* in,
                                                 Dtype* out) const {

    Dtype eps = this->smoothnesEps_ * this->smoothnesEps_;
    for (int i = 0; i < N_; ++i) {
        out[i] = in[i] / sqrt(in[i] * in[i] + eps);
    }
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::charbonnierD2_cpu(const Dtype* in,
                                                 Dtype* out) const {

    Dtype eps = this->smoothnesEps_ * this->smoothnesEps_;
    for (int i = 0; i < N_; ++i) {
        Dtype denom = in[i] * in[i] + eps;
        out[i] = eps / (denom * sqrt(denom));
    }
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::zeroLastRow_cpu(Dtype* v) const {
    caffe_set<Dtype>(width_, 0, v + N_ - width_);
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::zeroLastColumn_cpu(Dtype* v) const {
    for (int i = width_ - 1; i < N_; i += width_) {
        v[i] = 0;
    }
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::makeTriplesfromDiag(std::vector<Triplet<Dtype>>& to, const Dtype* from, int xOffset, int yOffset, int N) const {

    for(int i = 0; i < N; ++i) {
        to.emplace_back(yOffset + i, xOffset + i, from[i]);
    }
}

template<typename Dtype>
auto SegmentationEnergy<Dtype>::convertHessianToEigenSparse() const -> SparseMatrixT {

  SparseMatrix<Dtype, RowMajor> hessian(N_, N_);

  std::vector<Triplet<Dtype>> triplets;
  triplets.reserve(5 * N_ - 2 * (width_ + 1));

  const Dtype *diagP2 = bufferHessian_.cpu_data();
  const Dtype *diagP1 = diagP2 + N_;
  const Dtype *diag = diagP1 + N_;
  const Dtype *diagM1 = diag + N_;
  const Dtype *diagM2 = diagM1 + N_;

  makeTriplesfromDiag(triplets, diagM2 + width_, width_, 0, N_ - width_);
  makeTriplesfromDiag(triplets, diagM1 + 1, 1, 0, N_ - 1);
  makeTriplesfromDiag(triplets, diag, 0, 0, N_);
  makeTriplesfromDiag(triplets, diagP1, 0, 1, N_ - 1);
  makeTriplesfromDiag(triplets, diagP2, 0, width_, N_ - width_);

  hessian.setFromTriplets(triplets.begin(), triplets.end());
  return hessian;
}

INSTANTIATE_CLASS(SegmentationEnergy);
}  // namespace caffe
