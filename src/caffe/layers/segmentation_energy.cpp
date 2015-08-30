#include "segmentation_energy.h"

using namespace Eigen;

namespace caffe {

template<typename Dtype>
SegmentationEnergy<Dtype>::SegmentationEnergy(const SegmentationParameter& param, shared_ptr<Blob<Dtype>> dataWeight)
    : dataWeight_(dataWeight) {

    this->logBarrierWeight_ = param.log_barrier_weight();
    this->smoothnesEps_ = param.smoothnes_eps();
    this->minimizationIters_ = param.minimization_iters();
    this->convexParam_ = param.convex_param();
    this->initLipschnitzConstant_ = param.init_lipschitz_constant();
    this->minUpdateNorm_ = param.min_update_norm();
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


template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template<typename Dtype>
Dtype SegmentationEnergy<Dtype>::cubicRoot(Dtype x) const {
    return sgn(x) * std::cbrt(std::abs(x));
}

template<typename Dtype>
std::array<std::complex<Dtype>, 3> SegmentationEnergy<Dtype>::cubicRoots(Dtype a, Dtype b, Dtype c, Dtype d) const {

    using ComplexT = std::complex<Dtype>;
    static const ComplexT i(0, 1);
    std::array<ComplexT, 3> roots;

    CHECK(a != 0);
//    CHECK(d != 0);

    b /= a;
    c /= a;
    d /= a;

    Dtype q = (3 * c - b * b) / 9;
    Dtype r = -(27 * d) + b * (9 * c - 2 * b * b);
    r = r / 54;

    Dtype disc = q * q * q + r * r;
    Dtype term1 = b / 3;

    // one root real, two are complex
    if (disc > 0) {
        Dtype s = cubicRoot(r + std::sqrt(disc));
        Dtype t = cubicRoot(r - std::sqrt(disc));
        roots[0] = -term1 + s + t;
        term1 = term1 + (s + t) / 2;

        roots[1] = roots[2] = -term1;
        term1 = std::sqrt(3) * (-t + s) / 2;
        roots[1] += i * term1;
        roots[2] -= i * term1;

        // The remaining options are all real
    } else if (disc == 0) { // All roots real, at least two are equal.

        Dtype r13 = cubicRoot(r);
        roots[0] = -term1 + 2.0 * r13;
        roots[1] = roots[2] = -(r13 + term1);
    } else {// Only option left is that all roots are real and unequal (to get here, q < 0)
        q = -q;
        Dtype dum1 = q * q * q;
        dum1 = std::acos(r / std::sqrt(dum1));
        Dtype r13 = 2 * std::sqrt(q);
        roots[0] = -term1 + r13 * std::cos(dum1 / 3);
        roots[1] = -term1 + r13 * std::cos((dum1 + 2 * (Dtype)M_PI) / 3);
        roots[2] = -term1 + r13 * std::cos((dum1 + 4 * (Dtype)M_PI) / 3);
    }
    return roots;
}

template<typename Dtype>
bool SegmentationEnergy<Dtype>::getValidRoots(int N, Dtype* a, Dtype* b, Dtype* c, Dtype* d, Dtype* result) const {

    for(int i = 0; i < N; ++i) {
        auto roots = cubicRoots(a[i], b[i], c[i], d[i]);
        bool set = false;
        for(int j = 0; j < 3; ++j) {
            const auto& root = roots[j];
//            LOG(INFO) << root;
            if(root.imag() == 0 && root.real() > 0 && root.real() < 1) {
                result[i] = root.real();
//                LOG(INFO) << root;
                set = true;
                break;
            }
        }
        if(!set) {
            LOG(FATAL) << "Only imaginary or invalid roots!\n"  \
                   << "iter: " << i << "\n" \
                   << "coeffs: " << a[i] << " " << b[i] << " " << c[i] << " " << d[i] << "\n" \
                   << "roots: " << roots[0] << " ," << roots[1] << " ," << roots[2];
            return false;
        }

    }
    return true;
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
bool SegmentationEnergy<Dtype>::argMinEstFun(Dtype L, Dtype ak, const Dtype* v, Dtype* argMin) const {

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
Dtype SegmentationEnergy<Dtype>::gradMapValue(Dtype L, const Dtype* x, const Dtype* y) const {

  Dtype energySmooth = 0;
  Dtype energyData = 0;
  Dtype energyLogBarrier = 0;
  Dtype gradMap = 0;

  Dtype* eHoriz = bufferEnergyGrad_.mutable_cpu_data();
  Dtype* eVert = bufferEnergyGrad_.mutable_cpu_diff();


  this->timesHorizontalB_cpu(x, eHoriz);
  caffe_mul<Dtype>(N_, eHoriz, this->horizontalPotential_->cpu_data(), eHoriz);

  this->timesVerticalB_cpu(x, eVert);
  caffe_mul<Dtype>(N_, eVert, this->verticalPotential_->cpu_data(), eVert);




  Dtype eps = this->smoothnesEps_ * this->smoothnesEps_;
  for(int i = 0; i < N_; ++i) {
    energySmooth += sqrt(eHoriz[i] * eHoriz[i] + eps);
    energySmooth += sqrt(eVert[i] * eVert[i] + eps);
    energyData += this->unitaryPotential_->cpu_data()[i] * y[i];
    energyLogBarrier += log(y[i]) + log(1 - y[i]);
  }
  gradMap = energySmooth + this->dataWeight_->cpu_data()[0] * energyData - this->logBarrierWeight_ * energyLogBarrier;

  computeEnergyGradientPiecewise_cpu(x, eHoriz);
  caffe_copy<Dtype>(N_, x, eVert);
  caffe_axpy<Dtype>(N_, -1, y, eVert);
  gradMap += caffe_cpu_dot<Dtype>(N_, eHoriz, eVert) + 0.5 * L * caffe_cpu_dot<Dtype>(N_, eVert, eVert);

  return gradMap;

}

template<typename Dtype>
Dtype SegmentationEnergy<Dtype>::gradientIteration(Dtype* indicator, Dtype L) const {

  Dtype* argMin = bufferNCOBF2_.mutable_cpu_data();

  this->argMinGradMap(L, indicator, argMin);
  Dtype gradMap = gradMapValue(L, indicator, argMin);
  Dtype energy = energy_cpu(argMin);
  L = L / 2;
  while(energy > gradMap) {
    L = L * 2;
    this->argMinGradMap(L, indicator, argMin);
    gradMap = gradMapValue(L, indicator, argMin);
    energy = energy_cpu(argMin);
  }

  caffe_copy<Dtype>(N_, argMin, indicator);
  return L;
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

    // init stuff
    Dtype L = initLipschnitzConstant_;
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
      while(iter < minimizationIters_) {
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
        argMinGradMap(L, y, t);
        computeEnergyGradient_cpu(t, grad);

        caffe_axpy<Dtype>(N_, -1, y, t);
        gradDiff = -caffe_cpu_dot<Dtype>(N_, t, grad);
        gradConvexEst = caffe_cpu_nrm2<Dtype>(N_, grad);
        gradConvexEst *= gradConvexEst  / L;
      }

      A += ak;
      argMinGradMap(L, y, x);
      L /= gammaD;

      // compute new v
      argMinEstFun(L, ak, x, v);
      ++iter;
//      LOG(INFO) << "Energy at iter #" << iter << " = " << energy_cpu(x);
    }

    LOG(INFO) << "Iter #" << iter << " Minimized energy = " << energy_cpu(x);
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
