//
// Created by kosiorek on 22/04/15.
//

#include "segmentation_energy.h"
#include "caffe/layer.hpp"
#include <cmath>

#ifdef DBG

#define LOG_FUN_START LOG(INFO) << "Starting " << __FUNCTION__
#define LOG_FUN_END LOG(INFO) << "Finishing " << __FUNCTION__

#else

#define LOG_FUN_START
#define LOG_FUN_END

#endif

using namespace Eigen;

namespace caffe {

template<typename Dtype>
SegmentationEnergy<Dtype>::SegmentationEnergy(const SegmentationParameter& param, shared_ptr<Blob<Dtype>> dataWeight)
    : dataWeight_(dataWeight) {

    this->logBarrierWeight_ = param.log_barrier_weight();
    this->smoothnesEps_ = param.smoothnes_eps();
    this->stepSize_ = param.step_size();
    this->minimizationIters_ = param.minimization_iters();
    this->minGradNorm_ = param.min_grad_norm();
    this->stepSizeDecay_ = param.step_size_decay();

    this->convexParam_ = param.convex_param();
    this->initLipschnitzConstant_ = param.init_lipschitz_constant();
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::reshape(int width, int height) {

    this->width_ = width;
    this->height_ = height;
    this->N_ = width * height;

    bufferEnergyGrad_.Reshape(1, 1, height_, width_);
    bufferMinimization_.Reshape(1, 1, height_, width_);
    bufferResidualDirection_.Reshape(1, 1, height_, width_);
    bufferMatVecStorage_.Reshape(1, 1, height_, width_);
    bufferHessVec_.Reshape(1, 1, height_, width_);
    bufferHessian_.Reshape(5, 1, height_, width_);
    bufferNAG_.Reshape(1, 1, height_, width_);

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
void SegmentationEnergy<Dtype>::minimizeGD_cpu(Dtype *indicator) const {

    LOG(INFO) << "Starting GD!";
    Dtype* grad = bufferMinimization_.mutable_cpu_data();

//    LOG(INFO) << "data: " << this->dataWeight_->cpu_data()[0];
//    LOG(INFO) << "unit: " << vec2str(unitaryPotential_->cpu_data());
//    LOG(INFO) << "hori: " << vec2str(verticalPotential_->cpu_data());
//    LOG(INFO) << "vert: " << vec2str(horizontalPotential_->cpu_data());
//        LOG(INFO) << "indicator: " << vec2str(indicator);

        computeEnergyGradient_cpu(indicator, grad);

        Dtype oldGradientNorm = std::numeric_limits<Dtype>::max();
        Dtype gradientNorm = caffe_cpu_nrm2<Dtype>(N_, grad);
        Dtype energyOld = std::numeric_limits<Dtype>::max();
        Dtype energy = energy_cpu(indicator);
        int iter = 0;

//        LOG(INFO) << "Energy at iter #" << iter << " = " << energy;
        while (iter++ < minimizationIters_) {
            caffe_axpy<Dtype>(N_, -stepSize_, grad, indicator);
            computeEnergyGradient_cpu(indicator, grad);

            if (iter % 10 == 0) {
                oldGradientNorm = gradientNorm;
                gradientNorm = caffe_cpu_nrm2<Dtype>(N_, grad);

                energyOld = energy;
                energy = energy_cpu(indicator);

                if (std::isnan(energy) || (energy > energyOld) || (gradientNorm > oldGradientNorm) || gradientNorm < minGradNorm_) {
                    break;
                }

                if (iter % 100 == 0) {
                    LOG(INFO) << "Energy at #iter: " << iter << " = " << energy << "\tGradientNorm = " <<
                    gradientNorm;
                }
            }
        }
//    LOG(INFO) << "indicator: "cd << vec2str(indicator);

    if (std::isnan(energy) || std::isinf(energy)) {
        LOG(FATAL) << "Energy is Nan. Terminating";
    }

  LOG(INFO) << "Minimized energy = " << energy;
}

template<typename Dtype>
Dtype computeLambda(Dtype oldLambda) {
    return 0.5 * (1 + std::sqrt(1 + 4 * oldLambda * oldLambda));
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::minimizeNAG_cpu(Dtype *indicator) const {

    LOG(INFO) << "data: " << this->dataWeight_->cpu_data()[0];
    LOG(INFO) << "unit: " << vec2str(unitaryPotential_->cpu_data());
    LOG(INFO) << "hori: " << vec2str(verticalPotential_->cpu_data());
    LOG(INFO) << "vert: " << vec2str(horizontalPotential_->cpu_data());

    Dtype* grad = bufferMinimization_.mutable_cpu_data();

    computeEnergyGradient_cpu(indicator, grad);

    Dtype minGradientNorm = std::numeric_limits<Dtype>::max();
    Dtype gradientNorm = caffe_cpu_nrm2<Dtype>(N_, grad);
    Dtype minEnergy = std::numeric_limits<Dtype>::max();
    Dtype energy = energy_cpu(indicator);
    int iter = 0;

    // initialize algorithm params
    Dtype lambdaOld = 0;
    Dtype lambdaNew = computeLambda(lambdaOld);
    Dtype gamma = 0;
    Dtype* yOld = bufferNAG_.mutable_cpu_data();
    Dtype* yNew = bufferNAG_.mutable_cpu_data();

    caffe_set<Dtype>(N_, 0, yOld);
    caffe_copy<Dtype>(N_, indicator, yNew);

    Dtype stepSize = this->stepSize_;

    LOG(INFO) << "Energy at iter #" << iter << " = " << energy << " gradientNorm = " << gradientNorm;
    while (iter++ < minimizationIters_) {
//        caffe_axpy<Dtype>(N_, -stepSize_, grad, indicator);
//        computeEnergyGradient_cpu(indicator, grad);

        lambdaOld = lambdaNew;
        lambdaNew = computeLambda(lambdaOld);
        gamma = (1 - lambdaOld) / lambdaNew;

        // yOld = yNew
        caffe_copy<Dtype>(N_, yOld, yNew);

        // yNew = indicator - stepSize * grad(indicator)
        computeEnergyGradient_cpu(indicator, grad);
        caffe_copy<Dtype>(N_, indicator, yNew);
        caffe_axpy<Dtype>(N_, -stepSize, grad, yNew);

        // indicator = (1 - gamma) * yNew + gamma * yOld
        caffe_copy<Dtype>(N_, yNew, indicator);
        caffe_scal<Dtype>(N_, 1 - gamma, indicator);
        caffe_axpy<Dtype>(N_, gamma, yOld, indicator);

        if (iter % 10 == 0) {
            minGradientNorm = std::min(gradientNorm, minGradientNorm);
            gradientNorm = caffe_cpu_nrm2<Dtype>(N_, grad);

            minEnergy = std::min(energy, minEnergy);
            energy = energy_cpu(indicator);

//            if (std::isnan(energy) || gradientNorm < minGradNorm_) {
//            if (std::isnan(energy) || (energy > energyOld) || (gradientNorm > oldGradientNorm)  || gradientNorm < minGradNorm_) {
//            if (std::isnan(energy) || (gradientNorm > 1.2 * minGradientNorm)  || gradientNorm < minGradNorm_) {
            if (std::isnan(energy) || (energy > 1.1 * minEnergy)  || gradientNorm < minGradNorm_) {
                break;
            }

            if (iter % 500 == 0) {
                LOG(INFO) << "Energy at #iter: " << iter << " = " << energy << "\tGradientNorm = " <<
                gradientNorm;
            }

            if(iter % 1000 == 0) {
              stepSize *= stepSizeDecay_;
            }
        }
    }
//    LOG(INFO) << "indicator: "cd << vec2str(indicator);

    if (std::isnan(energy) || std::isinf(energy)) {
        LOG(FATAL) << "Energy is Nan. Terminating";
    }
    LOG(INFO) << "Terminating at #iter: " << iter << " Energy = " << energy << "\tGradientNorm = " << gradientNorm;
//    LOG(INFO) << "indicator: " << vec2str(indicator);
    LOG(INFO) << "indicator max = " << *std::max_element(indicator, indicator + N_) << " min = " << *std::min_element(indicator, indicator + N_);
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
//    LOG_FUN_START;

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
    LOG_FUN_START;

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
            LOG(WARNING) << "Only imaginary or invalid roots!\n"  \
                   << "iter: " << i << "\n" \
                   << "coeffs: " << a[i] << " " << b[i] << " " << c[i] << " " << d[i] << "\n" \
                   << "roots: " << roots[0] << " ," << roots[1] << " ," << roots[2];
            return false;
        }

    }
    return true;
}

template<typename Dtype>
bool SegmentationEnergy<Dtype>::argMinGrapMap(Dtype L, const Dtype* y, Dtype* argMin) const {
    LOG_FUN_START;

    Dtype* a = bufferArgMinGrapMap_.mutable_cpu_data();
    Dtype* b = a + N_;
    Dtype* c = b + N_;
    Dtype* d = c + N_;

    // save piecewiese energy gradient in c
    computeEnergyGradientPiecewise_cpu(y, c);

    // a
    caffe_set<Dtype>(N_, -L, a);
    caffe_axpy<Dtype>(N_, -dataWeight_->cpu_data()[0], unitaryPotential_->cpu_data(), a);

    // b
    caffe_set<Dtype>(N_, L, b);
    caffe_axpy<Dtype>(N_, L, y, b);
    caffe_axpy<Dtype>(N_, dataWeight_->cpu_data()[0], unitaryPotential_->cpu_data(), b);
    caffe_axpy<Dtype>(N_, -1, c, b);

    // c
    caffe_axpy<Dtype>(N_, -L, y, c);
    caffe_add_scalar<Dtype>(N_, 2 * logBarrierWeight_, c);

    // d
    caffe_set<Dtype>(N_, -logBarrierWeight_, d);

    return getValidRoots(N_, a, b, c, d, argMin);
};

template<typename Dtype>
bool SegmentationEnergy<Dtype>::argMinEstFun(Dtype L, Dtype ak, const Dtype* y, Dtype* argMin) const {
    LOG_FUN_START;

    Dtype* a = bufferArgMinEstFuns_.mutable_cpu_data();
    Dtype* b = a + N_;
    Dtype* c = b + N_;
    Dtype* d = c + N_;

    // update coefficients

    // a
    caffe_axpy<Dtype>(N_, -dataWeight_->cpu_data()[0] * ak, unitaryPotential_->cpu_data(), a);

    // b
    computeEnergyGradientPiecewise_cpu(y, argMin);
    caffe_axpy<Dtype>(N_, -ak, argMin, b);
    caffe_axpy<Dtype>(N_, dataWeight_->cpu_data()[0] * ak, unitaryPotential_->cpu_data(), b);

    // c
    caffe_add_scalar<Dtype>(N_, 2 * logBarrierWeight_ * ak, c);

    // d
    caffe_add_scalar<Dtype>(N_, -logBarrierWeight_ * ak, d);

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

  this->argMinGrapMap(L, indicator, argMin);
  Dtype gradMap = gradMapValue(L, indicator, argMin);
  Dtype energy = energy_cpu(argMin);
  L = L / 2;
  while(energy > gradMap) {
    L = L * 2;
    this->argMinGrapMap(L, indicator, argMin);
    gradMap = gradMapValue(L, indicator, argMin);
    energy = energy_cpu(argMin);
  }

  caffe_copy<Dtype>(N_, argMin, indicator);
  return L;
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::minimizeGradientMethod_cpu(Dtype* indicator) const {
  LOG_FUN_START;

    Dtype L = this->initLipschnitzConstant_;
    Dtype delta = std::numeric_limits<Dtype>::max();
    Dtype oldEnergy = std::numeric_limits<Dtype>::max();
    Dtype energy = this->energy_cpu(indicator);
    Dtype* oldIndicator = bufferNCOBF1_.mutable_cpu_data();
    Dtype* diffIndicator = bufferNCOBF1_.mutable_cpu_diff();
//  Reshape(1, 1, height_, width_);

  int iter = 0;
  LOG(INFO) << "Energy at iter #" << iter << " = " << energy;
  while(delta > 1e-6) {// && energy < oldEnergy) {
    caffe_copy<Dtype>(N_, indicator, oldIndicator);
    oldEnergy = energy;

    Dtype M = gradientIteration(indicator, L);

    // compute delta
    caffe_copy<Dtype>(N_, indicator, diffIndicator);
    caffe_axpy<Dtype>(N_, -1, oldIndicator, diffIndicator);
    delta = caffe_cpu_nrm2<Dtype>(N_, diffIndicator);

    energy = this->energy_cpu(indicator);

    L = std::max(this->initLipschnitzConstant_, M / 2);
    ++iter;
    LOG(INFO) << "Energy at iter #" << iter << " = " << energy << " delta = " << delta;
  }
//  LOG(INFO) << "Energy at iter #" << iter << " = " << energy;


  LOG_FUN_END;
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::minimizeDualGradientMethod_cpu(Dtype* indicator) const {
  LOG_FUN_START;

  LOG_FUN_END;
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::minimizeNCOBF_cpu(Dtype* x) const {
    LOG_FUN_START;

    static const Dtype gammaU = 2;  // Lipschitz constant scaling parameters
    static const Dtype gammaD = 2;  // Author mentioned that gammaU = gammaD is a reasonable choice

    Dtype* v = bufferNCOBF1_.mutable_cpu_data();
    Dtype* y = bufferNCOBF1_.mutable_cpu_diff();
    Dtype* t = bufferNCOBF2_.mutable_cpu_data();
    Dtype* grad = bufferNCOBF2_.mutable_cpu_diff();
    Dtype* bestX = bufferMinimization_.mutable_cpu_data();

    // init stuff
    Dtype L = initLipschnitzConstant_;
    Dtype A = 0;
    Dtype ak = 0;
    caffe_copy<Dtype>(N_, x, v);

    // intit coeffs for computing v
    {
        Dtype *a = bufferArgMinEstFuns_.mutable_cpu_data();
        Dtype *b = a + N_;
        Dtype *c = b + N_;
        Dtype *d = c + N_;

        caffe_set<Dtype>(N_, -1, a);
        caffe_set<Dtype>(N_, 1, b);
        caffe_axpy<Dtype>(N_, 1, x, b);
        caffe_set<Dtype>(N_, 0, c);
        caffe_set<Dtype>(N_, 0, d);
    }
    Dtype newEnergy = energy_cpu(x);
    Dtype firstEnergy = newEnergy;
    Dtype bestEnergy = newEnergy;
    caffe_copy<Dtype>(N_, x, bestX);

    int maxRepeats = 5;
    int repeat = 0;
    int iter = 0;
    LOG(INFO) << "Energy at iter #" << iter << " = " << newEnergy;
//    LOG(INFO) << "L = " << L;
//    for(; iter < minimizationIters_; ++iter) {
     while(newEnergy < bestEnergy || repeat < maxRepeats) {

        Dtype gradDiff = 0;
        Dtype gradConvexEst = std::numeric_limits<Dtype>::max();
        bool validRoots = true;
        while(gradDiff < gradConvexEst) {
            Dtype k = (1 + convexParam_ * A) / L;
            ak = 1/L + std::sqrt(k * (k + 2));

            // compute y
            caffe_cpu_scale<Dtype>(N_, A / (A + ak), x, y);
            caffe_axpy<Dtype>(N_, ak / (A + ak), v, y);

            // grad at gradMapMinimizer
            validRoots = argMinGrapMap(L, y, t);
            if(!validRoots) {
                break;
            }
            computeEnergyGradient_cpu(t, grad);

            caffe_axpy<Dtype>(N_, -1, y, t);
            gradDiff = -caffe_cpu_dot<Dtype>(N_, t, grad);
            gradConvexEst = caffe_cpu_nrm2<Dtype>(N_, grad);
            gradConvexEst *= gradConvexEst  / L;
            if(gradDiff < gradConvexEst) {
                L *= gammaU;
            }

        }

         if(!validRoots) {
             break;
         }

        A += ak;
        if(!argMinGrapMap(L, y, x)) {
            break;
        }

        L /= gammaD;
        LOG(INFO) << "L = " << L;

        LOG(INFO) << "Energy at iter #" << iter++ << " = " << energy_cpu(x);
        // compute new v
        if(!argMinEstFun(L, ak, x, v)) {
            break;
        }


         newEnergy = energy_cpu(x);
         if(newEnergy < bestEnergy) {
             repeat = 0;
             bestEnergy = newEnergy;
             caffe_copy<Dtype>(N_, x, bestX);
         } else {
             ++repeat;
         }
    }

    caffe_copy<Dtype>(N_, bestX, x);
    LOG(INFO) << "Minimized energy = " << bestEnergy << " from " << firstEnergy;

  LOG_FUN_END;
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::minimizeNewton(Dtype* indicator) const {

  Dtype* update = bufferNCOBF1_.mutable_cpu_data();
  Dtype* grad = bufferNCOBF1_.mutable_cpu_diff();

  Dtype updateNorm = std::numeric_limits<Dtype>::max();

  int iter = 0;
  LOG(ERROR) << "energy at iter: " << iter << " = " << this->energy_cpu(indicator) << " update norm = " <<
                                                                                      updateNorm;
  while(updateNorm > 1e-6 && iter < 10000) {
    this->computeEnergyGradient_cpu(indicator, grad);
    this->invHessianVector_cpu(indicator, grad, update);
    updateNorm = caffe_cpu_nrm2(N_, update);

    caffe_axpy<Dtype>(N_, -0.00001, update, indicator);
    ++iter;
    LOG(ERROR) << "energy at iter: " << iter << " = " << this->energy_cpu(indicator) << " update norm = " <<
                                                                                        updateNorm;
  }
  LOG(ERROR) << "energy at iter: " << iter << " = " << this->energy_cpu(indicator) << " update norm = " <<
  updateNorm;


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

    Dtype* du = bufferHessian_.mutable_cpu_diff();

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
    for(int i = 0; i < N_; ++i) {
        Dtype u = indicator[i];
        u = 1 / (u * u) + 1 / ((1 - u) * (1 - u));
        diag[i] += this->logBarrierWeight_ * u;
    }
}


//template<typename Dtype>
//void SegmentationEnergy<Dtype>::approxHessVec_cpu(const Dtype* indicator,
//                                                  const Dtype* vec,
//                                                  Dtype* Hv) const {
//  Dtype maxElem = fabs(vec[0]);
//  for (int i = 1; i < N_; ++i) {
//    if (fabs(vec[i]) > maxElem) {
//      maxElem = fabs(vec[i]);
//    }
//  }
//
//  Dtype eps = fmax(1e-6, fmin(1, 1e-3 / maxElem));
//  Dtype* point = bufferHessVec_.mutable_cpu_data();
//  Dtype* grad = bufferHessVec_.mutable_cpu_diff();
//
//  caffe_set<Dtype>(N_, 0, Hv);
//
////  //  2nd order
////  //f(x + h)
////  caffe_copy<Dtype>(N_, indicator, point);
////  caffe_axpy<Dtype>(N_, eps, vec, point);
////  computeEnergyGradient_cpu(point, grad);
////  caffe_axpy<Dtype>(N_, 0.5 / eps, grad, Hv);
////
////  //f(x - h)
////  caffe_copy<Dtype>(N_, indicator, point);
////  caffe_axpy<Dtype>(N_, -eps, vec, point);
////  computeEnergyGradient_cpu(point, grad);
////
////  //Subtract
////  caffe_axpy<Dtype>(N_, -0.5 / eps, grad, Hv);
//
//
//      //  4th order derivative approximation by Finite Differences
//      //f(x + 2h)
//      caffe_copy<Dtype>(N_, indicator, point);
//      caffe_axpy<Dtype>(N_, 2*eps, vec, point);
//      computeEnergyGradient_cpu(point, grad);
//      caffe_axpy<Dtype>(N_, -1 / (12*eps), grad, Hv);
//
//      //f(x - 2h)
//      caffe_copy<Dtype>(N_, indicator, point);
//      caffe_axpy<Dtype>(N_, -2*eps, vec, point);
//      computeEnergyGradient_cpu(point, grad);
//      caffe_axpy<Dtype>(N_, 1 / (12*eps), grad, Hv);
//
//      //f(x + h)
//      caffe_copy<Dtype>(N_, indicator, point);
//      caffe_axpy<Dtype>(N_, eps, vec, point);
//      computeEnergyGradient_cpu(point, grad);
//      caffe_axpy<Dtype>(N_, 2 / (3*eps), grad, Hv);
//
//      //f(x - 2h)
//      caffe_copy<Dtype>(N_, indicator, point);
//      caffe_axpy<Dtype>(N_, -eps, vec, point);
//      computeEnergyGradient_cpu(point, grad);
//      caffe_axpy<Dtype>(N_, -2 / (3*eps), grad, Hv);
//
//}


//template<typename Dtype>
//void SegmentationEnergy<Dtype>::sparseHessianMultiply_cpu(const Dtype* vec, Dtype* out) const {
//
//    // super diagonals
//    const Dtype* diagP2 = bufferHessian_.cpu_data();
//    const Dtype* diagP1 = diagP2 + N_;
//
//    // diag
//    const Dtype* diag = diagP1+ N_;
//
//    // sub diagonals
//    const Dtype* diagM1 = diag + N_;
//    const Dtype* diagM2 = diagM1 + N_;
//
//    out[0] = diag[0] * vec[0] + diagP1[0] * vec[1] + diagP2[0] * vec[width_];
//    for(int i = 1; i < width_; ++i) {
//        out[i] = diagM1[i] * vec[i - 1] + diag[i] * vec[i] + diagP1[i] * vec[i + 1] + diagP2[i] * vec[i + width_];
//    }
//    for(int i = width_; i < N_ - width_; ++i) {
//        out[i] = diagM2[i] * vec[i - width_] + diagM1[i] * vec[i - 1] + diag[i] * vec[i] + diagP1[i] * vec[i + 1] + diagP2[i] * vec[i + width_];
//    }
//    for(int i = N_ - width_; i < N_ - 1; ++i) {
//        out[i] = diagM2[i] * vec[i - width_] + diagM1[i] * vec[i - 1] + diag[i] * vec[i] + diagP1[i] * vec[i + 1];
//    }
//
//    out[N_ - 1] = diagM2[N_ - 1] * vec[N_ - width_ - 1] + diagM1[N_ - 1] * vec[N_ - 2] + diag[N_ - 1] * vec[N_ - 1];
//
//
////  typedef Matrix<Dtype, Dynamic, 1> VectorType;
////  auto hessian = convertHessianToEigenSparse();
////
////  MatrixXd dense(hessian);
////  LOG(ERROR) << "\n\n" << dense << "\n\n";
////
////  const Map<const VectorType> vector(vec, N_, 1);
////  Map<VectorType> result(out, N_, 1);
////  result = hessian * vector;
//
//}

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


































///**
// * Conjugate Gradient
// */
//template<typename Dtype>
//void SegmentationEnergy<Dtype>::invHessianVector_cpu(const Dtype* indicator,
//                                                    const Dtype* vec,
//                                                    Dtype* iHv) {
//
//    Dtype* residual = bufferResidualDirection_.mutable_cpu_data();
//    Dtype* direction = bufferResidualDirection_.mutable_cpu_diff();
//    Dtype* matVecProd = bufferMatVecStorage_.mutable_cpu_data();
//    Dtype* newDirectionBuffer = bufferMatVecStorage_.mutable_cpu_diff();
//
//    computeSparseHessian_cpu(indicator);
//
//    //initialize iHv
//    caffe_set<Dtype>(N_, 1.0, iHv);
//
//
//
////  approxHessVec_cpu(indicator, iHv, matVecProd);
//
//// residual = b - Ax
//    caffe_set<Dtype>(N_, 0, residual);
//    caffe_axpy<Dtype>(N_, -1, vec, residual);
//    sparseHessianMultiply_cpu(iHv, matVecProd);
//    caffe_axpy<Dtype>(N_, 1, matVecProd, residual);
//
////  caffe_copy<Dtype>(N_, vec, residual);
////  sparseHessianMultiply_cpu(iHv, matVecProd);
////  caffe_axpy<Dtype>(N_, -1, matVecProd, residual);
//
//    //direction
//    caffe_copy<Dtype>(N_, residual, direction);
//
//    Dtype residualNorm = caffe_cpu_nrm2<Dtype>(N_, residual);
//    Dtype residualNormOld = 0;
//    int iter = 0;
//    while (iter++ < invHessIters_) {
//        LOG(INFO) << "ResidualNorm at #iter = " << iter-1 << " = " << residualNorm;
//
////    approxHessVec_cpu(indicator, direction, matVecProd);
//        sparseHessianMultiply_cpu(direction, matVecProd);
//
//        Dtype alpha = residualNorm * residualNorm
//                      / caffe_cpu_dot<Dtype>(N_, direction, matVecProd);
//
//        //update iHv
//        caffe_axpy<Dtype>(N_, alpha, direction, iHv);
//
//        // new residual
//        residualNormOld = residualNorm;
//        caffe_axpy<Dtype>(N_, -alpha, matVecProd, residual);
//        residualNorm = caffe_cpu_nrm2<Dtype>(N_, residual);
//
//        if (residualNorm < invHessTolerance_) {
//            LOG(INFO) << "Terminated at #iter = " << iter << " with ResidualNorm = " << residualNorm;
//            break;
//        }
//
//        // new direction
//        Dtype beta = -residualNorm * residualNorm / (residualNormOld * residualNormOld);
//        caffe_copy<Dtype>(N_, residual, newDirectionBuffer);
//        caffe_axpy<Dtype>(N_, -beta, direction, newDirectionBuffer);
//        caffe_copy<Dtype>(N_, newDirectionBuffer, direction);
//    }
//}
