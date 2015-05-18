//
// Created by kosiorek on 22/04/15.
//

#include "segmentation_energy.h"
#include "caffe/layer.hpp"

#include <Eigen/Sparse>

namespace caffe {

    /**
     * TODO:
     * 1. implement Nesterov gradient
     * 2. implement tests
     **/


template<typename Dtype>
SegmentationEnergy<Dtype>::SegmentationEnergy(const SegmentationParameter& param, shared_ptr<Blob<Dtype>> dataWeight)
    : dataWeight_(dataWeight) {

    this->logBarrierWeight_ = param.log_barrier_weight();
    this->smoothnesEps_ = param.smoothnes_eps();
    this->stepSize_ = param.step_size();
    this->minimizationIters_ = param.minimization_iters();
    this->minGradNorm_ = param.min_grad_norm();
    this->stepSizeDecay_ = param.step_size_decay();
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
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::setData(Blob <Dtype> *unit, Blob <Dtype> *horizontal, Blob <Dtype> *vertical) {

    this->unitaryPotential_ = unit;
    this->horizontalPotential_ = horizontal;
    this->verticalPotential_ = vertical;
}

template<typename Dtype>
Dtype SegmentationEnergy<Dtype>::energy_cpu(const Dtype*indicator) {

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
void SegmentationEnergy<Dtype>::computeEnergyGradient_cpu(Dtype* indicator,
                                                         Dtype* grad) {
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

    //gradUnary
    caffe_axpy(N_, this->dataWeight_->cpu_data()[0], this->unitaryPotential_->cpu_data(), grad);

    //gradLog
    for (int i = 0; i < N_; ++i) {
        du[i] = 1 / indicator[i] - 1 / (1 - indicator[i]);
    }
    caffe_axpy<Dtype>(N_, -logBarrierWeight_, du, grad);
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::minimizeGD_cpu(Dtype *indicator) {

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
}

template<typename Dtype>
Dtype computeLambda(Dtype oldLambda) {
    return 0.5 * (1 + std::sqrt(1 + 4 * oldLambda * oldLambda));
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::minimizeNAG_cpu(Dtype *indicator) {

    LOG(INFO) << "data: " << this->dataWeight_->cpu_data()[0];
    LOG(INFO) << "unit: " << vec2str(unitaryPotential_->cpu_data());
    LOG(INFO) << "hori: " << vec2str(verticalPotential_->cpu_data());
    LOG(INFO) << "vert: " << vec2str(horizontalPotential_->cpu_data());

    Dtype* grad = bufferMinimization_.mutable_cpu_data();

    computeEnergyGradient_cpu(indicator, grad);

    Dtype oldGradientNorm = std::numeric_limits<Dtype>::max();
    Dtype gradientNorm = caffe_cpu_nrm2<Dtype>(N_, grad);
    Dtype energyOld = std::numeric_limits<Dtype>::max();
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
            oldGradientNorm = gradientNorm;
            gradientNorm = caffe_cpu_nrm2<Dtype>(N_, grad);

            energyOld = energy;
            energy = energy_cpu(indicator);

            if (std::isnan(energy) || gradientNorm < minGradNorm_) {
//            if (std::isnan(energy) || (energy > energyOld) || (gradientNorm > oldGradientNorm)  || gradientNorm < minGradNorm_) {
                break;
            }

            if (iter % 100 == 0) {
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


template<typename Dtype>
void SegmentationEnergy<Dtype>::timesHorizontalB_cpu(const Dtype* in, Dtype* out) {
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
void SegmentationEnergy<Dtype>::timesVerticalB_cpu(const Dtype* in, Dtype* out) {
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
                                                     Dtype* out) {
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
                                                   Dtype* out) {
    caffe_copy<Dtype>(N_ - width_, in, out);
    caffe_set<Dtype>(width_, 0, out + N_ - width_);
    caffe_axpy<Dtype>(N_ - width_, -1, in, out + width_);
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::computeSparseHessian_cpu(const Dtype* indicator) {

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


template<typename Dtype>
void SegmentationEnergy<Dtype>::approxHessVec_cpu(const Dtype* indicator,
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

  //  2nd order
  //f(x + h)
  caffe_copy<Dtype>(N_, indicator, point);
  caffe_axpy<Dtype>(N_, eps, vec, point);
  computeEnergyGradient_cpu(point, grad);
  caffe_axpy<Dtype>(N_, 0.5 / eps, grad, Hv);

  //f(x - h)
  caffe_copy<Dtype>(N_, indicator, point);
  caffe_axpy<Dtype>(N_, -eps, vec, point);
  computeEnergyGradient_cpu(point, grad);

  //Subtract
  caffe_axpy<Dtype>(N_, -0.5 / eps, grad, Hv);
}


template<typename Dtype>
void SegmentationEnergy<Dtype>::sparseHessianMultiply_cpu(const Dtype* vec, Dtype* out) {

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

using namespace Eigen;

template<typename Dtype>
void makeTriplesfromDiag(std::vector<Triplet<Dtype>>& to, const Dtype* from, int xOffset, int yOffset, int N) {

    for(int i = 0; i < N; ++i) {
        to.emplace_back(yOffset + i, xOffset + i, from[i]);
    }
}


template<typename Dtype>
void SegmentationEnergy<Dtype>::invHessianVector_cpu(const Dtype* indicator,
                                                    const Dtype* vec,
                                                    Dtype* iHv) {

    typedef SparseMatrix<Dtype, RowMajor> SparseMatrixType;
    typedef Matrix<Dtype, Dynamic, 1> VectorType;

    SparseMatrixType hessian(N_, N_);
    this->computeSparseHessian_cpu(indicator);

    {
        std::vector<Triplet<Dtype>> triplets;
        triplets.reserve(5 * N_ - 2 * (width_ + 1));

        const Dtype *diagP2 = bufferHessian_.cpu_data();
        const Dtype *diagP1 = diagP2 + N_;
        const Dtype *diag = diagP1 + N_;
        const Dtype *diagM1 = diag + N_;
        const Dtype *diagM2 = diagM1 + N_;

        makeTriplesfromDiag(triplets, diag, 0, 0, N_);
        makeTriplesfromDiag(triplets, diagP1, 1, 0, N_ - 1);
        makeTriplesfromDiag(triplets, diagM1 + 1, 0, 1, N_ - 1);
        makeTriplesfromDiag(triplets, diagP2, width_, 0, N_ - width_);
        makeTriplesfromDiag(triplets, diagM2, 0, width_, N_ - width_);
        hessian.setFromTriplets(triplets.begin(), triplets.end());
    }

    SparseLU<SparseMatrixType> solver;
    solver.analyzePattern(hessian);
    solver.factorize(hessian);

    Map<VectorType> vector(const_cast<Dtype*>(vec), N_, 1);
    Map<VectorType> result(iHv, N_, 1);

    result = solver.solve(vector);
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::charbonnierD1_cpu(const Dtype* in,
                                                 Dtype* out) {

    Dtype eps = this->smoothnesEps_ * this->smoothnesEps_;
    for (int i = 0; i < N_; ++i) {
        out[i] = in[i] / sqrt(in[i] * in[i] + eps);
    }
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::charbonnierD2_cpu(const Dtype* in,
                                                 Dtype* out) {

    Dtype eps = this->smoothnesEps_ * this->smoothnesEps_;
    for (int i = 0; i < N_; ++i) {
        Dtype denom = in[i] * in[i] + eps;
        out[i] = eps / (denom * sqrt(denom));
    }
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::zeroLastRow_cpu(Dtype* v) {
    caffe_set<Dtype>(width_, 0, v + N_ - width_);
}

template<typename Dtype>
void SegmentationEnergy<Dtype>::zeroLastColumn_cpu(Dtype* v) {
    for (int i = width_ - 1; i < N_; i += width_) {
        v[i] = 0;
    }
}

INSTANTIATE_CLASS(SegmentationEnergy);

}  // namespace caffe
























//    //  4th order derivative approximation by Finite Differences
//    //f(x + 2h)
//    caffe_copy<Dtype>(N_, indicator, point);
//    caffe_axpy<Dtype>(N_, 2*eps, vec, point);
//    computeEnergyGradient_cpu(point, grad);
//    caffe_axpy<Dtype>(N_, -1 / (12*eps), grad, Hv);
//
//    //f(x - 2h)
//    caffe_copy<Dtype>(N_, indicator, point);
//    caffe_axpy<Dtype>(N_, -2*eps, vec, point);
//    computeEnergyGradient_cpu(point, grad);
//    caffe_axpy<Dtype>(N_, 1 / (12*eps), grad, Hv);
//
//    //f(x + h)
//    caffe_copy<Dtype>(N_, indicator, point);
//    caffe_axpy<Dtype>(N_, eps, vec, point);
//    computeEnergyGradient_cpu(point, grad);
//    caffe_axpy<Dtype>(N_, 2 / (3*eps), grad, Hv);
//
//    //f(x - 2h)
//    caffe_copy<Dtype>(N_, indicator, point);
//    caffe_axpy<Dtype>(N_, -eps, vec, point);
//    computeEnergyGradient_cpu(point, grad);
//    caffe_axpy<Dtype>(N_, -2 / (3*eps), grad, Hv);










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
