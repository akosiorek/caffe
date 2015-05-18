//
// Created by kosiorek on 22/04/15.
//

#ifndef CAFFE_SEGMENTATION_ENERGY_H
#define CAFFE_SEGMENTATION_ENERGY_H

#include "caffe/common.hpp"
#include "caffe/blob.hpp"

namespace caffe {

template<typename Dtype>
class SegmentationEnergy {
public:

    SegmentationEnergy(const SegmentationParameter& param, shared_ptr<Blob<Dtype>> dataWeight);
    void reshape(int width, int height);
    void setData(Blob<Dtype>* unit, Blob<Dtype>* horizontal, Blob<Dtype>* vertical);

//  CPU ========================================================================

    Dtype energy_cpu(const Dtype* indicatorValue);
    void computeEnergyGradient_cpu(Dtype* indicatorValue, Dtype* indicatorGrad);
    // Gradient Descent
    void minimizeGD_cpu(Dtype *indicator);
    // Nesterov's Accelerated Gradient
    void minimizeNAG_cpu(Dtype *indicator);
    void timesHorizontalB_cpu(const Dtype* f, Dtype* dx);
    void timesVerticalB_cpu(const Dtype* f, Dtype* dy);
    void timesHorizontalBt_cpu(const Dtype* grad, Dtype* diffDx);
    void timesVerticalBt_cpu(const Dtype* grad, Dtype* diffDy);

    void computeSparseHessian_cpu(const Dtype* indicator);
    void sparseHessianMultiply_cpu(const Dtype* vec, Dtype* out);
    void approxHessVec_cpu(const Dtype* indicator, const Dtype* vec, Dtype* Hv);
    void invHessianVector_cpu(const Dtype* indicator, const Dtype* vec,
                              Dtype* iHv);

    void charbonnierD1_cpu(const Dtype* source, Dtype* dest);
    void charbonnierD2_cpu(const Dtype* source, Dtype* dest);
    void zeroLastRow_cpu(Dtype* v);
    void zeroLastColumn_cpu(Dtype* v);

//  GPU ========================================================================

//  Params ========================================================================

    int width_;
    int height_;
    int N_;

    Dtype stepSize_;
    int minimizationIters_;
    Dtype logBarrierWeight_;
    Dtype smoothnesEps_;
    Dtype minGradNorm_;
    Dtype stepSizeDecay_;

    shared_ptr<Blob<Dtype>> dataWeight_;
    Blob<Dtype>* unitaryPotential_;
    Blob<Dtype>* horizontalPotential_;
    Blob<Dtype>* verticalPotential_;

    Blob<Dtype> bufferEnergyGrad_;
    Blob<Dtype> bufferMinimization_;
    Blob<Dtype> bufferResidualDirection_;
    Blob<Dtype> bufferMatVecStorage_;
    Blob<Dtype> bufferHessVec_;
    Blob<Dtype> bufferHessian_;
    Blob<Dtype> bufferNAG_;
};

#include <sstream>
template<typename Dtype>
std::string vec2str(const Dtype* v) {
    std::stringstream ss;
    for(int i = 0; i < 9; ++i) {
        ss << v[i] << " ";
    }
    return ss.str();
}

}  // namespace caffe

#endif //CAFFE_SEGMENTATION_ENERGY_H
