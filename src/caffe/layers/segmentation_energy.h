//
// Created by kosiorek on 22/04/15.
//

#ifndef CAFFE_SEGMENTATION_ENERGY_H
#define CAFFE_SEGMENTATION_ENERGY_H

#include <Eigen/Sparse>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"

namespace caffe {

template<typename Dtype>
class SegmentationEnergy {
public:
  typedef Eigen::SparseMatrix<Dtype, Eigen::RowMajor> SparseMatrixT;
public:

  SegmentationEnergy(const SegmentationParameter &param,
                     shared_ptr<Blob<Dtype>> dataWeight);

  void reshape(int width, int height);

  void setData(const Blob<Dtype> *unit, const Blob<Dtype> *horizontal,
               const Blob<Dtype> *vertical);

//  CPU ========================================================================

  Dtype energy_cpu(const Dtype *indicatorValue) const;

  void computeEnergyGradient_cpu(const Dtype *indicator, Dtype *grad) const;

  void computeEnergyGradientPiecewise_cpu(const Dtype *indicator,
                                          Dtype *grad) const;

  Dtype gradientIteration(Dtype *indicator, Dtype L) const;

  // Nesterov's accelerated method for composite objective functions
  void minimizeNCOBF_cpu(Dtype *indicator) const;

  Dtype gradMapValue(Dtype L, const Dtype *x, const Dtype *y) const;

  bool argMinGradMap(Dtype L, const Dtype *y, Dtype *argMin) const;

  bool argMinEstFun(Dtype L, Dtype ak, const Dtype *v, Dtype *argMin) const;

  Dtype cubicRoot(Dtype x) const;

  std::array <std::complex<Dtype>, 3> cubicRoots(Dtype a, Dtype b, Dtype c,
                                                 Dtype d) const;

  bool getValidRoots(int N, Dtype *a, Dtype *b, Dtype *c, Dtype *d,
                     Dtype *result) const;

  void timesHorizontalB_cpu(const Dtype *f, Dtype *dx) const;

  void timesVerticalB_cpu(const Dtype *f, Dtype *dy) const;

  void timesHorizontalBt_cpu(const Dtype *grad, Dtype *diffDx) const;

  void timesVerticalBt_cpu(const Dtype *grad, Dtype *diffDy) const;

  void computeSparseHessian_cpu(const Dtype *indicator) const;

  void invHessianVector_cpu(const Dtype *indicator, const Dtype *vec,
                            Dtype *iHv) const;

  void charbonnierD1_cpu(const Dtype *source, Dtype *dest) const;

  void charbonnierD2_cpu(const Dtype *source, Dtype *dest) const;

  void zeroLastRow_cpu(Dtype *v) const;

  void zeroLastColumn_cpu(Dtype *v) const;

  void makeTriplesfromDiag(std::vector<Eigen::Triplet<Dtype>> &to,
                           const Dtype *from, int xOffset, int yOffset,
                           int N) const;

  SparseMatrixT convertHessianToEigenSparse() const;


//  GPU ========================================================================

//  Params ========================================================================

  int width_;
  int height_;
  int N_;

  int minimizationIters_;
  Dtype logBarrierWeight_;
  Dtype smoothnesEps_;
  Dtype minUpdateNorm_;

  shared_ptr<Blob<Dtype>> dataWeight_;
  const Blob<Dtype> *unitaryPotential_;
  const Blob<Dtype> *horizontalPotential_;
  const Blob<Dtype> *verticalPotential_;

  mutable Blob<Dtype> bufferEnergyGrad_;
  mutable Blob<Dtype> bufferMinimization_;

  mutable Blob<Dtype> bufferHessian_;

  mutable Blob<Dtype> bufferArgMinGrapMap_;
  mutable Blob<Dtype> bufferArgMinEstFuns_;
  mutable Blob<Dtype> bufferNCOBF1_;
  mutable Blob<Dtype> bufferNCOBF2_;

  Dtype convexParam_;
  Dtype initLipschnitzConstant_;
};

}  // namespace caffe

#endif //CAFFE_SEGMENTATION_ENERGY_H
