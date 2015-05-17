#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>
#include <cmath>
#include <memory>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/layers/segmentation_energy.h"

#define ASSERT_NEAR_VEC_WTOL(computed, expected, count, tol)   for(int i = 0; i < count; ++i) ASSERT_NEAR((computed)[i], (expected)[i], fmax((expected)[i] * tol, tol)) << "i = " << i
#define ASSERT_NEAR_VEC(computed, expected, count) ASSERT_NEAR_VEC_WTOL(computed, expected, count, this->tolerance)

namespace caffe {

template<typename T>
struct tolerance_trait {
  static const constexpr double tol = 1e-8;
};

template<>
struct tolerance_trait<float> {
  static const constexpr float tol = 1e-6;
};


template <typename Dtype>
class SegmentationenergyTest : public ::testing::Test {
 protected:
  SegmentationenergyTest()
 	 : tolerance(tolerance_trait<Dtype>::tol),
 	   //magic(3)/10
 	   indicator({ 0.8000,    0.1000,    0.6000,
                0.3000,    0.5000,    0.7000,
                0.4000,    0.9000,    0.2000}),
     output(9),
 	   potentialVec({&unitPotential, &horizontalPotential, &verticalPotential}),
     dataWeight(new Blob<Dtype>(1, 1, 1, 1)) {

    this->reshape(1, 1, 3, 3);
    this->setPotentials(0.5);

	  segmentationParam.set_step_size(0.001);
	  segmentationParam.set_minimization_iters(10);
	  segmentationParam.set_init_data_weight(1);
	  segmentationParam.set_log_barrier_weight(1);
	  segmentationParam.set_smoothnes_eps(1e-3);

	  // init data weight
	  *dataWeight->mutable_cpu_data() = 1;

	  energy = std::move(std::unique_ptr<SegmentationEnergy<Dtype>>(new SegmentationEnergy<Dtype>(segmentationParam, dataWeight)));
	  energy->reshape(3, 3);
	  energy->setData(&unitPotential, &horizontalPotential, &verticalPotential);
  }

  void reshape(int num, int c, int w, int h) {
    for(int i = 0; i < potentialVec.size(); ++i) {
       potentialVec[i]->Reshape(num, c, w, h);
    }
  }

  void setPotentials(Dtype value) {
	  for(int i = 0; i < potentialVec.size(); ++i) {
	    caffe_set<Dtype>(9, value, potentialVec[i]->mutable_cpu_data());
	  }
  }

  void printVec(const Dtype* vec) {
    for(int i = 0; i < this->unitPotential.count(); ++i) {
      std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
  }


public:
  Dtype tolerance;
  Blob<Dtype> unitPotential;
  Blob<Dtype> horizontalPotential;
  Blob<Dtype> verticalPotential;
  std::vector<Dtype> indicator;
  std::vector<Dtype> output;
  vector<Blob<Dtype>*> potentialVec;

  SegmentationParameter segmentationParam;
  shared_ptr<Blob<Dtype>> dataWeight;
  std::unique_ptr<SegmentationEnergy<Dtype>> energy;
};

TYPED_TEST_CASE(SegmentationenergyTest, TestDtypes);
//TYPED_TEST_CASE(SegmentationenergyTest, ::testing::Types<double>);


TYPED_TEST(SegmentationenergyTest, TestTimesHorizontalB_CPU) {
  static const TypeParam result[9] = { .7,    -.5,     .0,
                                      -.2,    -.2,     .0,
                                      -.5,     .7,     .0};

  this->energy->timesHorizontalB_cpu(this->indicator.data(), this->output.data());
  ASSERT_NEAR_VEC(this->output.data(), result, 9);
}

TYPED_TEST(SegmentationenergyTest, TestTimesVerticalB_CPU) {
  static const TypeParam result[9] = { .5,    -.4,     -.1,
                                      -.1,    -.4,     .5,
                                      .0,     .0,     .0};

  this->energy->timesVerticalB_cpu(this->indicator.data(), this->output.data());

  ASSERT_NEAR_VEC(this->output.data(), result, 9);
}

TYPED_TEST(SegmentationenergyTest, TestTimesHorizontalBt_CPU) {
  static const TypeParam result[9] = { .8,    -.7,     -.1,
                                      .3,    .2,     -.5,
                                      .4,     .5,     -.9};

  this->energy->timesHorizontalBt_cpu(this->indicator.data(), this->output.data());

  ASSERT_NEAR_VEC(this->output.data(), result, 9);
}

TYPED_TEST(SegmentationenergyTest, TestTimesVerticalBt_CPU) {
  static const TypeParam result[9] = { .8,    .1,     .6,
                                      -.5,    .4,     .1,
                                      -.3,     -.5,     -.7};

  this->energy->timesVerticalBt_cpu(this->indicator.data(), this->output.data());
  ASSERT_NEAR_VEC(this->output.data(), result, 9);
}

TYPED_TEST(SegmentationenergyTest, TestEnergy_CPU) {

    const TypeParam expected = 20.4989225685541;
    TypeParam energy = this->energy->energy_cpu(this->indicator.data());
    ASSERT_NEAR(energy, expected, this->tolerance);
}

TYPED_TEST(SegmentationenergyTest, TestEnergyGradient_CPU) {
  static const TypeParam result[9] = { 5.24999395924417,         -9.88887659825024,          1.33342930339133,
                                      -2.90463293667475,                       0.5,          3.90463293667475,
                                      -0.333429303391329,          10.8888765982502,         -4.24999395924417};

  this->energy->computeEnergyGradient_cpu(this->indicator.data(), this->output.data());
  ASSERT_NEAR_VEC(this->output.data(), result, 9);
}

TYPED_TEST(SegmentationenergyTest, TestEnergyMinimizationGradientDescent_CPU) {

    TypeParam energyBefore = this->energy->energy_cpu(this->indicator.data());
    TypeParam energyAfter = 0;
    for(int i = 0; i < 10; ++i) {
      this->energy->minimizeGD_cpu(this->indicator.data());

      energyAfter = this->energy->energy_cpu(this->indicator.data());
      ASSERT_GT(energyBefore, energyAfter) << "failed at i = " << i;
      energyBefore = energyAfter;
    }
}

TYPED_TEST(SegmentationenergyTest, TestEnergyMinimizationNesterovAcceleratedGradient_CPU) {

    TypeParam energyBefore = this->energy->energy_cpu(this->indicator.data());
    TypeParam energyAfter = 0;

    for(int i = 0; i < 10; ++i) {
      this->energy->minimizeNAG_cpu(this->indicator.data());

      energyAfter = this->energy->energy_cpu(this->indicator.data());
      ASSERT_GT(energyBefore, energyAfter) << "failed at i = " << i;
      energyBefore = energyAfter;
    }
}

// tests finite-difference hessian-vec approximation
TYPED_TEST(SegmentationenergyTest, TestApproxHessianVec_CPU) {

    TypeParam result[9] = { 0.0748398918926085,         0.861344049711832,        0.0599899557474082,
                            0.0733492169024963,        0.0506015600298948,        0.0434338498878084,
                            0.0382463873213612,          0.48423417715604,         0.165508019547289};

    TypeParam vec[9] = {0.002817504723421,   0.008508397242876,   0.006644304780617,
                        0.005576876972500,   0.006325079707341,   0.003303039106391,
                        0.004236820567728,   0.004783289295340,   0.006230888066461};

    this->energy->approxHessVec_cpu(this->indicator.data(), vec, this->output.data());

    // finite differences aren't too good, especially for floats
    ASSERT_NEAR_VEC_WTOL(this->output.data(), result, 9, this->tolerance * 1e3);
}

TYPED_TEST(SegmentationenergyTest, TestSparseHessianMultiply_CPU) {

  TypeParam result[9] = {0.7656, 1.2851, 1.4304, 0.5353, 1.4780, 0.8212, 0.7067, 0.9554, 0.6005};
  TypeParam diags[45] = { 0.46,    0.45,    0.67,    0.31,    0.19,    0.17,       0,       0,       0,
                          0.40,    0.26,    0.59,    0.01,    0.86,    0.56,    0.48,    0.31,       0,
                          0.60,    0.96,    0.34,    0.66,    0.02,    0.30,    0.32,    0.82,    0.55,
                             0,    0.05,    0.77,    0.04,    0.49,    0.48,    0.51,    0.52,    0.30,
                             0,       0,       0,    0.01,    0.96,    0.12,    0.31,    0.24,    0.35};
  TypeParam vec[9] =    {0.28,    0.85,    0.66,    0.56,    0.63,    0.33,    0.42,    0.48,    0.62};


  caffe_copy<TypeParam>(45, diags, this->energy->bufferHessian_.mutable_cpu_data());
  this->energy->sparseHessianMultiply_cpu(vec, this->output.data());
  ASSERT_NEAR_VEC(this->output.data(), result, 9);
}

TYPED_TEST(SegmentationenergyTest, TestComputeSparseHessian_CPU) {

  // expected result
  TypeParam diagP2[9] = {-1.59996160076799e-05,       -3.124882816162e-05,                         0
                          -0.00199880059972013,      -0.00199880059972013,                         0
                           -3.124882816162e-05,     -1.59996160076799e-05,                         0};

  TypeParam diagP1[9] = {-5.83083239199409e-06,     -1.59996160076798e-05,                         0,
                         -0.000249962504686953,     -0.000249962504686953,                         0,
                         -1.59996160076799e-05,      -5.8308323919941e-06,                         0};

  TypeParam diag0[9] =  {     26.5625218304484,          101.234620980511,          9.02979257799351,
                              13.1541922003621,           8.0005624226657,          13.1541922003621,
                               9.0297925779935,          101.234620980511,          26.5625218304484};

  TypeParam diagM1[9] = {                    0,     -5.83083239199409e-06,     -1.59996160076798e-05,
                                             0,     -0.000249962504686953,     -0.000249962504686953,
                                             0,     -1.59996160076799e-05,      -5.8308323919941e-06};

  TypeParam diagM2[9] = {                   0,                         0,                          0,
                        -1.59996160076799e-05,       -3.124882816162e-05,       -0.00199880059972013,
                         -0.00199880059972013,       -3.124882816162e-05,     -1.59996160076799e-05};

  this->energy->computeSparseHessian_cpu(this->indicator.data());
  const TypeParam* output = this->energy->bufferHessian_.cpu_data();

  ASSERT_NEAR_VEC(output, diagP2, 9);
  ASSERT_NEAR_VEC(output+9, diagP1, 9);
  ASSERT_NEAR_VEC(output+18, diag0, 9);
  ASSERT_NEAR_VEC(output+27, diagM1, 9);
  ASSERT_NEAR_VEC(output+36, diagM2, 9);
}

TYPED_TEST(SegmentationenergyTest, TestSparseHessianVec_CPU) {

  TypeParam finiteDifferenceApprox[9];

  this->energy->approxHessVec_cpu(this->indicator.data(), this->indicator.data(), finiteDifferenceApprox);
  this->energy->computeSparseHessian_cpu(this->indicator.data());
  this->energy->sparseHessianMultiply_cpu(this->indicator.data(), this->output.data());

  ASSERT_NEAR_VEC_WTOL(this->output.data(), finiteDifferenceApprox, 9, this->tolerance * 1e3);
}

TYPED_TEST(SegmentationenergyTest, TestInvertedHessianVec_CPU) {

  // symmetric positive definite
  TypeParam input[9] = {    0.5000,    0.3515,    0.2624,
                            0.3515,    0.4109,    0.3515,
                            0.2624,    0.3515,    0.5000};

    // computed in Matlab by inverting finite difference approximation of the Hessian; therefore
    // it isn't too accurate and we cannot impose too small a tolerance
    TypeParam result[9] = { 0.0350049302417931,        0.0811452351143305,        0.0403377151174284,
                            0.0534829919424272,        0.0714744202154454,        0.0315535278331092,
                            0.0256742209074436,        0.0458555914563377,        0.077494080887268};

    TypeParam vec[9] = {0.28,    0.85,    0.66,
                        0.56,    0.63,    0.33,
                        0.42,    0.48,    0.62};

    this->energy->computeSparseHessian_cpu(input);
    this->energy->invHessianVector_cpu(input, vec, this->output.data());

    ASSERT_NEAR_VEC_WTOL(this->output.data(), result, 9, 1e-4);
}

}  // namespace caffe
