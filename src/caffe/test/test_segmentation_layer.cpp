#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>
#include <cmath>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

#define ASSERT_NEAR_VEC_WTOL(computed, expected, count, tol)   for(int i = 0; i < count; ++i) ASSERT_NEAR((computed)[i], (expected)[i], fmax((expected)[i] * tol, tol)) << "i = " << i
#define ASSERT_NEAR_VEC(computed, expected, count) ASSERT_NEAR_VEC_WTOL(computed, expected, count, this->tolerance)
namespace caffe {

template <typename Dtype>
class SegmentationLayerMock : public SegmentationLayer<Dtype> {
public:
	explicit SegmentationLayerMock(const LayerParameter& layerParam) : SegmentationLayer<Dtype>(layerParam) {}
	void minimize_cpu_mock(Dtype* indicatorValue, Dtype* indicatorGrad,
				const Dtype* unit, const Dtype* horizontal, const Dtype* vertical) {
	  setPotential(unit, horizontal, vertical);

		this->minimize_cpu(indicatorValue, indicatorGrad);

		resetPotential();
	}
	void computeEnergyGradient_cpu_mock(Dtype* indicatorValue, Dtype* indicatorGrad,
				const Dtype* unit, const Dtype* horizontal, const Dtype* vertical) {

	  setPotential(unit, horizontal, vertical);

		this->computeEnergyGradient_cpu(indicatorValue, indicatorGrad);

		resetPotential();
	}

	void timesHorizontalB_cpu_mock(const Dtype* f, Dtype* dx) {
	  this->timesHorizontalB_cpu(f, dx);
	}

  void timesVerticalB_cpu_mock(const Dtype* f, Dtype* dy) {
    this->timesVerticalB_cpu(f, dy);
  }

  void timesHorizontalBt_cpu_mock(const Dtype* grad, Dtype* diffDx) {
    this->timesHorizontalBt_cpu(grad, diffDx);
  }

  void timesVerticalBt_cpu_mock(const Dtype* grad, Dtype* diffDx) {
    this->timesVerticalBt_cpu(grad, diffDx);
  }

  Dtype energy_cpu_mock(const Dtype* indicator, const Dtype* unit, const Dtype* horizontal, const Dtype* vertical) {
    setPotential(unit, horizontal, vertical);

    float energy = this->energy_cpu(indicator);

    resetPotential();
    return energy;
  }

  void approxHessVec_cpu_mock(const Dtype* indicator, const Dtype* vec, Dtype* Hv, const Dtype* unit, const Dtype* horizontal, const Dtype* vertical)  {
    setPotential(unit, horizontal, vertical);

    this->approxHessVec_cpu(indicator, vec, Hv);

    resetPotential();
  }

  void invertedHessianVector_cpu_mock(const Dtype* indicator, const Dtype* vec, Dtype* iHv, const Dtype* unit, const Dtype* horizontal, const Dtype* vertical)  {
    setPotential(unit, horizontal, vertical);

    this->invHessianVector_cpu(indicator, vec, iHv);
//    printVec(iHv);
    resetPotential();
  }

  void sparseHessianMultiply_cpu_mock(Dtype* vec, Dtype** diags, Dtype* output) {

    Dtype* data = this->bufferHessian_.mutable_cpu_data();
    for(int i = 0; i < 5; ++i) {
      for(int j = 0; j < this->N_; ++j) {
        data[i * this->N_ + j] = diags[i][j];
      }
    }

    this->sparseHessianMultiply_cpu(vec, output);
  }

  const Dtype* computeSparseHessian_cpu_mock(const Dtype* indicator, const Dtype* unit, const Dtype* horizontal, const Dtype* vertical) {
    setPotential(unit, horizontal, vertical);

    this->computeSparseHessian_cpu(indicator);
    resetPotential();
    return this->bufferHessian_.cpu_data();
  }

  void sparseHessianMultiply_cpu_mock(const Dtype* vec, Dtype* output) {
    this->sparseHessianMultiply_cpu(vec, output);
  }

	void printVec(const Dtype* vec) {
	  for(int i = 0; i < this->N_; ++i) {
	    std::cout << vec[i] << " ";
	  }
	  std::cout << std::endl;
	}

	void setPotential(const Dtype* unit, const Dtype* horizontal, const Dtype* vertical) {
	  this->unit_ = unit;
    this->horizontal_ = horizontal;
    this->vertical_ = vertical;
	}

	void resetPotential() {

	  this->unit_ = nullptr;
    this->horizontal_ = nullptr;
    this->vertical_ = nullptr;
	}
};





template<typename T>
struct tolerance_trait {
  static const constexpr double tol = 1e-6;
};

template<>
struct tolerance_trait<float> {
  static const constexpr float tol = 1e-3;
};


template <typename Dtype>
class SegmentationLayerTest : public ::testing::Test {
 protected:
  SegmentationLayerTest()
 	 : segmentationParam(layerParam.mutable_segmentation_param()),
	   tolerance(tolerance_trait<Dtype>::tol) {

	  bottomVec.push_back(&unitPotential);
	  bottomVec.push_back(&horizontalPotential);
	  bottomVec.push_back(&verticalPotential);
	  topVec.push_back(&topBlob);
	  segmentationParam->set_step_size(0.001);
	  segmentationParam->set_gradient_norm_tolerance(1e-4);
	  segmentationParam->set_minimization_iters(10);
	  segmentationParam->set_data_weight(1);
	  segmentationParam->set_log_barrier_weight(1);
	  segmentationParam->set_smoothnes_eps(1e-3);
	  segmentationParam->set_inv_hess_iters(10);
	  segmentationParam->set_inv_hess_tolerance(1e-8);

	  layer = new SegmentationLayerMock<Dtype>(layerParam);

	  this->reshape(1, 1, 3, 3);
    this->setBottom(0.5);
    layer->SetUp(this->bottomVec, this->topVec);
    indicatorValue = this->topBlob.mutable_cpu_data();

    //magic(3)/10
	  const Dtype indicator[9] = { 0.8000,    0.1000,    0.6000,
                                 0.3000,    0.5000,    0.7000,
                                 0.4000,    0.9000,    0.2000};

	  caffe_copy<Dtype>(9, indicator, indicatorValue);


  }
  virtual ~SegmentationLayerTest() {

    delete layer;
  }

  void reshape(int num, int c, int height, int width) {
	  for(int i = 0; i < bottomVec.size(); ++i) {
		  bottomVec[i]->Reshape(num, c, height, width);
	  }
//	  topBlob.Reshape(num, c, height, width);
  }

  void setBottom(Dtype value) {
	  for(int i = 0; i < bottomVec.size(); ++i) {
		  caffe_set<Dtype>(bottomVec[i]->count(), value, bottomVec[i]->mutable_cpu_data());
	  }
  }

  void setTop(Dtype value) {
    for(int i = 0; i < topVec.size(); ++i) {
      caffe_set<Dtype>(topVec[i]->count(), value, topVec[i]->mutable_cpu_data());
    }
  }

  void printVec(const Dtype* vec) {
    for(int i = 0; i < this->unitPotential.count(); ++i) {
      std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
  }

  Blob<Dtype> unitPotential;
  Blob<Dtype> horizontalPotential;
  Blob<Dtype> verticalPotential;
  Blob<Dtype> topBlob;
  vector<Blob<Dtype>*> bottomVec;
  vector<Blob<Dtype>*> topVec;
  LayerParameter layerParam;
  SegmentationParameter* segmentationParam;
  Dtype tolerance;

  SegmentationLayerMock<Dtype>* layer;
  Dtype* indicatorValue;

};

//TYPED_TEST_CASE(SegmentationLayerTest, TestDtypes);
TYPED_TEST_CASE(SegmentationLayerTest, ::testing::Types<double>);

TYPED_TEST(SegmentationLayerTest, TestSetup) {
  this->reshape(1, 1, 1, 1);
  SegmentationLayer<TypeParam> layer(this->layerParam);
  layer.SetUp(this->bottomVec, this->topVec);
  EXPECT_EQ(this->topBlob.num(), 1);
  EXPECT_EQ(this->topBlob.channels(), 1);
  EXPECT_EQ(this->topBlob.height(), 1);
  EXPECT_EQ(this->topBlob.width(), 1);
}

TYPED_TEST(SegmentationLayerTest, TestTimesHorizontalB_CPU) {
  static const TypeParam result[9] = { .7,    -.5,     .0,
                                      -.2,    -.2,     .0,
                                      -.5,     .7,     .0};

  TypeParam* indicatorGrad = this->topBlob.mutable_cpu_diff();
  this->layer->timesHorizontalB_cpu_mock(this->indicatorValue, indicatorGrad);

  ASSERT_NEAR_VEC(indicatorGrad, result, 9);
}

TYPED_TEST(SegmentationLayerTest, TestTimesVerticalB_CPU) {
  static const TypeParam result[9] = { .5,    -.4,     -.1,
                                      -.1,    -.4,     .5,
                                      .0,     .0,     .0};

  TypeParam* indicatorGrad = this->topBlob.mutable_cpu_diff();
  this->layer->timesVerticalB_cpu_mock(this->indicatorValue, indicatorGrad);

  ASSERT_NEAR_VEC(indicatorGrad, result, 9);
}

TYPED_TEST(SegmentationLayerTest, TestTimesHorizontalBt_CPU) {
  static const TypeParam result[9] = { .8,    -.7,     -.1,
                                      .3,    .2,     -.5,
                                      .4,     .5,     -.9};

  TypeParam* indicatorGrad = this->topBlob.mutable_cpu_diff();
  this->layer->timesHorizontalBt_cpu_mock(this->indicatorValue, indicatorGrad);

  ASSERT_NEAR_VEC(indicatorGrad, result, 9);
}

TYPED_TEST(SegmentationLayerTest, TestTimesVerticalBt_CPU) {
  static const TypeParam result[9] = { .8,    .1,     .6,
                                      -.5,    .4,     .1,
                                      -.3,     -.5,     -.7};

  TypeParam* indicatorGrad = this->topBlob.mutable_cpu_diff();
  this->layer->timesVerticalBt_cpu_mock(this->indicatorValue, indicatorGrad);

  ASSERT_NEAR_VEC(indicatorGrad, result, 9);
}

TYPED_TEST(SegmentationLayerTest, TestEnergy_CPU) {

    const TypeParam expected = 20.4989225685541;

    const TypeParam* unit = this->unitPotential.cpu_data();
    const TypeParam* horizontal = this->horizontalPotential.cpu_data();
    const TypeParam* vertical = this->verticalPotential.cpu_data();

    TypeParam energy = this->layer->energy_cpu_mock(this->indicatorValue, unit, horizontal, vertical);

    ASSERT_NEAR(energy, expected, this->tolerance);
}

TYPED_TEST(SegmentationLayerTest, TestEnergyGradient_CPU) {
  static const TypeParam result[9] = { 5.24999395924417,         -9.88887659825024,          1.33342930339133,
                                      -2.90463293667475,                       0.5,          3.90463293667475,
                                      -0.333429303391329,          10.8888765982502,         -4.24999395924417};

  TypeParam* indicatorGrad = this->topBlob.mutable_cpu_diff();
  const TypeParam* unit = this->unitPotential.cpu_data();
  const TypeParam* horizontal = this->horizontalPotential.cpu_data();
  const TypeParam* vertical = this->verticalPotential.cpu_data();
  this->layer->computeEnergyGradient_cpu_mock(this->indicatorValue, indicatorGrad, unit, horizontal, vertical);

  ASSERT_NEAR_VEC(indicatorGrad, result, 9);
}

TYPED_TEST(SegmentationLayerTest, TestEnergyMinimization_CPU) {

    const TypeParam expected = 20.493477172301080;

    TypeParam* indicatorGrad = this->topBlob.mutable_cpu_diff();
    const TypeParam* unit = this->unitPotential.cpu_data();
    const TypeParam* horizontal = this->horizontalPotential.cpu_data();
    const TypeParam* vertical = this->verticalPotential.cpu_data();

    TypeParam energyBefore = this->layer->energy_cpu_mock(this->indicatorValue, unit, horizontal, vertical);
    TypeParam energyAfter = 0;

    for(int i = 0; i < 10; ++i) {
//      std::cout << energyBefore << std::endl;
      this->layer->minimize_cpu_mock(this->indicatorValue, indicatorGrad, unit, horizontal, vertical);
//      this->printVec(this->indicatorValue);

      energyAfter = this->layer->energy_cpu_mock(this->indicatorValue, unit, horizontal, vertical);
      ASSERT_GT(energyBefore, energyAfter);
      energyBefore = energyAfter;
    }
}

// works for double, for float smoothnes_eps is too small!
TYPED_TEST(SegmentationLayerTest, TestApproxHessianVec_CPU) {

    TypeParam result[9] = { 0.0748398918926085,         0.861344049711832,        0.0599899557474082,
                            0.0733492169024963,        0.0506015600298948,        0.0434338498878084,
                            0.0382463873213612,          0.48423417715604,         0.165508019547289};

    TypeParam vec[9] = {0.002817504723421,   0.008508397242876,   0.006644304780617,
                        0.005576876972500,   0.006325079707341,   0.003303039106391,
                        0.004236820567728,   0.004783289295340,   0.006230888066461};

    TypeParam* indicatorGrad = this->topBlob.mutable_cpu_diff();
    const TypeParam* unit = this->unitPotential.cpu_data();
    const TypeParam* horizontal = this->horizontalPotential.cpu_data();
    const TypeParam* vertical = this->verticalPotential.cpu_data();


    this->layer->approxHessVec_cpu_mock(this->indicatorValue, vec, indicatorGrad, unit, horizontal, vertical);

    ASSERT_NEAR_VEC(indicatorGrad, result, 9);
}

TYPED_TEST(SegmentationLayerTest, TestSparseHessianMultiply_CPU) {

  TypeParam diagP2[9] = {0.46,    0.45,    0.67,    0.31,    0.19,    0.17,       0,       0,       0};
  TypeParam diagP1[9] = {0.40,    0.26,    0.59,    0.01,    0.86,    0.56,    0.48,    0.31,       0};
  TypeParam diag0[9] =  {0.60,    0.96,    0.34,    0.66,    0.02,    0.30,    0.32,    0.82,    0.55};
  TypeParam diagM1[9] = {   0,    0.05,    0.77,    0.04,    0.49,    0.48,    0.51,    0.52,    0.30};
  TypeParam diagM2[9] = {   0,       0,       0,    0.01,    0.96,    0.12,    0.31,    0.24,    0.35};
  TypeParam vec[9] =    {0.28,    0.85,    0.66,    0.56,    0.63,    0.33,    0.42,    0.48,    0.62};
  TypeParam result[9] = {0.7656, 1.2851, 1.4304, 0.5353, 1.4780, 0.8212, 0.7067, 0.9554, 0.6005};

  TypeParam* diags[5] = {diagP2, diagP1, diag0, diagM1, diagM2};
  TypeParam output[9];

  this->layer->sparseHessianMultiply_cpu_mock(vec, diags, output);

  ASSERT_NEAR_VEC(output, result, 9);
}

TYPED_TEST(SegmentationLayerTest, TestComputeSparseHessian_CPU) {

  // expected result
  TypeParam diagP2[9] = {-1.59996160076799e-05,       -3.124882816162e-05,                         0
                          -0.00199880059972013,      -0.00199880059972013,                         0
                           -3.124882816162e-05,     -1.59996160076799e-05,                         0};

  TypeParam diagP1[9] = {-5.83083239199409e-06,     -1.59996160076798e-05,     -0.000249962504686953,
                         -0.000249962504686953,     -1.59996160076799e-05,      -5.8308323919941e-06,
                                             0,                         0,                         0};

  TypeParam diag0[9] =  { 26.5625218304484,          101.234620980511,          9.02979257799351,
                          13.1541922003621,           8.0005624226657,          13.1541922003621,
                           9.0297925779935,          101.234620980511,          26.5625218304484};
//  TypeParam diagM1[9] = {                    0,    - 1.59996160076799e-05,      -0.00199880059972013,                        0,        -3.124882816162e-05,       -3.124882816162e-05,                        0,      -0.00199880059972013,     -1.59996160076799e-05};
//  TypeParam diagM2[9] = {                    0,                         0,                         0,    -5.83083239199409e-06,      -0.000249962504686953,     -1.59996160076799e-05,    -1.59996160076798e-05,     -0.000249962504686953,      -5.8308323919941e-06};

  const TypeParam* unit = this->unitPotential.cpu_data();
  const TypeParam* horizontal = this->horizontalPotential.cpu_data();
  const TypeParam* vertical = this->verticalPotential.cpu_data();

  const TypeParam* output = this->layer->computeSparseHessian_cpu_mock(this->indicatorValue, unit, horizontal, vertical);

  ASSERT_NEAR_VEC_WTOL(output, diagP2, 9, 1e-4);
//  ASSERT_NEAR_VEC(output+9, diagP1, 9);
  ASSERT_NEAR_VEC(output+18, diag0, 9);
//  ASSERT_NEAR_VEC(output+27, diagM1, 9);
//  ASSERT_NEAR_VEC(output+36, diagM2, 9);
}

TYPED_TEST(SegmentationLayerTest, TestSparseHessianVec_CPU) {

  // expected result
  TypeParam vec[9] =    {0.28,    0.85,    0.66,    0.56,    0.63,    0.33,    0.42,    0.48,    0.62};
  TypeParam output[9];

  const TypeParam* unit = this->unitPotential.cpu_data();
  const TypeParam* horizontal = this->horizontalPotential.cpu_data();
  const TypeParam* vertical = this->verticalPotential.cpu_data();

  TypeParam* approxFD = this->topBlob.mutable_cpu_diff();

  this->layer->approxHessVec_cpu_mock(this->indicatorValue, vec, approxFD, unit, horizontal, vertical);
  this->layer->computeSparseHessian_cpu_mock(this->indicatorValue, unit, horizontal, vertical);
  this->layer->sparseHessianMultiply_cpu_mock(vec, output);

  ASSERT_NEAR_VEC(output, approxFD, 9);
}


TYPED_TEST(SegmentationLayerTest, DISABLED_TestInvertedHessianVec_CPU) {

  // symmetric positive definite
  TypeParam input[9] = {    0.5000,    0.3515,    0.2624,
                            0.3515,    0.4109,    0.3515,
                            0.2624,    0.3515,    0.5000};

    TypeParam result[9] = { 0.0350049302417931,        0.0811452351143305,        0.0403377151174284,
                            0.0534829919424272,        0.0714744202154454,        0.0315535278331092,
                            0.0256742209074436,        0.0458555914563377,        0.077494080887268};

    TypeParam vec[9] = {0.28,    0.85,    0.66,
                        0.56,    0.63,    0.33,
                        0.42,    0.48,    0.62};

    TypeParam* indicatorGrad = this->topBlob.mutable_cpu_diff();
    const TypeParam* unit = this->unitPotential.cpu_data();
    const TypeParam* horizontal = this->horizontalPotential.cpu_data();
    const TypeParam* vertical = this->verticalPotential.cpu_data();


    this->layer->invertedHessianVector_cpu_mock(input, vec, indicatorGrad, unit, horizontal, vertical);

    ASSERT_NEAR_VEC_WTOL(indicatorGrad, result, 9, 1e-8);
}




}  // namespace caffe
