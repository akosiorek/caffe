#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

#define ASSERT_NEAR_VEC(x, y, count)   for(int i = 0; i < count; ++i) ASSERT_NEAR(x[i], y[i], this->tolerance)

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

  void hessianVector_cpu_mock(const Dtype* indicator, const Dtype* vec, Dtype* Hv, const Dtype* unit, const Dtype* horizontal, const Dtype* vertical)  {
    setPotential(unit, horizontal, vertical);

    this->hessianVector_cpu(indicator, vec, Hv);

    resetPotential();
  }

  void invertedHessianVector_cpu_mock(const Dtype* indicator, const Dtype* vec, Dtype* iHv, const Dtype* unit, const Dtype* horizontal, const Dtype* vertical)  {
    setPotential(unit, horizontal, vertical);

    this->invHessianVector_cpu(indicator, vec, iHv);

    resetPotential();
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
	  segmentationParam->set_smoothnes_eps(1e-6);

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

TYPED_TEST_CASE(SegmentationLayerTest, TestDtypes);

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

    const TypeParam expected = 20.492882713775742;

    const TypeParam* unit = this->unitPotential.cpu_data();
    const TypeParam* horizontal = this->horizontalPotential.cpu_data();
    const TypeParam* vertical = this->verticalPotential.cpu_data();

    TypeParam energy = this->layer->energy_cpu_mock(this->indicatorValue, unit, horizontal, vertical);

    ASSERT_NEAR(energy, expected, this->tolerance);
}

TYPED_TEST(SegmentationLayerTest, TestEnergyGradient_CPU) {
  static const TypeParam result[9] = { 5.249999999993960,  -9.888888888876599,   1.333333333429334,
                                      -2.904761904632904,   0.500000000000000,   3.904761904632906,
                                      -0.333333333429333,  10.888888888876600,  -4.249999999993959};

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

      energyAfter = this->layer->energy_cpu_mock(this->indicatorValue, unit, horizontal, vertical);
      ASSERT_GT(energyBefore, energyAfter);
      energyBefore = energyAfter;
    }
}

// works for double, for float smoothnes_eps is too small!
TYPED_TEST(SegmentationLayerTest, DISABLED_TestHessianVec_CPU) {

    TypeParam result[9] = { 0.074839969066431,   0.861343918323598,   0.059983307099110,
                            0.073346681261910,   0.050600637657716,   0.043441330666028,
                            0.038249074565955,   0.484234225250901,   0.165507964267064};

    TypeParam vec[9] = {0.002817504723421,   0.008508397242876,   0.006644304780617,
                        0.005576876972500,   0.006325079707341,   0.003303039106391,
                        0.004236820567728,   0.004783289295340,   0.006230888066461};

    TypeParam* indicatorGrad = this->topBlob.mutable_cpu_diff();
    const TypeParam* unit = this->unitPotential.cpu_data();
    const TypeParam* horizontal = this->horizontalPotential.cpu_data();
    const TypeParam* vertical = this->verticalPotential.cpu_data();


    this->layer->hessianVector_cpu_mock(this->indicatorValue, vec, indicatorGrad, unit, horizontal, vertical);

    ASSERT_NEAR_VEC(indicatorGrad, result, 9);
}

TYPED_TEST(SegmentationLayerTest, TestInvertedHessianVec_CPU) {

    TypeParam result[9] = { -0.000075259725310,   0.010523110505268,   0.008217025491338,
                            -0.000466077418319,   0.019251288916940,  -0.000639117458407,
                             0.007952348694153,   0.010487494106637,   0.000053236451596};

    TypeParam vec[9] = {0.002817504723421,   0.008508397242876,   0.006644304780617,
                        0.005576876972500,   0.006325079707341,   0.003303039106391,
                        0.004236820567728,   0.004783289295340,   0.006230888066461};

    TypeParam* indicatorGrad = this->topBlob.mutable_cpu_diff();
    const TypeParam* unit = this->unitPotential.cpu_data();
    const TypeParam* horizontal = this->horizontalPotential.cpu_data();
    const TypeParam* vertical = this->verticalPotential.cpu_data();


    this->layer->invertedHessianVector_cpu_mock(this->indicatorValue, vec, indicatorGrad, unit, horizontal, vertical);

    ASSERT_NEAR_VEC(indicatorGrad, result, 9);
}




}  // namespace caffe
