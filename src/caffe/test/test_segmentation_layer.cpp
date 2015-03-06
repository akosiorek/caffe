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

namespace caffe {

template <typename Dtype>
class SegmentationLayerMock : public SegmentationLayer<Dtype> {
public:
	explicit SegmentationLayerMock(const LayerParameter& layerParam) : SegmentationLayer<Dtype>(layerParam) {}
	void minimize_cpu_mock(Dtype* indicatorValue, Dtype* indicatorGrad,
				const Dtype* unit, const Dtype* horizontal, const Dtype* vertical) {
		this->minimize_cpu(indicatorValue, indicatorGrad, unit, horizontal, vertical);
	}
	void computeEnergyGradient_cpu_mock(Dtype* indicatorValue, Dtype* indicatorGrad,
				const Dtype* unit, const Dtype* horizontal, const Dtype* vertical) {
		this->computeEnergyGradient_cpu(indicatorValue, indicatorGrad, unit, horizontal, vertical);
	}
};

template <typename Dtype>
class SegmentationLayerTest : public ::testing::Test {
 protected:
  SegmentationLayerTest()
 	 : segmentationParam(layerParam.mutable_segmentation_param()),
	   tolerance(1e-10) {

	  bottomVec.push_back(&unitPotential);
	  bottomVec.push_back(&horizontalPotential);
	  bottomVec.push_back(&verticalPotential);
	  topVec.push_back(&topBlob);
	  segmentationParam->set_learning_rate(0.01);
	  segmentationParam->set_gradient_norm_tolerance(1e-4);
  }
  virtual ~SegmentationLayerTest() {}

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

  Blob<Dtype> unitPotential;
  Blob<Dtype> horizontalPotential;
  Blob<Dtype> verticalPotential;
  Blob<Dtype> topBlob;
  vector<Blob<Dtype>*> bottomVec;
  vector<Blob<Dtype>*> topVec;
  LayerParameter layerParam;
  SegmentationParameter* segmentationParam;
  float tolerance;

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

TYPED_TEST(SegmentationLayerTest, TestForwardGradientCPU) {
  static const TypeParam result[9] = {1.0, 1.0, 1.0,
		  	  	  	  	  	  	  	  1.0, 1.0, 1.0,
		  	  	  	  	  	  	  	  1.0, 1.0, 1.0};

  this->reshape(1, 1, 3, 3);
  this->setBottom(1.0);
  SegmentationLayerMock<TypeParam> layer(this->layerParam);
  layer.SetUp(this->bottomVec, this->topVec);

  TypeParam* indicatorValue = this->topBlob.mutable_cpu_data();
  TypeParam* indicatorGrad = this->topBlob.mutable_cpu_diff();
  const TypeParam* unit = this->unitPotential.cpu_data();
  const TypeParam* horizontal = this->horizontalPotential.cpu_data();
  const TypeParam* vertical = this->verticalPotential.cpu_data();
  layer.computeEnergyGradient_cpu_mock(indicatorValue, indicatorGrad, unit, horizontal, vertical);

  for(int i = 0; i < this->unitPotential.count(); ++i) {
	  ASSERT_NEAR(indicatorGrad[i], result[i], this->tolerance);
  }
}

TYPED_TEST(SegmentationLayerTest, TestEnergyMinimizationCPU) {
  static const TypeParam result[9] = {1.0, 1.0, 1.0,
		  	  	  	  	  	  	  	  1.0, 1.0, 1.0,
		  	  	  	  	  	  	  	  1.0, 1.0, 1.0};

  this->reshape(1, 1, 3, 3);
  this->setBottom(1.0);
  SegmentationLayerMock<TypeParam> layer(this->layerParam);
  layer.SetUp(this->bottomVec, this->topVec);

  TypeParam* indicatorValue = this->topBlob.mutable_cpu_data();
  TypeParam* indicatorGrad = this->topBlob.mutable_cpu_diff();
  const TypeParam* unit = this->unitPotential.cpu_data();
  const TypeParam* horizontal = this->horizontalPotential.cpu_data();
  const TypeParam* vertical = this->verticalPotential.cpu_data();
  layer.minimize_cpu_mock(indicatorValue, indicatorGrad, unit, horizontal, vertical);

  for(int i = 0; i < this->unitPotential.count(); ++i) {
	  ASSERT_NEAR(indicatorGrad[i], result[i], this->tolerance);
  }
}

//TYPED_TEST(AccuracyLayerTest, TestSetup) {
//  LayerParameter layer_param;
//  AccuracyLayer<TypeParam> layer(layer_param);
//  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//  EXPECT_EQ(this->blob_top_->num(), 1);
//  EXPECT_EQ(this->blob_top_->channels(), 1);
//  EXPECT_EQ(this->blob_top_->height(), 1);
//  EXPECT_EQ(this->blob_top_->width(), 1);
//}
//
//TYPED_TEST(AccuracyLayerTest, TestSetupTopK) {
//  LayerParameter layer_param;
//  AccuracyParameter* accuracy_param =
//      layer_param.mutable_accuracy_param();
//  accuracy_param->set_top_k(5);
//  AccuracyLayer<TypeParam> layer(layer_param);
//  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//  EXPECT_EQ(this->blob_top_->num(), 1);
//  EXPECT_EQ(this->blob_top_->channels(), 1);
//  EXPECT_EQ(this->blob_top_->height(), 1);
//  EXPECT_EQ(this->blob_top_->width(), 1);
//}
//
//TYPED_TEST(AccuracyLayerTest, TestForwardCPU) {
//  LayerParameter layer_param;
//  Caffe::set_mode(Caffe::CPU);
//  AccuracyLayer<TypeParam> layer(layer_param);
//  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//
//  TypeParam max_value;
//  int max_id;
//  int num_correct_labels = 0;
//  for (int i = 0; i < 100; ++i) {
//    max_value = -FLT_MAX;
//    max_id = 0;
//    for (int j = 0; j < 10; ++j) {
//      if (this->blob_bottom_data_->data_at(i, j, 0, 0) > max_value) {
//        max_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
//        max_id = j;
//      }
//    }
//    if (max_id == this->blob_bottom_label_->data_at(i, 0, 0, 0)) {
//      ++num_correct_labels;
//    }
//  }
//  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
//              num_correct_labels / 100.0, 1e-4);
//}
//
//TYPED_TEST(AccuracyLayerTest, TestForwardCPUTopK) {
//  LayerParameter layer_param;
//  AccuracyParameter* accuracy_param = layer_param.mutable_accuracy_param();
//  accuracy_param->set_top_k(this->top_k_);
//  AccuracyLayer<TypeParam> layer(layer_param);
//  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
//
//  TypeParam current_value;
//  int current_rank;
//  int num_correct_labels = 0;
//  for (int i = 0; i < 100; ++i) {
//    for (int j = 0; j < 10; ++j) {
//      current_value = this->blob_bottom_data_->data_at(i, j, 0, 0);
//      current_rank = 0;
//      for (int k = 0; k < 10; ++k) {
//        if (this->blob_bottom_data_->data_at(i, k, 0, 0) > current_value) {
//          ++current_rank;
//        }
//      }
//      if (current_rank < this->top_k_ &&
//          j == this->blob_bottom_label_->data_at(i, 0, 0, 0)) {
//        ++num_correct_labels;
//      }
//    }
//  }
//
//  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0),
//              num_correct_labels / 100.0, 1e-4);
//}

}  // namespace caffe
