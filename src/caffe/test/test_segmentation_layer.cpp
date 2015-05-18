#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>
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
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename Dtype>
class SegmentationLayerTest : public ::testing::Test {
 protected:
  SegmentationLayerTest()
 	 : segmentationParam(layerParam.mutable_segmentation_param()),
 	   bottomVec({&unitPotential, &horizontalPotential, &verticalPotential}),
 	   topVec({&topBlob}) {

	  segmentationParam->set_step_size(10);
	  segmentationParam->set_minimization_iters(1e2);
	  segmentationParam->set_init_data_weight(0.5);
	  segmentationParam->set_log_barrier_weight(1e-3);
	  segmentationParam->set_smoothnes_eps(1e-3);
	  segmentationParam->set_min_grad_norm(0);
	  segmentationParam->set_step_size_decay(1);
	  FillerParameter* filler = segmentationParam->mutable_indicator_filler();
	  filler->set_type("uniform");
	  filler->set_min(0.1);
	  filler->set_max(0.9);
//	  filler->set_type("constant");
//    filler->set_value(0.5);
	  layer = std::move(std::unique_ptr<SegmentationLayer<Dtype>>(new SegmentationLayer<Dtype>(layerParam)));


	  for(auto bottom : bottomVec) {
	    bottom->Reshape(1, 1, 3, 3);
	  }

    layer->SetUp(this->bottomVec, this->topVec);
  }

  LayerParameter layerParam;
  SegmentationParameter* segmentationParam;
  Blob<Dtype> unitPotential;
  Blob<Dtype> horizontalPotential;
  Blob<Dtype> verticalPotential;
  Blob<Dtype> topBlob;
  vector<Blob<Dtype>*> bottomVec;
  vector<Blob<Dtype>*> topVec;

  std::unique_ptr<SegmentationLayer<Dtype>> layer;
};

//TYPED_TEST_CASE(SegmentationLayerTest, TestDtypes);
TYPED_TEST_CASE(SegmentationLayerTest, ::testing::Types<double>);

TYPED_TEST(SegmentationLayerTest, TestSetUp) {
  typedef TypeParam Dtype;

  LayerParameter layerParam;
  shared_ptr<SegmentationLayer<Dtype> > layer(
      new SegmentationLayer<Dtype>(layerParam));
  layer->SetUp(this->bottomVec, this->topVec);

  EXPECT_EQ(this->topBlob.num(), 1);
  EXPECT_EQ(this->topBlob.channels(), 1);
  EXPECT_EQ(this->topBlob.height(), 3);
  EXPECT_EQ(this->topBlob.width(), 3);
}

TYPED_TEST(SegmentationLayerTest, ForwardTest) {
  typedef TypeParam Dtype;

  this->layer->Forward(this->bottomVec, this->topVec);
  const Dtype* data = this->topBlob.cpu_data();

  shared_ptr<Blob<Dtype>> dataWeight(new Blob<Dtype>);
  dataWeight->Reshape(1, 1, 1, 1);
  *dataWeight->mutable_cpu_data() = 1;
  SegmentationEnergy<Dtype> energy(*this->segmentationParam, dataWeight);
  energy.reshape(3, 3);
  energy.setData(&this->unitPotential, &this->horizontalPotential, &this->verticalPotential);


  auto energyValue = energy.energy_cpu(data);

  // not that it proves anything...
  EXPECT_LE(energyValue, 1);
//  LOG(ERROR) << energyValue;
}

TYPED_TEST(SegmentationLayerTest, GradientTest) {
  typedef TypeParam Dtype;

  //only for doubles due to cancellation
  if(sizeof(Dtype) == sizeof(double)) {
  GradientChecker<Dtype> checker(1e-6, 1e-3);
  checker.CheckGradientExhaustive(this->layer.get(), this->bottomVec,
          this->topVec);
  }
}


}  // namespace caffe
