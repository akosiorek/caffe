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

	  segmentationParam->set_step_size(1e-3);
	  segmentationParam->set_minimization_iters(1e3);
	  segmentationParam->set_init_data_weight(0.5);
	  segmentationParam->set_log_barrier_weight(1e-3);
	  segmentationParam->set_smoothnes_eps(1e-3);
	  segmentationParam->set_min_grad_norm(0);
	  segmentationParam->set_step_size_decay(1);
	  segmentationParam->set_convex_param(1e1);
	  segmentationParam->set_init_lipschitz_constant(1e4);
	
	  FillerParameter* filler = segmentationParam->mutable_indicator_filler();
	  //filler->set_type("uniform");
	  //filler->set_min(0.1);
	  //filler->set_max(0.9);
	  filler->set_type("gaussian");
	  filler->set_mean(0.5);
	  filler->set_std(0.05);

	  layer = std::move(std::unique_ptr<SegmentationLayer<Dtype>>(new SegmentationLayer<Dtype>(layerParam)));

	  for(auto blob : bottomVec) {
	    blob->Reshape(1, 1, 3, 3);
	    caffe_rng_uniform<Dtype>(9, 0.499, 0.501, blob->mutable_cpu_data());
	  }
	  caffe_set<Dtype>(9, 0.1, unitPotential.mutable_cpu_data());
	  caffe_set<Dtype>(9, 0.3, horizontalPotential.mutable_cpu_data());
	  caffe_set<Dtype>(9, 0.4, verticalPotential.mutable_cpu_data());

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

  const auto& energy = this->layer->getEnergy();
  // energy value
  auto energyValue = energy.energy_cpu(data);
  ASSERT_LE(energyValue, 0.1);


  // energy gradient norm
  energy.computeEnergyGradient_cpu(this->topBlob.cpu_data(), this->topBlob.mutable_cpu_diff());
  auto gradientNorm = caffe_cpu_nrm2<Dtype>(9, this->topBlob.cpu_diff());
//  LOG(ERROR) << "Gradient norm = " << gradientNorm;

  ASSERT_LE(gradientNorm, 0.35);
  LOG(INFO) << energyValue << " " << gradientNorm;
}

TYPED_TEST(SegmentationLayerTest, GradientTest) {
  typedef TypeParam Dtype;

  //only for doubles due to cancellation
  if(sizeof(Dtype) == sizeof(double)) {
  GradientChecker<Dtype> checker(1e-2, 1e-3);
//  checker.CheckGradientExhaustive(this->layer.get(), this->bottomVec, this->topVec);
  checker.CheckGradientSingle(this->layer.get(), this->bottomVec, this->topVec, 0, -1, 0);
  }
}


}  // namespace caffe
