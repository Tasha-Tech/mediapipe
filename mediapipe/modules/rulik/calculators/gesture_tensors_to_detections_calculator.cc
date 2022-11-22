// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <unordered_map>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
//#include "mediapipe/calculators/tensor/tensors_to_detections_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location.h"
//#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/ret_check.h"

// Note: On Apple platforms MEDIAPIPE_DISABLE_GL_COMPUTE is automatically
// defined in mediapipe/framework/port.h. Therefore,
// "#ifndef MEDIAPIPE_DISABLE_GL_COMPUTE" and "#if MEDIAPIPE_METAL_ENABLED"
// below are mutually exclusive.
#ifndef MEDIAPIPE_DISABLE_GL_COMPUTE
//#include "mediapipe/gpu/gl_calculator_helper.h"
#endif  // !defined(MEDIAPIPE_DISABLE_GL_COMPUTE)

#if MEDIAPIPE_METAL_ENABLED
#import <CoreVideo/CoreVideo.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#import "mediapipe/gpu/MPPMetalHelper.h"
#include "mediapipe/gpu/MPPMetalUtil.h"
#endif  // MEDIAPIPE_METAL_ENABLED

namespace mediapipe {
namespace api2 {

// Convert result Tensors from object detection models into MediaPipe
// Detections.
//
// Input:
//  TENSORS - Vector of Tensors of type kFloat32. The vector of tensors can have
//            2 or 3 tensors. First tensor is the predicted raw boxes/keypoints.
//            The size of the values must be (num_boxes * num_predicted_values).
//            Second tensor is the score tensor. The size of the valuse must be
//            (num_boxes * num_classes). It's optional to pass in a third tensor
//            for anchors (e.g. for SSD models) depend on the outputs of the
//            detection model. The size of anchor tensor must be (num_boxes *
//            4).
//
// Input side packet:
//  ANCHORS (optional) - The anchors used for decoding the bounding boxes, as a
//      vector of `Anchor` protos. Not required if post-processing is built-in
//      the model.
//  IGNORE_CLASSES (optional) - The list of class ids that should be ignored, as
//      a vector of integers. It overrides the corresponding field in the
//      calculator options.
//
// Output:
//  DETECTIONS - Result MediaPipe detections.
//
// Usage example:
// node {
//   calculator: "GestureTensorsToDetectionsCalculator"
//   input_stream: "TENSORS:tensors"
//   output_stream: "DETECTIONS:detections"
// }
class GestureTensorsToDetectionsCalculator : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kInTensors{"TENSORS"};
  
  static constexpr Output<std::vector<Detection>> kOutDetections{"DETECTIONS"};

  MEDIAPIPE_NODE_CONTRACT(kInTensors, kOutDetections);  

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status ProcessCPU(CalculatorContext* cc,
                          std::vector<Detection>* output_detections);
    
  absl::Status DecodeBoxes(const float* raw_boxes, std::vector<float>* boxes);
  absl::Status ConvertToDetections(const float* detection_boxes,
                                   const float* detection_scores,
                                   const int* detection_classes,
                                   std::vector<Detection>* output_detections);

  Detection ConvertToDetection(float box_ymin, float box_xmin, float box_ymax,
                               float box_xmax, float score, int class_id,
                               bool flip_vertically);

  int num_classes_ = 10;
  int num_boxes_ = 16;
  int num_coords_ = 4;
  int max_results_ = -1;

  bool scores_tensor_index_is_set_ = false;  
  std::vector<int> box_indices_ = {0, 1, 2, 3};
  bool has_custom_box_indices_ = false;  

  std::unique_ptr<Tensor> raw_anchors_buffer_;
  std::unique_ptr<Tensor> decoded_boxes_buffer_;
  std::unique_ptr<Tensor> scored_boxes_buffer_;

  bool gpu_inited_ = false;
  bool gpu_input_ = false;
  bool anchors_init_ = false;
};

MEDIAPIPE_REGISTER_NODE(GestureTensorsToDetectionsCalculator);

absl::Status GestureTensorsToDetectionsCalculator::Open(CalculatorContext* cc) {  
  return absl::OkStatus();
}

absl::Status GestureTensorsToDetectionsCalculator::Process(CalculatorContext* cc) {
  auto output_detections = absl::make_unique<std::vector<Detection>>();  
 
  MP_RETURN_IF_ERROR(ProcessCPU(cc, output_detections.get()));  

  kOutDetections(cc).Send(std::move(output_detections));
  return absl::OkStatus();
}

absl::Status GestureTensorsToDetectionsCalculator::ProcessCPU(CalculatorContext* cc, 
                                                              std::vector<Detection>* output_detections) {
  const auto& input_tensors = *kInTensors(cc);

  std::cout << "Received " << input_tensors.size() << " tensors" << std::endl;

  std::vector<float> boxes(num_boxes_ * num_coords_);  
  std::vector<float> detection_scores(num_boxes_);
  std::vector<int> detection_classes(num_boxes_);
  for (int i = 0; i < num_boxes_; ++i) {
    detection_scores[i] = 0; //max_score;
    detection_classes[i] = 0; //class_id;
  }

  for (int i = 0; i < num_boxes_; ++i) {
    const int box_offset = i * num_coords_;
    const float ymin = 0;
    const float xmin = 0;
    const float ymax = 0;
    const float xmax = 0;
    (boxes)[i * num_coords_ + 0] = ymin;
    (boxes)[i * num_coords_ + 1] = xmin;
    (boxes)[i * num_coords_ + 2] = ymax;
    (boxes)[i * num_coords_ + 3] = xmax;
  }

# if 0
  // Postprocessing on CPU for model without postprocessing op. E.g. output
  // raw score tensor and box tensor. Anchor decoding will be handled below.
  // TODO: Add flexible input tensor size handling.
  auto raw_box_tensor =
      &input_tensors[tensor_mapping_.detections_tensor_index()];
  RET_CHECK_EQ(raw_box_tensor->shape().dims.size(), 3);
  RET_CHECK_EQ(raw_box_tensor->shape().dims[0], 1);
  RET_CHECK_GT(num_boxes_, 0) << "Please set num_boxes in calculator options";
  RET_CHECK_EQ(raw_box_tensor->shape().dims[1], num_boxes_);
  RET_CHECK_EQ(raw_box_tensor->shape().dims[2], num_coords_);
  auto raw_score_tensor =
      &input_tensors[tensor_mapping_.scores_tensor_index()];
  RET_CHECK_EQ(raw_score_tensor->shape().dims.size(), 3);
  RET_CHECK_EQ(raw_score_tensor->shape().dims[0], 1);
  RET_CHECK_EQ(raw_score_tensor->shape().dims[1], num_boxes_);
  RET_CHECK_EQ(raw_score_tensor->shape().dims[2], num_classes_);
  auto raw_box_view = raw_box_tensor->GetCpuReadView();
  auto raw_boxes = raw_box_view.buffer<float>();
  auto raw_scores_view = raw_score_tensor->GetCpuReadView();
  auto raw_scores = raw_scores_view.buffer<float>();

  // TODO: Support other options to load anchors.
  if (!anchors_init_) {
    if (input_tensors.size() == kNumInputTensorsWithAnchors) {
      auto anchor_tensor =
          &input_tensors[tensor_mapping_.anchors_tensor_index()];
      RET_CHECK_EQ(anchor_tensor->shape().dims.size(), 2);
      RET_CHECK_EQ(anchor_tensor->shape().dims[0], num_boxes_);
      RET_CHECK_EQ(anchor_tensor->shape().dims[1], kNumCoordsPerBox);
      auto anchor_view = anchor_tensor->GetCpuReadView();
      auto raw_anchors = anchor_view.buffer<float>();
      ConvertRawValuesToAnchors(raw_anchors, num_boxes_, &anchors_);
    } else if (!kInAnchors(cc).IsEmpty()) {
      anchors_ = *kInAnchors(cc);
    } else {
      return absl::UnavailableError("No anchor data available.");
    }
    anchors_init_ = true;
  }
  std::vector<float> boxes(num_boxes_ * num_coords_);
  MP_RETURN_IF_ERROR(DecodeBoxes(raw_boxes, anchors_, &boxes));

  std::vector<float> detection_scores(num_boxes_);
  std::vector<int> detection_classes(num_boxes_);

  // Filter classes by scores.
  for (int i = 0; i < num_boxes_; ++i) {
    int class_id = -1;
    float max_score = -std::numeric_limits<float>::max();
    // Find the top score for box i.
    for (int score_idx = 0; score_idx < num_classes_; ++score_idx) {
      if (IsClassIndexAllowed(score_idx)) {
        auto score = raw_scores[i * num_classes_ + score_idx];
        if (options_.sigmoid_score()) {
          if (options_.has_score_clipping_thresh()) {
            score = score < -options_.score_clipping_thresh()
                        ? -options_.score_clipping_thresh()
                        : score;
            score = score > options_.score_clipping_thresh()
                        ? options_.score_clipping_thresh()
                        : score;
          }
          score = 1.0f / (1.0f + std::exp(-score));
        }
        if (max_score < score) {
          max_score = score;
          class_id = score_idx;
        }
      }
    }
    detection_scores[i] = max_score;
    detection_classes[i] = class_id;
  }
#endif
  MP_RETURN_IF_ERROR(
      ConvertToDetections(boxes.data(), detection_scores.data(),
                          detection_classes.data(), output_detections));
  
  return absl::OkStatus();
}


absl::Status GestureTensorsToDetectionsCalculator::Close(CalculatorContext* cc) {
  return absl::OkStatus();
}


absl::Status GestureTensorsToDetectionsCalculator::ConvertToDetections(
    const float* detection_boxes, const float* detection_scores,
    const int* detection_classes, std::vector<Detection>* output_detections) {
  for (int i = 0; i < num_boxes_; ++i) {  
    const int box_offset = i * num_coords_;
    Detection detection = ConvertToDetection(
        /*box_ymin=*/detection_boxes[box_offset + box_indices_[0]],
        /*box_xmin=*/detection_boxes[box_offset + box_indices_[1]],
        /*box_ymax=*/detection_boxes[box_offset + box_indices_[2]],
        /*box_xmax=*/detection_boxes[box_offset + box_indices_[3]],
        detection_scores[i], detection_classes[i], false);
    const auto& bbox = detection.location_data().relative_bounding_box();
    if (bbox.width() < 0 || bbox.height() < 0 || std::isnan(bbox.width()) ||
        std::isnan(bbox.height())) {
      // Decoded detection boxes could have negative values for width/height due
      // to model prediction. Filter out those boxes since some downstream
      // calculators may assume non-negative values. (b/171391719)
      continue;
    }

    output_detections->emplace_back(detection);
  }
  return absl::OkStatus();
}

Detection GestureTensorsToDetectionsCalculator::ConvertToDetection(
    float box_ymin, float box_xmin, float box_ymax, float box_xmax, float score,
    int class_id, bool flip_vertically) {
  Detection detection;
  detection.add_score(score);
  detection.add_label_id(class_id);

  LocationData* location_data = detection.mutable_location_data();
  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);

  LocationData::RelativeBoundingBox* relative_bbox =
      location_data->mutable_relative_bounding_box();

  relative_bbox->set_xmin(box_xmin);
  relative_bbox->set_ymin(flip_vertically ? 1.f - box_ymax : box_ymin);
  relative_bbox->set_width(box_xmax - box_xmin);
  relative_bbox->set_height(box_ymax - box_ymin);
  return detection;
}

}  // namespace api2
}  // namespace mediapipe
