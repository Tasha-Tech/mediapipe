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
  int num_boxes_ = 8;
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

  //std::cout << "Received " << input_tensors.size() << " tensors" << std::endl;

  auto feature_map_tensor = &input_tensors[0];  
  auto ct_offset_tensor = &input_tensors[1];

  auto feature_map_view = feature_map_tensor->GetCpuReadView();
  auto feature_map = feature_map_view.buffer<float>();

  auto ct_offset_view = ct_offset_tensor->GetCpuReadView();  
  auto ct_offset = ct_offset_view.buffer<float>();

  int height = feature_map_tensor->shape().dims[1];
  int width = feature_map_tensor->shape().dims[2];
  int num_classes_ = feature_map_tensor->shape().dims[3];

#if 0
  std::vector<float> feature_map_peaks_flat(height * width * num_classes_);
  const float * feature_map_ptr = feature_map;  
  const float * feature_map_max_pool_ptr = feature_map_max_pool;
  for (auto peak = begin (feature_map_peaks_flat); peak != end (feature_map_peaks_flat); ++peak) {
    if(abs(*feature_map_ptr - *feature_map_max_pool_ptr) < 1e-6){
      *peak = *feature_map_ptr;
    } else {
      *peak = -1;
    }
    feature_map_ptr++;
    feature_map_max_pool_ptr++;
  }
#endif

  std::vector<int> top_k_indexes(num_boxes_);
  std::vector<float> top_k_scores(num_boxes_);
  
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
                      std::greater<std::pair<float, int>>>
      pq;

  for (int i = 0; i < height * width * num_classes_; ++i) {
    if (feature_map[i] < 0.1) {
      continue;
    }
    
    if (pq.size() < num_boxes_) {
      pq.push(std::pair<float, int>(feature_map[i], i));
    } else if (pq.top().first < feature_map[i]) {
      pq.pop();
      pq.push(std::pair<float, int>(feature_map[i], i));
    }    
  }

  while (!pq.empty()) {
    top_k_indexes.push_back(pq.top().second);
    top_k_scores.push_back(pq.top().first);
    pq.pop();
  }
  reverse(top_k_indexes.begin(), top_k_indexes.end());
  reverse(top_k_scores.begin(), top_k_scores.end());

  const float x_scale = 8.0 / 512.0;
  const float y_scale = 8.0 / 256.0; // 144.0; // 256 * (1080/1920) = 144
  const float aspect_ratio = 1920.0 / 1080.0;

  std::vector<float> boxes(num_boxes_ * num_coords_, 0.0);  
  std::vector<float> detection_scores(num_boxes_, 0.0);
  std::vector<int> detection_classes(num_boxes_,0);

  for (int i = 0; i < num_boxes_; ++i) {
    if(top_k_scores[i] < 1e-3){
      continue;
    }
    detection_scores[i] = top_k_scores[i]; //max_score;
    detection_classes[i] = top_k_indexes[i] % num_classes_; //class_id;
    int y = top_k_indexes[i] / num_classes_ / width;
    int x = top_k_indexes[i] / num_classes_ - (y * width);
    // ct_offset[y, x]
    
    const int box_offset = i * num_coords_;
    const float yoff = ct_offset[x*2 + y*2*width];
    const float xoff = ct_offset[x*2 + y*2*width + 1];

    //const float ymin = ((y + yoff) * 8 - 56.0) / 144.0; // (256 - 144) / 2 = 56
    const float ymin = (y + yoff) * y_scale;
    
    const float xmin = ((x + xoff) * 8 - 28.5) / 455.1;
    //const float xmin = (x + xoff) * x_scale;
    const float ymax = ymin + 0.1;
    const float xmax = xmin + 0.1;
    (boxes)[i * num_coords_ + 0] = ymin;
    (boxes)[i * num_coords_ + 1] = xmin;
    (boxes)[i * num_coords_ + 2] = ymax;
    (boxes)[i * num_coords_ + 3] = xmax;
  }

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