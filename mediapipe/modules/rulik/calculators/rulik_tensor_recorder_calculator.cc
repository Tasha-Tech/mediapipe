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
//   calculator: "RulikTensorRecorderCalculator"
//   input_stream: "TENSORS:tensors"
//   output_stream: "DETECTIONS:detections"
// }
class RulikTensorRecorderCalculator : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kInTensors{"TENSORS"};
  

  MEDIAPIPE_NODE_CONTRACT(kInTensors);  

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status ProcessCPU(CalculatorContext* cc);
    
};

MEDIAPIPE_REGISTER_NODE(RulikTensorRecorderCalculator);

absl::Status RulikTensorRecorderCalculator::Open(CalculatorContext* cc) {  
  return absl::OkStatus();
}

absl::Status RulikTensorRecorderCalculator::Process(CalculatorContext* cc) {
  auto output_detections = absl::make_unique<std::vector<Detection>>();  
 
  MP_RETURN_IF_ERROR(ProcessCPU(cc));  
  
  return absl::OkStatus();
}

absl::Status RulikTensorRecorderCalculator::ProcessCPU(CalculatorContext* cc) {
  const auto& input_tensors = *kInTensors(cc);

  //std::cout << "Received " << input_tensors.size() << " tensors" << std::endl;

  auto tensor = &input_tensors[0];    
  auto tensor_view = tensor->GetCpuReadView();
  auto tensor_buffer = tensor_view.buffer<float>();

  FILE *fp;
  fp = fopen("tensor.dat","w");
  for(int i = 0; i < tensor->shape().num_elements(); i++){
    fprintf(fp, "%.3f ", tensor_buffer[i]);
  }

  fclose(fp);
  
  return absl::OkStatus();
}


absl::Status RulikTensorRecorderCalculator::Close(CalculatorContext* cc) {
  return absl::OkStatus();
}

}  // namespace api2
}  // namespace mediapipe
