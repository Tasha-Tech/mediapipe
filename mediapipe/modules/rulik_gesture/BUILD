# Copyright 2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_simple_subgraph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])


mediapipe_simple_subgraph(
    name = "rulik_gesture_image",
    graph = "rulik_gesture_image.pbtxt",
    register_as = "RulikGestureImage",
    deps = [
        "//mediapipe/calculators/tensor:tensors_to_detections_calculator",
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/util:detection_projection_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",           
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        "//mediapipe/calculators/core:begin_loop_calculator",
        "//mediapipe/calculators/core:clip_vector_size_calculator",
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:end_loop_calculator",        
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/image:image_clone_calculator",
        "//mediapipe/calculators/util:association_norm_rect_calculator",
        "//mediapipe/calculators/util:collection_has_min_size_calculator",
        "//mediapipe/calculators/util:filter_collection_calculator",
        "//mediapipe/calculators/util:detection_classifications_merger_calculator",
        
        "//mediapipe/calculators/tensor:tensors_to_classification_calculator",

        "//mediapipe/modules/palm_detection:palm_detection_gpu",
        "//mediapipe/modules/hand_landmark:palm_detection_detection_to_roi",
    ],    
)

exports_files(
    srcs = [
        "ruliknet_v3_fp16.tflite",
        "rulikseg_v2e.tflite",
    ],
)

mediapipe_simple_subgraph(
    name = "rulik_gesture_image_v3",
    graph = "rulik_gesture_image_v3.pbtxt",
    register_as = "RulikGestureImageV3",
    deps = [
        # "//mediapipe/calculators/tensor:tensors_to_detections_calculator",
        # "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        # "//mediapipe/calculators/util:detection_projection_calculator",
        # "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",           
        "//mediapipe/calculators/tensor:image_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator",
        # "//mediapipe/calculators/core:begin_loop_calculator",
        # "//mediapipe/calculators/core:clip_vector_size_calculator",
        # "//mediapipe/calculators/core:constant_side_packet_calculator",
        # "//mediapipe/calculators/core:end_loop_calculator",        
        "//mediapipe/calculators/core:gate_calculator",
        # "//mediapipe/calculators/core:previous_loopback_calculator",
        # "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/image:image_clone_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        # "//mediapipe/calculators/util:association_norm_rect_calculator",
        # "//mediapipe/calculators/util:collection_has_min_size_calculator",
        # "//mediapipe/calculators/util:filter_collection_calculator",
        # "//mediapipe/calculators/util:detection_classifications_merger_calculator",
        
        # "//mediapipe/calculators/tensor:tensors_to_classification_calculator",

        # "//mediapipe/modules/palm_detection:palm_detection_gpu",
        # "//mediapipe/modules/hand_landmark:palm_detection_detection_to_roi",
        "//mediapipe/modules/rulik/calculators:gesture_tensors_to_detections_calculator",
        "//mediapipe/calculators/image:color_convert_calculator",
    ],    
)

