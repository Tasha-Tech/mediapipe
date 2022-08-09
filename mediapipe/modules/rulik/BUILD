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
    name = "rulik_gpu_image",
    graph = "rulik_gpu_image.pbtxt",
    register_as = "RulikGpuImage",
    deps = [        
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/util:from_image_calculator",
        "//mediapipe/calculators/util:to_image_calculator",
        "//mediapipe/calculators/image:recolor_calculator",        
        "//mediapipe/modules/palm_detection:palm_detection_gpu",
        "//mediapipe/modules/selfie_segmentation:selfie_segmentation_gpu",        
    ],
)
