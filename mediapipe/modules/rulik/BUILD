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

cc_library(
    name = "rulik_gpu_deps",
    deps = [
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:mux_calculator_v1",
        "//mediapipe/calculators/core:packet_cloner_calculator",

        "//mediapipe/calculators/image:image_transformation_calculator",
        # "//mediapipe/calculators/image:recolor_calculator_v1",
        # "//mediapipe/calculators/image:color_convert_calculator",
        "//mediapipe/calculators/image:background_extractor_calculator",
        "//mediapipe/calculators/image:image_clone_calculator",
        
        "//mediapipe/calculators/util:from_image_calculator",
        "//mediapipe/calculators/util:to_image_calculator",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
        "//mediapipe/gpu:image_frame_to_gpu_buffer_calculator",
 
        "//mediapipe/framework/stream_handler:sync_set_input_stream_handler",
        # "//mediapipe/modules/selfie_segmentation:selfie_segmentation_gpu",
        # "//mediapipe/modules/palm_detection:palm_detection_gpu",
        # "//mediapipe/modules/palm_detection:palm_detection_gpu_image",
        # "//mediapipe/modules/face_detection:face_detection_full_range_gpu",
        # "//mediapipe/modules/face_detection:face_detection_short_range_image",
        # "//mediapipe/modules/face_detection:face_detection_full_range_image",
        "//mediapipe/modules/rulik_gesture:rulik_gesture_image",
    ],
)

mediapipe_simple_subgraph(
    name = "rulik_gpu",
    graph = "rulik_gpu.pbtxt",
    register_as = "RulikGpu",
    deps = [":rulik_gpu_deps"],
)

mediapipe_simple_subgraph(
    name = "rulik_gpu_v1",
    graph = "rulik_gpu_v1.pbtxt",
    register_as = "RulikGpuV1",
    deps = [":rulik_gpu_deps"],
)

mediapipe_simple_subgraph(
    name = "rulik_gpu_v2",
    graph = "rulik_gpu_v2.pbtxt",
    register_as = "RulikGpuV2",
    deps = [
        ":rulik_gpu_deps",
        "//mediapipe/modules/rulik_gesture:rulik_gesture_image",
    ],
)

mediapipe_simple_subgraph(
    name = "rulik_gpu_v3",
    graph = "rulik_gpu_v3.pbtxt",
    register_as = "RulikGpuV3",
    deps = [
        ":rulik_gpu_deps",
        "//mediapipe/modules/rulik_gesture:rulik_gesture_image_v3",
    ],
)