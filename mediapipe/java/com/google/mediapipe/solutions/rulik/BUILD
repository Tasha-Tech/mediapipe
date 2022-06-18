# Copyright 2021 The MediaPipe Authors.
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

licenses(["notice"])

android_library(
    name = "rulik",
    srcs = [
    ],
    assets = [
        "//mediapipe/modules/pose_landmark:pose_landmark_full.tflite",
        "//mediapipe/modules/pose_detection:pose_detection.tflite",
        "//mediapipe/modules/pose_landmark:pose_landmark_cpu.binarypb",
        "//mediapipe/modules/pose_landmark:pose_landmark_gpu_image.binarypb",
    ],
    assets_dir = "",
    javacopts = ["-Acom.google.auto.value.AutoBuilderIsUnstable"],
    manifest = ":AndroidManifest.xml",
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/framework/formats:classification_java_proto_lite",
        "//mediapipe/framework/formats:landmark_java_proto_lite",
        "//mediapipe/java/com/google/mediapipe/framework:android_framework",
        "//mediapipe/java/com/google/mediapipe/solutioncore:solution_base",
        "//third_party:autovalue",
        "@maven//:androidx_annotation_annotation",
        "@maven//:com_google_code_findbugs_jsr305",
        "@maven//:com_google_guava_guava",
    ],
)


load("//mediapipe/java/com/google/mediapipe:mediapipe_aar.bzl", "mediapipe_aar")

cc_binary(
    name = "libmediapipe_jni.so",
    linkshared = 1,
    linkstatic = 1,
    deps = [
        "//mediapipe/graphs/pose_tracking:pose_tracking_gpu_deps",
        "//mediapipe/modules/pose_landmark:pose_landmark_gpu_image",
        "//mediapipe/java/com/google/mediapipe/framework/jni:mediapipe_framework_jni",
    ],
)

cc_library(
    name = "mediapipe_jni_lib",
    srcs = [":libmediapipe_jni.so"],
    alwayslink = 1,
)

mediapipe_aar(
    name = "rulik_aar",
    #srcs = glob(["**/*.java"]),
    gen_libmediapipe = False,
    assets = [
        "//mediapipe/modules/pose_landmark:pose_landmark_full.tflite",
        "//mediapipe/modules/pose_detection:pose_detection.tflite",
        "//mediapipe/modules/pose_landmark:pose_landmark_gpu_image.binarypb",
    ],
    assets_dir = "",
)