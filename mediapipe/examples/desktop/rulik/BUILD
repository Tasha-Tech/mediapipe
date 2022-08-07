licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

# Linux only
cc_binary(
    name = "rulik_gpu",
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main_gpu",
        "//mediapipe/graphs/hand_tracking:mobile_calculators",
        "//mediapipe/graphs/selfie_segmentation:selfie_segmentation_gpu_deps",
    ],
)
