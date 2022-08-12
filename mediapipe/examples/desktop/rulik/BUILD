licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

# Linux only
cc_binary(
    name = "rulik_gpu",
    deps = [
        "//mediapipe/examples/desktop:rulik_run_graph_main_gpu",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
        "//mediapipe/modules/rulik:rulik_gpu",
    ],
)


# bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11   mediapipe/examples/desktop/rulik:rulik_gpu
# GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/rulik/rulik_gpu --calculator_graph_config_file=mediapipe/graphs/rulik/rulik_desktop_gpu.pbtxt
