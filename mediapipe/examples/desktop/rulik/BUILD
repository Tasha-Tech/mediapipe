licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

# Linux only
cc_binary(
    name = "rulik_gpu",
    deps = [
        "//mediapipe/examples/desktop:rulik_run_graph_main_gpu",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
        "//mediapipe/modules/rulik:rulik_gpu_v1",
    ],
)

cc_binary(
    name = "rulik_gpu_v2",
    deps = [
        "//mediapipe/examples/desktop:rulik_run_graph_main_gpu_v2",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
        "//mediapipe/modules/rulik:rulik_gpu_v2",
    ],
)

cc_binary(
    name = "rulik_gpu_v3",
    deps = [
        "//mediapipe/examples/desktop:rulik_run_graph_main_gpu_v3",
        "//mediapipe/gpu:gpu_buffer_to_image_frame_calculator",
        "//mediapipe/modules/rulik:rulik_gpu_v3",
    ],
)


# bazel build -c opt --compilation_mode dbg --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11   mediapipe/examples/desktop/rulik:rulik_gpu_v2
# GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/rulik/rulik_gpu_v2 --calculator_graph_config_file=mediapipe/graphs/rulik/rulik_desktop_gpu_v2.pbtxt
# --input_image_path=/home/peter/work/RulikCamera2/python/Samples/3/JPEG_20220910_184622_8734468100911835426.jpg
# --input_image_path=/home/peter/work/RulikCamera2/python/Samples/3/JPEG_20220910_182916_5744066802982518760.jpg
# JPEG_20220910_183911_2501240202291940017.jpg
# JPEG_20220910_183237_1597562394550193403.jpg
# JPEG_20220909_194714_1645100728330327233.jpg
# JPEG_20220909_194904_8961579120081375466.jpg


# GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/rulik/rulik_gpu_v2 --calculator_graph_config_file=mediapipe/graphs/rulik/rulik_desktop_gpu_v2.pbtxt \
# --input_image_dir=/home/peter/work/RulikCamera2/python/Samples/4

# GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/rulik/rulik_gpu_v2 --calculator_graph_config_file=mediapipe/graphs/rulik/rulik_desktop_gpu_v2.pbtxt --input_image_path=/home/peter/work/RulikCamera2/python/Samples/4/JPEG_20220913_184757_5253500207203509635.jpg

# bazel build -c opt --compilation_mode dbg --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11   mediapipe/examples/desktop/rulik:rulik_gpu_v3
# GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/rulik/rulik_gpu_v3 --calculator_graph_config_file=mediapipe/graphs/rulik/rulik_desktop_gpu_v3.pbtxt