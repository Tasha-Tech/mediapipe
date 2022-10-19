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
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
// This example requires a linux computer and a GPU with EGL support drivers.
#include <cstdlib>
#include <dirent.h>
#include <vector>
#include <iomanip>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/detection.pb.h"

#include "mediapipe/framework/formats/tensor.h"

#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

constexpr char kInputStream[] = "input_video";
constexpr char kSelector[] = "output_selector";
constexpr char kOutputStream[] = "output_video";
constexpr char kOutputPalmDetections[] = "palm_detections";
constexpr char kOutputGestureDetections[] = "gesture_detections";

constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, input_image_path, "",
          "Full path of image to load. "
          "If not provided, attempt to use a webcam.");

ABSL_FLAG(std::string, input_image_dir, "",
          "Full path of image directory. "
          "If not provided, attempt to use a webcam.");

ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");

absl::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      absl::GetFlag(FLAGS_calculator_graph_config_file),
      &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Initialize the GPU.";
  ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
  MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
  mediapipe::GlCalculatorHelper gpu_helper;
  gpu_helper.InitializeForTest(graph.GetGpuResources().get());

  LOG(INFO) << "Initialize the camera or load the video.";
  cv::VideoCapture capture;

  int current_image = 0;
  const bool load_image = !absl::GetFlag(FLAGS_input_image_path).empty();
  const bool load_dir = !absl::GetFlag(FLAGS_input_image_dir).empty();
  const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
  bool use_capture = true;
  if (load_video) {
    capture.open(absl::GetFlag(FLAGS_input_video_path));
  } else if(load_image || load_dir){
    use_capture = false;
  }else {
    capture.open(0);    
  }
  if(use_capture) {
    RET_CHECK(capture.isOpened());
  }

  cv::VideoWriter writer;
  const bool save_video = !absl::GetFlag(FLAGS_output_video_path).empty();
  if (!save_video) {
    cv::namedWindow(kWindowName, 0 /*WINDOW_NORMAL*/ /*flags=WINDOW_AUTOSIZE 1*/);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    if(use_capture){
      //capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
      //capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
      capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
      capture.set(cv::CAP_PROP_FRAME_HEIGHT, 360);
      capture.set(cv::CAP_PROP_FPS, 30);
    }
#endif
  }

  LOG(INFO) << "Start running the calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller, graph.AddOutputStreamPoller(kOutputStream));

  #define ENABLE_PALM_DETECTONS 1
  #if ENABLE_PALM_DETECTONS
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller palm_detections_poller, graph.AddOutputStreamPoller(kOutputPalmDetections));
  palm_detections_poller.SetMaxQueueSize(3);
  #endif /* ENABLE_PALM_DETECTONS */

  #define ENABLE_GESTURE_DETECTONS 1
  #if ENABLE_GESTURE_DETECTONS
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller gesture_detections_poller, graph.AddOutputStreamPoller(kOutputGestureDetections));
  gesture_detections_poller.SetMaxQueueSize(3);
  #endif /* ENABLE_GESTURE_DETECTONS */

  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;
  mediapipe::Timestamp program_timestamp = mediapipe::Timestamp::Unstarted();
  int64 color = 0;
  mediapipe::Timestamp color_timestamp = mediapipe::Timestamp(0);
  int selector = 0;
  mediapipe::Timestamp select_timestamp = mediapipe::Timestamp(0);  

  std::vector<std::string> images;

  cv::Mat camera_frame_raw;
  if(load_image){
    images.push_back(absl::GetFlag(FLAGS_input_image_path));
  }
  
  if(load_dir) {
    DIR *d;
    struct dirent *dir;
    d = opendir(absl::GetFlag(FLAGS_input_image_dir).c_str());
    if (d) {
      while ((dir = readdir(d)) != NULL) {
        if(strlen(dir->d_name) > 3){
          images.push_back(absl::GetFlag(FLAGS_input_image_dir) + "/" + std::string(dir->d_name));
          printf("%s\n", dir->d_name);
        }        
      }
      closedir(d);
    }
  }
  if(!use_capture) {
    camera_frame_raw = cv::imread(images[current_image]);
    printf("Current image: %s \n", images[current_image].c_str());
    if(++current_image >= images.size()) {
      current_image = 0;
    }
  }
  //return absl::OkStatus();
  

  while (grab_frames) {
    // Capture opencv camera or video frame.    
    if(use_capture) {
      capture >> camera_frame_raw;
    }
    if (camera_frame_raw.empty()) {
      if (!load_video) {
        LOG(INFO) << "Ignore empty frames from camera.";
        continue;
      }
      LOG(INFO) << "Empty frame, end of video reached.";      
      break;            
    }

    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGBA);
    if (!load_video) {
      cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    }

    cv::Mat input_frame_mat;
    std::unique_ptr<mediapipe::ImageFrame> input_frame;
    if(use_capture) {
      // Wrap Mat into an ImageFrame.
      input_frame = absl::make_unique<mediapipe::ImageFrame>(
          mediapipe::ImageFormat::SRGBA, 1280, 720, 
          mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);

      input_frame_mat = mediapipe::formats::MatView(input_frame.get());
      cv::resize(camera_frame, input_frame_mat, input_frame_mat.size());
    } else {
      const int cropSizeX = camera_frame.cols;
      const int cropSizeY = camera_frame.rows;

      // Wrap Mat into an ImageFrame.      
      input_frame = absl::make_unique<mediapipe::ImageFrame>(
          mediapipe::ImageFormat::SRGBA, cropSizeX, cropSizeY, //camera_frame.cols, camera_frame.rows, // 1280x720
          mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);

      input_frame_mat = mediapipe::formats::MatView(input_frame.get());
      
      
      const int offsetW = (camera_frame.cols - cropSizeX) / 2;
      const int offsetH = (camera_frame.rows - cropSizeY) / 2;
      const cv::Rect roi(offsetW, offsetH, cropSizeX, cropSizeY);
      camera_frame = camera_frame(roi).clone();

      camera_frame.copyTo(input_frame_mat);
    }
    

    // Prepare and add graph input packet.
    size_t frame_timestamp_us = (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    mediapipe::Timestamp ts_ = mediapipe::Timestamp(frame_timestamp_us);


    MP_RETURN_IF_ERROR(
        gpu_helper.RunInGlContext([&input_frame, &frame_timestamp_us, &graph,
                                   &gpu_helper, &ts_]() -> absl::Status {
          // Convert ImageFrame to GpuBuffer.
          auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
          auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
          glFlush();
          texture.Release();
          // Send GPU image packet into the graph.
          MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
              kInputStream, mediapipe::Adopt(gpu_frame.release()).At(ts_)));          
          return absl::OkStatus();
        }));

    if(program_timestamp == mediapipe::Timestamp::Unstarted()){      
      program_timestamp = mediapipe::Timestamp(frame_timestamp_us);    
    }    

    mediapipe::TimestampDiff diff = mediapipe::Timestamp(frame_timestamp_us) - program_timestamp;
    if(!use_capture){
      //graph.AddPacketToInputStream(kSelector, mediapipe::MakePacket<int>(1).At(++select_timestamp));
    } else if(diff.Seconds() > 3){
      //graph.AddPacketToInputStream(kSelector, mediapipe::MakePacket<int>(1).At(++select_timestamp));

      program_timestamp = mediapipe::Timestamp(frame_timestamp_us);
      if(color == 0) {
        color = 1;
      } else {
        color = 0;
      }
    }

   // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    if (!poller.Next(&packet)) break;
    std::unique_ptr<mediapipe::ImageFrame> output_frame;       

    // Convert GpuBuffer to ImageFrame.
    MP_RETURN_IF_ERROR(gpu_helper.RunInGlContext(
        [&packet, &output_frame, &gpu_helper]() -> absl::Status {
          auto& gpu_frame = packet.Get<mediapipe::GpuBuffer>();
          auto texture = gpu_helper.CreateSourceTexture(gpu_frame);
          output_frame = absl::make_unique<mediapipe::ImageFrame>(
              mediapipe::ImageFormatForGpuBufferFormat(gpu_frame.format()),
              gpu_frame.width(), gpu_frame.height(),
              mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
          gpu_helper.BindFramebuffer(texture);
          const auto info = mediapipe::GlTextureInfoForGpuBufferFormat(
              gpu_frame.format(), 0, gpu_helper.GetGlVersion());
          glReadPixels(0, 0, texture.width(), texture.height(), info.gl_format,
                       info.gl_type, output_frame->MutablePixelData());
          glFlush();
          texture.Release();
          return absl::OkStatus();
        }));

    // Convert back to opencv for display or saving.
    cv::Mat output_frame_mat = mediapipe::formats::MatView(output_frame.get());
    if (output_frame_mat.channels() == 4)
      cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGBA2BGR);
    else
      cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    
    #if ENABLE_PALM_DETECTONS
    // Get the graph result packets, or stop if that fails.
    if(palm_detections_poller.QueueSize() > 0){                   
      mediapipe::Packet detections_packet;
      while(palm_detections_poller.QueueSize() > 0){
        palm_detections_poller.Next(&detections_packet);
      }
      
      const auto& detections = detections_packet.Get<std::vector<mediapipe::Detection>>();    
      for (const auto& detection : detections) {
        const auto& score = detection.score();
        const auto& location = detection.location_data();
        const auto& relative_bounding_box = location.relative_bounding_box();
        for(int i = 0; i < detection.label_id_size(); i++){          
          std::cout << "Palm Score " << score[i] << std::endl;
        }
        int x = relative_bounding_box.xmin() * output_frame_mat.cols;
        int y = (/* 1.0 - */relative_bounding_box.ymin()) * output_frame_mat.rows;
        int width = relative_bounding_box.width() * output_frame_mat.cols;
        int height = relative_bounding_box.height() * output_frame_mat.rows;
        //y -= height;
        cv::Rect rect(x, y, width, height);
        //cv::rectangle(output_frame_mat, rect, cv::Scalar(255, 0, 0), 3);
        char text[255];
        std::sprintf(text,"%0.2f",score[0]);
        //putText(output_frame_mat, text, cv::Point(x + 10, y + height / 2), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
      }
    }    
    #endif /* ENABLE_PALM_DETECTONS */

    #if ENABLE_GESTURE_DETECTONS
    if(!use_capture || gesture_detections_poller.QueueSize() > 0){
      std::cout << "QueueSize " << gesture_detections_poller.QueueSize() << std::endl;
      mediapipe::Packet detections_packet;
      
      if(!use_capture){
        gesture_detections_poller.Next(&detections_packet);
      } else {
        while(gesture_detections_poller.QueueSize() > 0){
          gesture_detections_poller.Next(&detections_packet);
        }
      }

      if(!detections_packet.IsEmpty()) {
        const auto& detections = detections_packet.Get<std::vector<mediapipe::Detection>>();      
        for (const auto& detection : detections) {
          const auto& score = detection.score();
          const auto& location = detection.location_data();
          const auto& relative_bounding_box = location.relative_bounding_box();
          float max_score = score[0];
          float delta = 0;
          int max_index = 0;
          std::vector<std::string> label = { "fist_left", "fist_right", "like_left", "like_right", "no_gesture", "peace_left", "peace_right", "stop_left", "stop_right" };

          std::cout << "Scores " << std::fixed << std::setprecision(2);
          
          for(int i = 0; i < detection.label_id_size(); i++){
            if(max_score < score[i]) {
              delta = score[i] - max_score;
              max_score = score[i];
              max_index = i;
            }
            std::cout << label[i] << ": " << score[i] << ", ";
          }
          std::cout << std::endl;

          int x = relative_bounding_box.xmin() * output_frame_mat.cols;
          int y = (/* 1.0 - */relative_bounding_box.ymin()) * output_frame_mat.rows;
          int width = relative_bounding_box.width() * output_frame_mat.cols;
          int height = relative_bounding_box.height() * output_frame_mat.rows;
          //y -= height;
          cv::Rect rect(x, y, width, height);
          //cv::rectangle(output_frame_mat, rect, cv::Scalar(0, 255, 0), 3);
          char text[255];
          if(max_index != 4) {
            std::sprintf(text,"%s %0.2f ", label[max_index].c_str(), max_score);
            putText(output_frame_mat, text, cv::Point(x + 10, y + height / 2), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
          }
        }   
      }   
    }

    #endif /* ENABLE_GESTURE_DETECTONS */

    cv::resize(output_frame_mat, output_frame_mat, cv::Size(1280, 720));
    cv::imshow(kWindowName, output_frame_mat);
    // Press 'Esc' key to exit.
    int pressed_key;
    if(!use_capture){
      pressed_key = cv::waitKey(0);
      if(pressed_key == 83 || pressed_key == 32) {         
        current_image++;
      } else if (pressed_key == 81) {
        current_image--;
      }
      if(current_image < 0){
        current_image = images.size() - 1;
      }
      if(current_image > images.size() - 1){
        current_image = 0;
      }        
      camera_frame_raw = cv::imread(images[current_image]);
      printf("Current image: %s \n", images[current_image].c_str());
    } else {
      pressed_key = cv::waitKey(30);
    }

    if (pressed_key == 27) {
      grab_frames = false;
    }     
  } //while (grab_frames) 

  LOG(INFO) << "Shutting down.";
  if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));  
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kSelector));
  
  graph.WaitUntilDone();
  //cv::waitKey(500);
  return absl::OkStatus();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  absl::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
