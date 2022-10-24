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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/source_location.h"
#include "mediapipe/framework/port/status_builder.h"

#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"

namespace mediapipe {
namespace {
void SetColorChannel(int channel, uint8 value, cv::Mat* mat) {
  CHECK(mat->depth() == CV_8U);
  CHECK(channel < mat->channels());
  const int step = mat->channels();
  for (int r = 0; r < mat->rows; ++r) {
    uint8* row_ptr = mat->ptr<uint8>(r);
    for (int offset = channel; offset < mat->cols * step; offset += step) {
      row_ptr[offset] = value;
    }
  }
}

constexpr char kRgbInTag[] = "RGB_IN";
constexpr char kRgbOutTag[] = "RGB_OUT";
constexpr char kCommandTag[] = "COMMAND";
constexpr char kDetectionsTag[] = "DETECTIONS";
}  // namespace


typedef enum mode_e{
  MODE_FIRST        = 0,
  MODE_NORMAL       = MODE_FIRST,
  MODE_BLENDER      = 1,
  MODE_CALIBRATION  = 2,
  MODE_LAST         = 3
} mode_t;

typedef enum command_e{
  CMD_NONE                  = 0,
  CMD_SET_MODE_NORMAL       = 1,
  CMD_SET_MODE_BLENDER      = 2,
  CMD_SET_MODE_CALIBRATION  = 3,
  CMD_SET_NEXT_MODE         = 4,
  CMD_SET_LEFT              = 5,
  CMD_SET_RIGHT             = 6,
  CMD_MAIN_ACTION           = 7,  
}command_t;

class BackgroundExtractorCalculator : public CalculatorBase {
 public:    
  cv::Mat bg_mat;
  cv::Mat bg_mat_small;
  cv::Mat film_mat;
  
  int mode = MODE_CALIBRATION; // 1 = blender
  int command = CMD_NONE;

  ~BackgroundExtractorCalculator() override = default;
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Normal(CalculatorContext* cc);
  absl::Status Blender(CalculatorContext* cc);
  absl::Status Calibration(CalculatorContext* cc);

  absl::Status PhotoBooth(CalculatorContext* cc);
};

REGISTER_CALCULATOR(BackgroundExtractorCalculator);

absl::Status BackgroundExtractorCalculator::GetContract(CalculatorContract* cc) {  
  RET_CHECK_EQ(cc->Outputs().NumEntries(), 1) << "Only one output stream is allowed.";

  if (cc->Inputs().HasTag(kRgbInTag)) {
    cc->Inputs().Tag(kRgbInTag).Set<mediapipe::ImageFrame>();
  }

  if (cc->Inputs().HasTag(kCommandTag)) {
    cc->Inputs().Tag(kCommandTag).Set<int>().Optional();
  }

  if (cc->Inputs().HasTag(kDetectionsTag)) {
    cc->Inputs().Tag(kDetectionsTag).Set<std::vector<mediapipe::Detection>>().Optional();
  }

  if (cc->Outputs().HasTag(kRgbOutTag)) {
    cc->Outputs().Tag(kRgbOutTag).Set<mediapipe::ImageFrame>();
  }

  return absl::OkStatus();
}

absl::Status BackgroundExtractorCalculator::Process(CalculatorContext* cc) {  
  if (cc->Inputs().HasTag(kCommandTag) && !cc->Inputs().Tag(kCommandTag).IsEmpty()) {
    command = cc->Inputs().Tag(kCommandTag).Get<int>();
    switch(command){
      case CMD_SET_MODE_NORMAL:
        mode = MODE_NORMAL;
        break;
      case CMD_SET_MODE_BLENDER:
        mode = MODE_BLENDER;
        break;
      case CMD_SET_MODE_CALIBRATION:
        mode = MODE_CALIBRATION;
        break;
      case CMD_SET_NEXT_MODE:
        mode++;
        if(mode >= MODE_LAST){
          mode = MODE_FIRST;
        }
    }
  }

  if(cc->Inputs().Tag(kRgbInTag).IsEmpty()) {
    return absl::OkStatus();
  }

  switch(mode){
    case MODE_NORMAL:
    Normal(cc);
    break;
    case MODE_BLENDER:
    Blender(cc);
    break;    
    case MODE_CALIBRATION:
    Calibration(cc);
    break;    
  }

  command = CMD_NONE;
  return absl::OkStatus();  
}

absl::Status BackgroundExtractorCalculator::Normal(CalculatorContext* cc) {
  const ImageFrame& inputFrame = cc->Inputs().Tag(kRgbInTag).Get<ImageFrame>();
  const cv::Mat& input_mat = formats::MatView(&inputFrame);
  ImageFormat::Format format = inputFrame.Format();

  std::unique_ptr<ImageFrame> outputFrame( new ImageFrame(format /*ImageFormat::SRGBA*/, input_mat.cols, input_mat.rows) );  
  cv::Mat output_mat = formats::MatView(outputFrame.get());  

  input_mat.copyTo(output_mat);  
  cc->Outputs().Tag(kRgbOutTag).Add(outputFrame.release(), cc->InputTimestamp());  
  return absl::OkStatus();    
}

absl::Status BackgroundExtractorCalculator::Calibration(CalculatorContext* cc) {
  const ImageFrame& inputFrame = cc->Inputs().Tag(kRgbInTag).Get<ImageFrame>();
  const cv::Mat& input_mat = formats::MatView(&inputFrame);
  ImageFormat::Format format = inputFrame.Format();
  
  if(bg_mat.empty()){
    input_mat.copyTo(bg_mat);
    bg_mat.setTo(cv::Scalar::all(0));
    cv::resize(bg_mat, bg_mat_small, cv::Size(input_mat.cols/4, input_mat.rows/4));
  }

  if(command == CMD_SET_LEFT || command == CMD_SET_RIGHT) {
    int x = 0;
    if(command == CMD_SET_RIGHT){
      x = input_mat.cols/2;
    }
    
    cv::Mat input_roi = input_mat(cv::Rect(x, 0,input_mat.cols/2, input_mat.rows));
    cv::Mat bg_roi = bg_mat(cv::Rect(x, 0, input_mat.cols/2, input_mat.rows));
    input_roi.copyTo(bg_roi);
    // We also need to update bg_mat_small
    cv::resize(bg_mat, bg_mat_small, cv::Size(input_mat.cols/4, input_mat.rows/4));
  }

  cv::Mat diff_mat;
  cv::absdiff(input_mat, bg_mat, diff_mat);
  cv::cvtColor(diff_mat, diff_mat, cv::COLOR_RGB2GRAY);
  
  cv::Mat mask(diff_mat.size(), CV_8UC1);
  mask.setTo(0);
  cv::threshold(diff_mat, mask, 15, 1, cv::THRESH_BINARY);


  std::unique_ptr<ImageFrame> outputFrame( new ImageFrame(format /*ImageFormat::SRGBA*/, input_mat.cols, input_mat.rows) );  
  cv::Mat output_mat = formats::MatView(outputFrame.get());  
  output_mat.setTo(cv::Scalar::all(0));
  
  input_mat.copyTo(output_mat, mask);  
  cc->Outputs().Tag(kRgbOutTag).Add(outputFrame.release(), cc->InputTimestamp());  
  return absl::OkStatus();    
}

// Multiple exposure mode
absl::Status BackgroundExtractorCalculator::Blender(CalculatorContext* cc) {
  const ImageFrame& inputFrame = cc->Inputs().Tag(kRgbInTag).Get<ImageFrame>();
  const cv::Mat& input_mat = formats::MatView(&inputFrame);
  ImageFormat::Format format = inputFrame.Format();  

  cv::Mat diff_mat;
  cv::absdiff(input_mat, bg_mat, diff_mat);
  cv::cvtColor(diff_mat, diff_mat, cv::COLOR_RGB2GRAY);
  
  cv::Mat mask(diff_mat.size(), CV_8UC1);
  mask.setTo(0);
  cv::threshold(diff_mat, mask, 20, 1, cv::THRESH_BINARY);

  if(film_mat.empty()){
    bg_mat.copyTo(film_mat);
  } 

  if(command == CMD_MAIN_ACTION) {
    std::cout << "Updating film ..." << std::endl;
    input_mat.copyTo(film_mat, mask);  
  } else if (command == CMD_SET_RIGHT) {
    bg_mat.copyTo(film_mat);
  }  

  std::unique_ptr<ImageFrame> outputFrame( new ImageFrame(format /*ImageFormat::SRGBA*/, input_mat.cols, input_mat.rows) );  
  cv::Mat output_mat = formats::MatView(outputFrame.get());  
  
  film_mat.copyTo(output_mat);
  input_mat.copyTo(output_mat, mask);

  cc->Outputs().Tag(kRgbOutTag).Add(outputFrame.release(), cc->InputTimestamp());
  return absl::OkStatus();  
}  

// Multiple exposure mode
absl::Status BackgroundExtractorCalculator::PhotoBooth(CalculatorContext* cc) {
  const ImageFrame& inputFrame = cc->Inputs().Tag(kRgbInTag).Get<ImageFrame>();
  const cv::Mat& input_mat = formats::MatView(&inputFrame);
  ImageFormat::Format format = inputFrame.Format();  

  cv::Mat input_mat_small;
  cv::resize(input_mat, input_mat_small, cv::Size(input_mat.cols/4, input_mat.rows/4));
  cv::Mat diff_mat;
  cv::absdiff(input_mat_small, bg_mat_small, diff_mat);
  cv::cvtColor(diff_mat, diff_mat, cv::COLOR_RGB2GRAY);

  cv::Mat dx, dy;
  cv::spatialGradient(diff_mat, dx, dy);
  cv::convertScaleAbs(dx, dx);
  cv::convertScaleAbs(dy, dy);

  cv::Mat output_mat_small;
  cv::addWeighted(dx, 0.5, dy, 0.5, 0, output_mat_small);

  cv::cvtColor(output_mat_small, output_mat_small, cv::COLOR_GRAY2RGBA);

  std::unique_ptr<ImageFrame> outputFrame( new ImageFrame(format /*ImageFormat::SRGBA*/, input_mat.cols, input_mat.rows) );  
  cv::Mat output_mat = formats::MatView(outputFrame.get());  
  cv::resize(output_mat_small, output_mat, output_mat.size());
  cc->Outputs().Tag(kRgbOutTag).Add(outputFrame.release(), cc->InputTimestamp());
  return absl::OkStatus();  
}  

} // mediapipe namespace



#if 0
  //BGR spliting:
  std::vector<cv::Mat> bgrChannels(3);
  cv::split( colorInput, bgrChannels );
  //Mask every channel:
  cv::bitwise_and( bgrChannels[0], imageMask, bgrChannels[0] ); //B
  cv::bitwise_and( bgrChannels[1], imageMask, bgrChannels[1] ); //G
  cv::bitwise_and( bgrChannels[2], imageMask, bgrChannels[2] ); //R
  //Merge back the channels
  cv::merge( bgrChannels, maskedImage );
#endif




#if 0
    cv::Mat gray_mat;
    cv::cvtColor(input_mat, gray_mat, cv::COLOR_RGB2GRAY);
    
    cv::Mat small_gray_mat;
    cv::resize(gray_mat, small_gray_mat, cv::Size(input_mat.cols/4, input_mat.rows/4));

    cv::Mat dx, dy;
    cv::spatialGradient(small_gray_mat, dx, dy);

    cv::convertScaleAbs(dx, dx);
    cv::convertScaleAbs(dy, dy);
    cv::addWeighted(dx, 0.5, dy, 0.5, 0, small_gray_mat);

    cv::resize(small_gray_mat, gray_mat, gray_mat.size());

    cv::cvtColor(gray_mat, output_mat, cv::COLOR_GRAY2RGBA);
    /*
    if (cc->Inputs().HasTag(kDetectionsTag) &&  !cc->Inputs().Tag(kDetectionsTag).Value().IsEmpty()) {
      const auto& detections = cc->Inputs().Tag(kDetectionsTag).Get<std::vector<mediapipe::Detection>>();
    
      for (const auto& detection : detections) {
        const auto& score = detection.score();
        const auto& location = detection.location_data();
        const auto& relative_bounding_box = location.relative_bounding_box();
        //std::cout << "Score " << score[0] << std::endl;
        int x = relative_bounding_box.xmin() * output_mat.cols;
        int y = (1.0 - relative_bounding_box.ymin()) * output_mat.rows;
        int width = relative_bounding_box.width() * output_mat.cols;
        int height = relative_bounding_box.height() * output_mat.rows;
        y -= height;
        cv::Rect rect(x, y, width, height);
        cv::rectangle(output_mat, rect, cv::Scalar(0, 0, 255), 3);
        //char text[255];
        //std::sprintf(text,"%0.2f",score[0]);
        //putText(output_mat, text, cv::Point(x + 10, y + height / 2), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
      }
    }
    */
#endif