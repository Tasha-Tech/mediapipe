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
constexpr char kCommand[] = "COMMAND";
}  // namespace

class BackgroundExtractorCalculator : public CalculatorBase {
 public:  
  int output_in_gray = 0;

  ~BackgroundExtractorCalculator() override = default;
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

};

REGISTER_CALCULATOR(BackgroundExtractorCalculator);

absl::Status BackgroundExtractorCalculator::GetContract(CalculatorContract* cc) {  
  RET_CHECK_EQ(cc->Outputs().NumEntries(), 1) << "Only one output stream is allowed.";

  if (cc->Inputs().HasTag(kRgbInTag)) {
    cc->Inputs().Tag(kRgbInTag).Set<mediapipe::ImageFrame>();
  }

  if (cc->Inputs().HasTag(kCommand)) {
    cc->Inputs().Tag(kCommand).Set<int>().Optional();
  }

  if (cc->Outputs().HasTag(kRgbOutTag)) {
    cc->Outputs().Tag(kRgbOutTag).Set<mediapipe::ImageFrame>();
  }

  return absl::OkStatus();
}

absl::Status BackgroundExtractorCalculator::Process(CalculatorContext* cc) {

  if (cc->Inputs().HasTag(kCommand) && !cc->Inputs().Tag(kCommand).IsEmpty()) {
    int command = cc->Inputs().Tag(kCommand).Get<int>();
    //std::cout << "select " << select << std::endl;
    if(command == 1){
      output_in_gray = !output_in_gray;
    }
  }

  if(cc->Inputs().Tag(kRgbInTag).IsEmpty()) {
    return absl::OkStatus();
  }

  const ImageFrame& inputFrame = cc->Inputs().Tag(kRgbInTag).Get<ImageFrame>();
  const cv::Mat& input_mat = formats::MatView(&inputFrame);
  ImageFormat::Format format = inputFrame.Format();
  //std::cout << "format " << format << std::endl;

  //const cv::Mat& input_mat = formats::MatView(&cc->Inputs().Tag(kRgbInTag).Get<ImageFrame>());
  
  std::unique_ptr<ImageFrame> outputFrame( new ImageFrame(format /*ImageFormat::SRGBA*/, input_mat.cols, input_mat.rows) );  
  cv::Mat output_mat = formats::MatView(outputFrame.get());  

  if(output_in_gray){
    //std::unique_ptr<ImageFrame> grayFrame( new ImageFrame(ImageFormat::GRAY8, input_mat.cols, input_mat.rows) );
    //cv::Mat gray_mat = formats::MatView(grayFrame.get());
    cv::Mat gray_mat;
    cv::cvtColor(input_mat, gray_mat, cv::COLOR_RGB2GRAY);

    //std::unique_ptr<ImageFrame> smallGrayFrame( new ImageFrame(ImageFormat::GRAY8, input_mat.cols/4, input_mat.rows/4) );
    cv::Mat small_gray_mat;
    cv::resize(gray_mat, small_gray_mat, cv::Size(input_mat.cols/4, input_mat.rows/4));

    cv::Mat dx, dy;
    cv::spatialGradient(small_gray_mat, dx, dy);

    cv::convertScaleAbs(dx, dx);
    cv::convertScaleAbs(dy, dy);
    cv::addWeighted(dx, 0.5, dy, 0.5, 0, small_gray_mat);

    cv::resize(small_gray_mat, gray_mat, gray_mat.size());

    cv::cvtColor(gray_mat, output_mat, cv::COLOR_GRAY2RGBA);  
  } else {
    input_mat.copyTo(output_mat);
  }

  cc->Outputs().Tag(kRgbOutTag).Add(outputFrame.release(), cc->InputTimestamp());  
    
  return absl::OkStatus();  
}


}  // namespace mediapipe