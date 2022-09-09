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
constexpr char kGrayOutTag[] = "GRAY_OUT";
}  // namespace

class BackgroundExtractorCalculator : public CalculatorBase {
 public:
  ~BackgroundExtractorCalculator() override = default;

  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Process(CalculatorContext* cc) override;

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

 private:  
  absl::Status ConvertAndOutput(const std::string& input_tag,
                                const std::string& output_tag,
                                ImageFormat::Format output_format,
                                int open_cv_convert_code,
                                CalculatorContext* cc);
};

REGISTER_CALCULATOR(BackgroundExtractorCalculator);

absl::Status BackgroundExtractorCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK_EQ(cc->Inputs().NumEntries(), 1) << "Only one input stream is allowed.";
  RET_CHECK_EQ(cc->Outputs().NumEntries(), 1) << "Only one output stream is allowed.";

  if (cc->Inputs().HasTag(kRgbInTag)) {
    cc->Inputs().Tag(kRgbInTag).Set<mediapipe::ImageFrame>();
  }

  if (cc->Outputs().HasTag(kRgbOutTag)) {
    cc->Outputs().Tag(kRgbOutTag).Set<mediapipe::ImageFrame>();
  }

  return absl::OkStatus();
}

absl::Status BackgroundExtractorCalculator::Process(CalculatorContext* cc) {
  const cv::Mat& input_mat = formats::MatView(&cc->Inputs().Tag(kRgbInTag).Get<ImageFrame>());

  std::unique_ptr<ImageFrame> grayFrame( new ImageFrame(ImageFormat::GRAY8, input_mat.cols, input_mat.rows) );
  cv::Mat gray_mat = formats::MatView(grayFrame.get());

  std::unique_ptr<ImageFrame> rgbFrame( new ImageFrame(ImageFormat::SRGB, input_mat.cols, input_mat.rows) );
  cv::Mat rgb_mat = formats::MatView(rgbFrame.get());

  cv::cvtColor(input_mat, gray_mat, cv::COLOR_RGB2GRAY);
  cv::cvtColor(gray_mat, rgb_mat, cv::COLOR_GRAY2RGB);  

  cc->Outputs().Tag(kRgbOutTag).Add(rgbFrame.release(), cc->InputTimestamp());  
    
  return absl::OkStatus();  
}


}  // namespace mediapipe
