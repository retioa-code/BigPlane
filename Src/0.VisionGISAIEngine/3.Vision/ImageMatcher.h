#pragma once

#include "../1.Common/HeaderFiles.h"
#include "../1.Common/MacroEnumStruct.h"

namespace VisionGISAIEngine {

    VGA_API void MatchByTemplate(cv::Mat &big_image, cv::Mat &small_image, cv::Point &left_top, cv::Point &center, double &confidence, cv::Mat &coor_surface, double down_sample_factor = 1);

    VGA_API void MatchByGradCorrelative(cv::Mat &big_image, cv::Mat &small_image, cv::Point &left_top, cv::Point &center, double &confidence);

}