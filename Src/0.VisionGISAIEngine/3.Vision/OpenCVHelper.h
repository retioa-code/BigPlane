#pragma once

#include "../1.Common/HeaderFiles.h"
#include "../1.Common/MacroEnumStruct.h"

namespace VisionGISAIEngine {

    VGA_API void SaveShowMatchResult(Mat &src_image, Mat &dst_mage, std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> &key_points, std::string path, bool need_to_show = false);

    VGA_API void CalculateHomographyMat(std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> key_points_pair, cv::Mat &homography_mat);

    VGA_API bool BigBufferUYVYToRGB(unsigned char *yuv_buffer, unsigned char *rgb_buffer, int cols, int rows);
}