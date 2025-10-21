#include "ImageMatcher.h"

    void VisionGISAIEngine::MatchByTemplate(cv::Mat &big_image, cv::Mat &small_image, cv::Point &left_top, cv::Point &center, double &confidence, cv::Mat &coor_surface, double down_sample_factor) {
        if (down_sample_factor != 1) {
            resize(big_image, big_image, Size(int(big_image.cols / down_sample_factor), int(big_image.rows / down_sample_factor)));
            resize(small_image, small_image, Size(int(small_image.cols / down_sample_factor), int(small_image.rows / down_sample_factor)));
        }
        matchTemplate(big_image, small_image, coor_surface, cv::TM_CCOEFF_NORMED);
        double max_value, min_value;
        cv::Point max_point, min_point;
        minMaxLoc(coor_surface, &min_value, &max_value, &min_point, &max_point);
        left_top = cv::Point(max_point.x, max_point.y) * down_sample_factor;
        center = cv::Point(max_point.x + small_image.cols / 2, max_point.y + small_image.rows / 2) * down_sample_factor;
        confidence = max_value;
    }

    void VisionGISAIEngine::MatchByGradCorrelative(cv::Mat &big_image, cv::Mat &small_image, cv::Point &left_top, cv::Point &center, double &confidence) {
        cv::Mat sobelXMat;
        cv::Mat sobelYMat;
        Sobel(big_image, sobelXMat, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_REFLECT);
        sobelXMat = abs(sobelXMat);
        Sobel(big_image, sobelYMat, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_REFLECT);
        sobelYMat = abs(sobelYMat);

        cv::Mat tempSobelXMat;
        cv::Mat tempSobelYMat;
        Sobel(small_image, tempSobelXMat, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_REFLECT);
        tempSobelXMat = abs(tempSobelXMat);
        Sobel(small_image, tempSobelYMat, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_REFLECT);
        tempSobelYMat = abs(tempSobelYMat);

        cv::Mat crossXMat;
        cv::Mat crossYMat;
        matchTemplate(sobelXMat, tempSobelXMat, crossXMat, cv::TM_CCOEFF_NORMED);
        matchTemplate(sobelYMat, tempSobelYMat, crossYMat, cv::TM_CCOEFF_NORMED);
        cv::Mat resultMat = (crossXMat + crossYMat) / 2;

        double max_value, min_value;
        cv::Point max_point, min_point;
        minMaxLoc(resultMat, &min_value, &max_value, &min_point, &max_point);
        left_top = cv::Point(max_point.x, max_point.y);
        center = cv::Point(max_point.x + small_image.cols / 2, max_point.y + small_image.rows / 2);
        confidence = max_value;
    }
