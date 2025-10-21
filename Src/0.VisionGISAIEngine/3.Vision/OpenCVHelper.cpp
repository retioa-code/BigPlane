#include "OpenCVHelper.h"
#include "../1.Common/EngineFunction.h"

void VisionGISAIEngine::SaveShowMatchResult(Mat &src_image, Mat &dst_mage, std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> &key_points, std::string path, bool need_to_show) {
    if (!NeedSaveDebugInfo()) return;
    cv::Mat concat_image;
    hconcat(src_image, dst_mage, concat_image);
    line(concat_image, Point(0, 2), Point(concat_image.cols, 2), Scalar(0, 255, 255), 3, LINE_AA);
    line(concat_image, Point(concat_image.cols / 2, 0), Point(concat_image.cols / 2, concat_image.rows), Scalar(0, 255, 255), 3, LINE_AA);
    for (int index = 0; index < key_points.first.size(); ++index) {
        auto pt0 = key_points.first[index];
        auto pt1 = key_points.second[index];
        pt1 = cv::Point(pt1.x + src_image.cols, pt1.y);
        circle(concat_image, pt0, 2, cv::Scalar(0, 255, 0), -1);
        circle(concat_image, pt1, 2, cv::Scalar(0, 255, 0), -1);
        line(concat_image, pt0, pt1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }
    imwrite(path, concat_image);
    if (!need_to_show) return;
    imshow("MatchResult", concat_image);
    cv::waitKey(0);
}

void VisionGISAIEngine::CalculateHomographyMat(std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> key_points_pair, cv::Mat &homography_mat) {
    auto first = key_points_pair.first;
    auto second = key_points_pair.second;
    if (first.size() < 5 || first.size() != second.size()) {
        homography_mat = cv::Mat();
        return;
    }
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;
    for (int index = 0; index < first.size(); ++index) {
        obj.push_back(cv::Point2f(first[index].x, first[index].y));
        scene.push_back(cv::Point2f(second[index].x, second[index].y));
    }
    homography_mat = findHomography(cv::Mat(obj), cv::Mat(scene), cv::RANSAC);
//    // ʹ�õ�Ӧ�Ծ����ĳ��ͼ���еĵ������ص����͸�ӱ任���ݲ�ɾ��
//    std::vector<Point2f> obj_corners(5), scene_corners(5);
//    obj_corners[0] = Point(0, 0);
//    obj_corners[1] = Point(image_object.cols, 0);
//    obj_corners[2] = Point(image_object.cols, image_object.rows);
//    obj_corners[3] = Point(0, image_object.rows);
//    obj_corners[4] = Point(image_object.cols / 2, image_object.rows / 2);
//    perspectiveTransform(obj_corners, scene_corners, H);
//    auto point = Point(int(scene_corners[4].x), int(scene_corners[4].y));
//    circle(image_scene, point, 6, Scalar(255, 0, 0), 4, LINE_AA);
//    imwrite(("H:/PythonCpp/Result/" + temp + ".jpg").toLocal8Bit().data(), image_scene);
}

// 通用的 uyvy 到 rgb 格式转换，转换大图
bool BigBufferUYVYToRGB(unsigned char *yuv_buffer, unsigned char *rgb_buffer, int cols, int rows) {
    int numPixels = cols * rows / 2;
    for (int index = 0; index < numPixels; index++) {
        auto yuv_start_ptr = yuv_buffer + index * 4;
        auto rgb_start_ptr = rgb_buffer + index * 6;
        auto U = *(yuv_start_ptr) - 128;
        auto Y1 = *(yuv_start_ptr + 1);
        auto V = *(yuv_start_ptr + 2) - 128;
        auto Y2 = *(yuv_start_ptr + 3);
        auto R1 = static_cast<int>(Y1 + 1.4075 * V);
        auto G1 = static_cast<int>(Y1 - 0.3455 * U - 0.7169 * V);
        auto B1 = static_cast<int>(Y1 + 1.779 * U);
        auto R2 = static_cast<int>(Y2 + 1.4075 * V);
        auto G2 = static_cast<int>(Y2 - 0.3455 * U - 0.7169 * V);
        auto B2 = static_cast<int>(Y2 + 1.779 * U);
        *(rgb_start_ptr + 0) = static_cast<unsigned char>((std::min)((std::max)(R1, 0), 255));
        *(rgb_start_ptr + 1) = static_cast<unsigned char>((std::min)((std::max)(G1, 0), 255));
        *(rgb_start_ptr + 2) = static_cast<unsigned char>((std::min)((std::max)(B1, 0), 255));
        *(rgb_start_ptr + 3) = static_cast<unsigned char>((std::min)((std::max)(R2, 0), 255));
        *(rgb_start_ptr + 4) = static_cast<unsigned char>((std::min)((std::max)(G2, 0), 255));
        *(rgb_start_ptr + 5) = static_cast<unsigned char>((std::min)((std::max)(B2, 0), 255));
    }
    return true;
}
