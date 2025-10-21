// 几个图像匹配算法函数
#ifndef MATCHALGORITHM_H
#define MATCHALGORITHM_H
#include "HeaderFiles.h"

class MatchAlgorithm
{
public:
	static cv::Point CrossCoGradXY(cv::Mat& big_image, cv::Mat& small_image);
	static cv::Mat CalculateHogFeature(cv::Mat& image);
	static cv::Point MatchByHogAndTemplate(cv::Mat& big_image, cv::Mat& big_hog_mat, cv::Mat& small_image, cv::Mat small_hog_mat);
	static cv::Point MatchByGradIntensity(cv::Mat& big_image, cv::Mat& small_image);
	static cv::Point MatchByGradDirection(cv::Mat& big_image, cv::Mat& small_image);
	static double CalcCorelativeCoefficient(QList<double> list1, QList<double> list2);
	static cv::Point MatchByOpenCVTemplate(cv::Mat& big_image, cv::Mat& small_image);
#ifdef PlatformIsWindows
	static cv::Point MatchAlgorithm::MatchByHuMoment(cv::Mat& Big_image, cv::Mat& small_image);
	static cv::Point2f MatchBySift(cv::Mat image_scene, cv::Mat image_object);
	static cv::Point2f MatchBySurf(cv::Mat image_scene, cv::Mat image_object);
	static cv::Point2f MatchByOrb(cv::Mat image_scene, cv::Mat image_object);
	static cv::Point2f MatchByAkaza(cv::Mat image_scene, cv::Mat image_object);
	static cv::Point2f MatchByBrisk(cv::Mat image_scene, cv::Mat image_object);
#endif
};


#endif // MATCHALGORITHM_H
