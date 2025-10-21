// 存放提取特征的函数
#pragma once
#include <opencv2/opencv.hpp>
using namespace std;

struct CorrelativeSurfaceStruct
{
	float MainOrMaxPeak;
	float SubPeakValue;
	float MainSubPeakRatio;
	float MainSubPeakDifference;
	float MainPeakSharpness;
	int RepeatMode;
	int MainOrMaxPeakCol;
	int MainOrMaxPeakRow;
};

void CalcMeanStdDev(cv::Mat image, double& mmean, double& std_dev);
double CalcMaxMinLocalStdDev(cv::Mat& image, double& max_std_dev, double& min_std_dev);
double CalcHogEntropy(cv::Mat& image, cv::Size block_stride, cv::Size block_size, cv::Size cell_size, int bins);
double CalcEdgeDensity(cv::Mat& image, cv::Mat gradient_mat, int contour_length_threshold = 30);
double CalcEdgeDensity2(cv::Mat& image, int contour_length_threshold);
double CalcMinLocalEdgeDensity(cv::Mat& image, cv::Mat& gradient_mat, int contour_length_threshold=12);
cv::Mat GetLOGKernel(int ksize, double sigma);
double CalcSingleImageMSE(cv::Mat& image);
double CalcTwoImageMSE(cv::Mat& source_image, cv::Mat& target_image);
double CalcTwoImagePSNR(cv::Mat& source_image, cv::Mat& target_image);
double CalcSingleImagePSNR(cv::Mat& image);
double CalcAbsoluteRough(cv::Mat& image);
double CalcLaplacianEdgeMean(cv::Mat& image);
//double CalcGrayEntropy(cv::Mat& image);
double CalcGrayEntropy_2(cv::Mat& image);
double CalcFriedenEntropy(cv::Mat& image);
void CalcSobelGradXY(cv::Mat& image, cv::Mat& sobel_x, cv::Mat& sobel_y, bool need_absolute = true);
double CalcDirectionGradEntropy(cv::Mat& image);
double CalcIndependentPixels(cv::Mat& image);
double CalcZeroCrossDensity2(cv::Mat& image);
double CalcTextureComplexity(cv::Mat& image);
void CalcHuMoment(cv::Mat& image, map<string, double>& feature_key_value);
void CalcFourierVeinsPercent(cv::Mat& image, std::map<std::string, float>& feature_key_value);
void CalcGrayDifference(cv::Mat& image, std::map<std::string, double>& feature_key_value);
void CalcGrayHistogramMoment(cv::Mat& image, std::map<std::string, double>& feature_key_value);
void CalcGLCM(cv::Mat& image, map<string, double>& feature_key_value);
cv::Mat ConvertToFourier(cv::Mat& image);
cv::Mat GetFourierMask(int col, int row);
cv::Mat FrequencyFilter(cv::Mat& fourier, int cols, int rows);
void CalcGradByDifference(cv::Mat& image, cv::Mat& grad_x, cv::Mat& grad_y, bool is_absolute = false);
double CalcENL(cv::Mat& image);
double CalcGrayContrast(cv::Mat& image);
void CalcGradientFeature(cv::Mat gradient_image, map<string, double>& feature_key_value);
double CalcDefinition(cv::Mat& image);
double CalcSingnalNoiseRatio(cv::Mat& image);
double CalcRobertGradientMean(cv::Mat& image);
int CalcOrbKeyPointCount(cv::Mat& big_image);
//double CalcSiftMatchedPointCount(cv::Mat& big_image, cv::Mat& small_image);
//double CalcSurfMatchedPointCount(cv::Mat& scene_image, cv::Mat& object_image);
//double CalcOrbMatchedPointCount(cv::Mat& big_image, cv::Mat& small_image);
void CalcGGCM(cv::Mat& image, map<string, double>& feature_key_value);
double CalcDistortion(cv::Mat& source_image, cv::Mat& target_image);
double CalcGradientCovariance(cv::Mat& source_image, cv::Mat& target_image);
double CalcDeviationIndex(cv::Mat& image, cv::Mat& target_image);
double CalcCorrelationCoefficient(cv::Mat& source_image, cv::Mat& target_image);
cv::Mat CalcHistogram(cv::Mat& image, int min_value, int max_value, bool need_normed = true);
double CalcCrossEntropy(cv::Mat& image, cv::Mat& target_image);
double CalcGraySSIM(cv::Mat& source_image, cv::Mat& target_image);
double CalcGradientSSIM(cv::Mat& source_image, cv::Mat& target_image);
double CalcEdgeKeepingIndex(cv::Mat& source_image, cv::Mat& target_image);
double ComEntropy(cv::Mat& matImage1, cv::Mat& matImage2);
double NormMutualInformation(double dEntropy1, double dEntropy2, double dComEntropy);
//子图唯一性系数
double SubImageUniqueness(cv::Mat& matSmallImage, cv::Mat& matBigImage, cv::Mat& matBigEntropy);
//整个基准图唯一性系数
double CalUniquenessCoefficient(cv::Mat& matRefImage, int nRealWidth, int nRealHeight);
cv::Mat CalcGradientIntensity(cv::Mat& image);
cv::Mat CalcGradientDirection(cv::Mat& matImage);
cv::Mat CalcGradAngleImage(cv::Mat grad_x, cv::Mat grad_y);
cv::Mat CalcHistograph(cv::Mat& grayImage);
//0交叉点密度
double CalcZeroCrossDensity1(cv::Mat& matImage);
//角点密度
double CalCornerPointsDensity(cv::Mat& matImage);
//关键点强度
void CalcKeyPoint(cv::Mat& matImage, std::map<std::string, float>& feature_key_value);
void CalcGradCorrelativeSurface(cv::Mat& big_image_grad, cv::Mat& small_image_grad, int real_row, int real_col, std::map<std::string, float>& feature_key_value);
void CalcGradCorrelativeSurface(cv::Mat& image, std::map<std::string, float>& feature_key_value);
void CalcMainPeakSubPeak(cv::Mat& correlative_surface_image, cv::Rect truth_value_region_rect, float& main_peak_value, int& main_peak_col, int& main_peak_row, float& sub_peak_value, float& main_sub_ratio);
void CalcMaxPeakSecondPeak(float* correlation_surface, int cols, int rows, float& max_peak, float& second_peak, float& max_second_ratio);
void CalcMaxPeakSecondPeak(cv::Mat correlation_surface, float& max_peak, float& second_peak, float& max_second_ratio);
std::map<float, std::pair<int, int> > FindPeaks(cv::Mat image);
void CalcRepeatMode(cv::Mat& correlative_surface, float peak_value_threshold, std::map<std::pair<int, int>, float>& peak_list, cv::Mat& flag_mat, int roi_width_half, int roi_height_half);
double CalcMainPeakSharpness(cv::Mat& correlative_surface, int col_index, int row_index);
void StretchContrast(cv::Mat& input_image, cv::Mat& output_image, double stretch_density_threshold);
void ReduceImageMass(cv::Mat original_image, cv::Mat& noise_image, double blur_size, double sigma, double noise_density);
cv::Mat AddGaussMultipleNoise(cv::Mat& image, double noise_stddev);
cv::Mat AddFourierNoise(cv::Mat& image);
void CalcMaxPeakSubPeak(float* correlation_surface, int cols, int rows, float& max_peak, float& sub_peak, float& max_sub_ratio);
void CalcLbp(cv::Mat& source_image, map<string, double>& key_value);