// 和图像处理相关的函数
#ifndef PUBLIC_H
#define PUBLIC_H
//#include <object.h>

#include "HeaderFiles.h"

cv::Mat ShuffleRows(const cv::Mat& image);
cv::Mat MeanFilter(cv::Mat& image, int nBlurSizesar_realtime_mat);
cv::Mat AddGaussAddNoise(cv::Mat& imagesar_realtime_mat);
bool ImageHaveBlackArea(cv::Mat& src_mat);
QMap<QString, QString> CalcImageFeature(cv::Mat& realtime_image, cv::Mat& big_image, cv::Mat& small_image, int real_row, int real_col);
void CalcFourierFeature(cv::Mat& big_image, QStringList& feature_key, QMap<QString, QString>& feature_key_value);
void CalcGLCMFeature(cv::Mat& big_image, QStringList& feature_key, QMap<QString, QString>& feature_key_value);
void CalcGGCMFeature(cv::Mat& big_image, QStringList& feature_key, QMap<QString, QString>& feature_key_value);
void CalcCorrelativeSurfaceFeature(cv::Mat& realtime_image, cv::Mat& big_image_grad, cv::Mat& small_image_grad, int real_row, int real_col, QStringList& feature_key, QMap<QString, QString>& feature_key_value);
void OutputResultImage(QString image_path, QString csv_path, ExecuteFlowEnum execute_flow);
//void OutputCrossWire(cv::Mat& input_image, QMap<QPair<int, int>, QString> first_result,
//	QMap<QPair<int, int>, QString> second_result, QMap<QPair<int, int>, QString> third_result);
//void AddMaskRectangleToImage(cv::Mat& image, cv::Rect roi_rect, cv::Scalar scalar, double alpha = 0.22);
void StretchContrast(cv::Mat& input_image, cv::Mat& output_image, double stretch_density_threshold);
void OutputRemoveSmallPieces(QString original_image_path, cv::Mat& cover_image);
cv::Mat ConnectiveFilter(cv::Mat& image, long pixel_count_threshold);
void OutputHeatMapResult(QString original_image_path, cv::Mat& color_map, QTextStream& text_stream, int tlwh_index, int predict_probability_index, ExecuteFlowEnum execute_flow);
void OutputSuitabilityVector(QString image_path, QTextStream* text_stream,int ltwh_index, int predict_probability_index, ExecuteFlowEnum execute_flow);
void AddCoordinateReference(QString from_image, QString to_image);

QList<QPair<int, int> > GetPositionIndex(int count_x, int count_y,QString image_path, ExecuteFlowEnum execute_flow, 
	double optical_geo_transform[6], double sar_geo_transform[6], cv::Mat sar_opencv_mat, QList<cv::Rect>& sar_rect_list);
bool ExtractSingleImage(QString optical_image_path, ExecuteFlowEnum execute_flow, QString sar_image_path="");
bool ExtractSingleBlock(QString image_path, cv::Mat optical_opencv_mat,QString sar_image_path, cv::Mat sar_opencv_mat, QString csv_path, ExecuteFlowEnum execute_flow, int start_y = 0);
void ExecuteParallelly(cv::Mat optical_opencv_mat, cv::Mat sar_opencv_mat, QList<QPair<int, int> > col_row_list, int margin_x, int margin_y, int start_y,
	ExecuteFlowEnum execute_flow, QTextStream* text_stream, QList<cv::Rect> sar_rect_list);
void ExecuteInOrder(cv::Mat optical_opencv_mat, cv::Mat sar_opencv_mat, QList<QPair<int, int> > col_row_list, int margin_x, int margin_y, int start_y,
	ExecuteFlowEnum execute_flow, QTextStream* text_stream, QList<cv::Rect> sar_rect_list);
QString ExtractSingleSlidingWindow(cv::Mat& optical_opencv_mat, cv::Mat& sar_opencv_mat, int col, int row, int margin_x, int margin_y, int start_y, ExecuteFlowEnum execute_flow,cv::Rect rect = cv::Rect(-1,-1,1,1));
//QgsVectorLayer* ConvertToMemoryLayer(QgsVectorLayer* vector_layer);
QString MatchImagePair(cv::Mat& small_image, cv::Mat& big_image, cv::Mat& assistant_mat, QString col_row,QMap<QString,QString>& feature_key_value);
cv::Point MatchByAlgorithm(cv::Mat& big_image, cv::Mat& rotate_resize_image, float& max_peak, float& sub_peak, float& max_sub_ratio);
// 不删，这个是接口形式
//void MatchByGradient(uchar* pImgBig, uchar* pImgSmall, int widthBig, int heightBig, int widthSmall, int heightSmall, int matchX, int matchY, float* pCorrelation, float* pBuffer);
#endif // PUBLIC_H
