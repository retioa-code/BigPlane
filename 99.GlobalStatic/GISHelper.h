// 和地理信息系统相关的函数
#pragma once
#include "HeaderFiles.h"

class GISHelper
{
public:
	static void OpenImage(QString image_file_path, QgsMapCanvas* map_canvas);
	static void CloseImage(QgsMapCanvas* map_canvas);

	void OpenShapeFile(QString shp_file_path, QgsMapCanvas* map_canvas);
	static void CreateCache(QString file_path, QgsMapCanvas* map_canvas);
	static void GetCoordinateSystemInfo(QString image_path, double geo_transform[6],QString& projection_ref);
	static void PixelPointToGeographyPoint(int pixel_col, int pixel_row, double& geography_x, double& geography_y, double geo_transform[6]);
	static void GeographyPointToPixelPoint(double geography_x, double geography_y, int& pixel_col, int& pixel_row, double geo_transform[6]);
	static void ReadImage(QString optical_image_path, cv::Mat& optical_image_mat, cv::Mat& sar_image_mat,
		GDALDataset*& optical_gdal_dataset, GDALDataset*& sar_gdal_dataset, ExecuteFlowEnum execute_flow, QString sar_image_path = "");

	static void ClipImage(QString source_path, double geo_left, double geo_bottom, double geo_right, double geo_top, QString& result_path);
	static int ResampleImage(const char* pszSrcFile, const char* pszOutFile, float fResX, float fResY, GDALResampleAlg eResample);
	static void SaveImageByGDAL(const QString &file_name, void* image, int width, int height, int band);
};
