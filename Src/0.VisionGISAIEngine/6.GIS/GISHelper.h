#pragma once

#include "../1.Common/HeaderFiles.h"
#include "../1.Common/MacroEnumStruct.h"

namespace VisionGISAIEngine {

    VGA_API StdVector2 CalcGroundToAir(double ground_height, StdVector3 pos_posture);

    VGA_API void PixelToGeography(int pixel_col, int pixel_row, double &geography_x, double &geography_y, double geo_transform[6]);

    VGA_API void GeographyToPixel(double geography_x, double geography_y, int &pixel_col, int &pixel_row, double geo_transform[6]);

    VGA_API bool ConvertGauss6ToLLH(double gauss_x, double gauss_y, double &out_longitude_angle, double &out_latitude_angle);

    VGA_API bool ConvertLLHToGauss6(double longitude_angle, double latitude_angle, double &out_gauss_x, double &out_gauss_y, int iBand = -1);

    VGA_API void ConvertLLHToGauss3(double longitude, double latitude, double &gauss3_x, double &gauss3_y);

    VGA_API void ConvertGauss3ToLLH(double gauss3_x, double gauss3_y, double &longitude, double &latitude);

    VGA_API double GetElevation(double longitude, double latitude, Mat &dem_mat, double geo_transform[6]);

    VGA_API void SavePositioningPoint(QString shape_file_path, QList<QMap<QString, QString>> point_list);

    VGA_API void ConvertGSDToDegrees(double gsd, double latitude, double &gsdLongitude, double &gsdLatitude);

// 判断偏航角 yaw 是否在横、纵两个坐标轴的正负 yaw_tolerance 度以内
    VGA_API bool IsYawWithinAxesRange(double yaw, double yaw_tolerance);

    VGA_API void PointListToShp(QList<QPair<QString, QPointF>> point_list, QString &shp_file_path);

    VGA_API void ConvertFileNameToShp(QString &dir_path, QString &file_suffix, QString &shp_file_path);

    VGA_API bool IsGaussKruger3Degree(GDALDataset* dataset);

    VGA_API bool IsWGS84(GDALDataset* dataset);


}