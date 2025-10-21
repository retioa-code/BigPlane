#include "PlatformVariable.h"


RunningStatusEnum RunningStatus;
QString LogFolderPath = "";
Mat BaseDomMat;
Mat BaseDemMat;
double DomGeoTransform[6] = {1, 1, 1, 1, 1, 1};
double DemGeoTransform[6] = {1, 1, 1, 1, 1, 1};
double MinWorkHeight = -1;
double MaxWorkHeight = -1;


int CameraVerticalFov = 57;
int CameraImageCols = 2880;
int CameraImageRows = 1860;
int UsedImageCols = 1860;  // 使用多大的图像区域用于生成实时图，考虑wrj可能按任意航向角飞行，设置长宽相等
int UsedImageRows = 1860;  // 若无人机可正东西向或正南北向飞行，这几个变量的值，可另行设计
int PaddingImageCols = 864;
int PaddingImageRows = 864;
int AtomImageCols = 800;
int AtomImageRows = 800;


//QString ChangedAreaRGB = "254_254_0";
//QString GroundEndIp = "-1";
//int MinAreaThreshold = -1;
//bool CompressJpgImage = false;
//int CompressQuality = 11;
//double BarometerHeightAdjust = 1;

//QString CurrentDateTimeStr = "";
//bool OnlyGatherImage = false;
//bool OnlyGatherGpsRtk = false;
//bool OnlyUnderImage = false;
//double MinWorkLongitude =-1;
//double MaxWorkLongitude=-1;
//double MinWorkLatitude=-1;
//double MaxWorkLatitude=-1;
//double HeightTolerance = 41;
//int MinWorkHeight = 801;
//int MaxWorkHeight = 801;
//double LiftoffBarometerHeight = 0;
//double LiftoffElevation = 0;
//bool NeedCheckHomography = false;
//double ZoomThreshold = 1.01;
//double ShiftThreshold = 1;
//double SkewThreshold = 0.01;
//int UavPodAngle = 0;
//double GsdLongitude = -1;
//double GsdLatitude = -1;

