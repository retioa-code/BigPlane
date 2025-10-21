#pragma once

#include "../../0.VisionGISAIEngine/1.Common/HeaderFiles.h"
#include "../../0.VisionGISAIEngine/1.Common/MacroEnumStruct.h"

extern RunningStatusEnum RunningStatus;
extern QString LogFolderPath;
extern Mat BaseDomMat;
extern Mat BaseDemMat;
extern double DomGeoTransform[6];
extern double DemGeoTransform[6];
extern double MinWorkHeight;
extern double MaxWorkHeight;

extern int CameraVerticalFov;
extern int CameraImageRows;
extern int CameraImageCols;
extern int CameraImageRows;
extern int UsedImageCols;
extern int UsedImageRows;
extern int PaddingImageCols;
extern int PaddingImageRows;
extern int AtomImageCols;
extern int AtomImageRows;



//extern QString ChangedAreaRGB;
//extern QString GroundEndIp;
//extern int MinAreaThreshold;
//extern bool CompressJpgImage;
//extern int CompressQuality;
//extern double BarometerHeightAdjust;

//extern QString CurrentDateTimeStr;
//extern bool OnlyGatherImage;
//extern bool OnlyGatherGpsRtk;
//extern bool OnlyUnderImage;
//extern double MinWorkLongitude;
//extern double MaxWorkLongitude;
//extern double MinWorkLatitude;
//extern double MaxWorkLatitude;
//extern double HeightTolerance;
//extern int MinWorkHeight;
//extern int MaxWorkHeight;
//extern double LiftoffBarometerHeight;
//extern double LiftoffElevation;
//extern bool NeedCheckHomography;
//extern double ZoomThreshold;
//extern double ShiftThreshold;
//extern double SkewThreshold;
//extern int UavPodAngle;
//extern double GsdLongitude;
//extern double GsdLatitude;
