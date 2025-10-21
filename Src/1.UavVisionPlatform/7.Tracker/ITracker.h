#pragma once

#include "../../0.VisionGISAIEngine/1.Common/HeaderFiles.h"
#include "../../0.VisionGISAIEngine/1.Common/MacroEnumStruct.h"

namespace UavVisionPlatform {

    class ITracker {
        void ReadPlatformLocationConfig();

        void SetParamDebugging(QString work_dir, string id, Mat &realtime_rgb_image, Mat &padding_reference_image, int &clip_left, int &clip_top, StdVector3 &position, StdVector3 &orientation);

        bool GpsRtkIsOn = true;
        StdVector3 PreviousPosition = StdVector3(0, 0, 0);
        StdVector3 PreviousOrientation = StdVector3(0, 0, 0);
        QString WorkDir = "";  //仅室内调试用
        QString DomPath = "";  //仅室内调试用
        double DownSampleFactor = 1.5;

    public:

        UVP_API void Initialize(QString dom_path, QString dem_path, QString work_dir);

        UVP_API void Initialize(QString guaranteed_data_dir, QString work_dir);

        UVP_API void Track(cv::Mat &realtime_image, StdVector3 position, StdVector3 orientation);
    };

}