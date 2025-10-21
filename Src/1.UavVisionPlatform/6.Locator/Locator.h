#pragma once

#include "1.Common/HeaderFiles.h"
#include "1.Common/MacroEnumStruct.h"

namespace UavVisionPlatform {

    class Locator {
        void ReadPlatformLocationConfig();

        void SetParamDebugging(QString work_dir, string id, Mat &realtime_rgb_image, Mat &padding_reference_image, int &clip_left, int &clip_top, StdVector3 &position, StdVector3 &orientation);

        void ExecuteDebugging();

        StdVector3 Locate(string id, cv::Mat &realtime_image, cv::Mat &reference_image, int clip_left, int clip_top, double height, StdVector3 current_orientation);

        bool GpsRtkIsOn = true;
        StdVector3 PreviousPosition = StdVector3(0, 0, 0);
        StdVector3 PreviousOrientation = StdVector3(0, 0, 0);
        QString WorkDir = "";  //仅室内调试用
        QString DomPath = "";  //仅室内调试用
        double DownSampleFactor = 1.5;

    public:

        UVP_API void Initialize(QString dom_path, QString dem_path, QString work_dir = "");

        UVP_API StdVector3 Locate(cv::Mat realtime_image, double height, StdVector3 current_orientation);
    };

}