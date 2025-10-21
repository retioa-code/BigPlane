#pragma once

#include "PlatformVariable.h"

namespace UavVisionPlatform {

    bool LoadGuaranteedData(QString dom_path, QString dem_path);

    bool LoadGuaranteedData(QString guaranteed_data_dir);

    void CreateReferenceMultiple(string realtime_dir, string reference_dir, QString dom_path);

    bool CreateReferenceSingle(StdVector3 position, StdVector3 orientation, cv::Mat &reference_image, int &clip_left, int &clip_top);

    VGA_API bool InitializeLiftOff();

}