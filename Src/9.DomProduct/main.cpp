# pragma execution_character_set("utf-8")
#include "0.Platform/PlatformFunctions.h"
#include "8.Mapper/Mapper.h"
#include "6.GIS/GISHelper.h"

using namespace VisionGISAIEngine;
using namespace UavVisionPlatform;

int main(int argc, char *argv[]) {
    QCoreApplication application(argc, argv);

    Mapper mapper;
    QString path="E:/991.35JD/FlyExperiment/FlyTest1212/00Log_001850";
    QString suffix="*.raw";
    QString shp_path="H:/TempCanDelete00/Temp11/xx.shp";
    VisionGISAIEngine::ConvertFileNameToShp(path, suffix, shp_path);

    return application.exec();
}