# pragma execution_character_set("utf-8")
#include <iostream>
#include <QtCore/QJsonParseError>
#include <opencv2/opencv.hpp>
#include "../0.VisionGISAIEngine/1.Common/MacroEnumStruct.h"
#include "../0.VisionGISAIEngine/1.Common/EngineFunction.h"
#include "../0.VisionGISAIEngine/3.Vision/OpenCVHelper.h"
#include "../1.UavVisionPlatform/0.Platform/PlatformFunctions.h"
#include "../1.UavVisionPlatform/8.Detector/Detector.h"


using namespace VisionGISAIEngine;
using namespace UavVisionPlatform;

int InvokeInterval = 40;
QString WorkDir = "/data/sn/Code/WorkDir/DZK/";
QString GuaranteedDataDir = "/data/sn/Code/GuaranteedData/DZK";
QString DomPath = "";
QString DemPath = "";
int ImageRow=1860;
int ImageCol=2880;

void ReadProjectLocationConfig() {
    QString path = QCoreApplication::applicationDirPath() + "/ProjectConfig.json";
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        qDebug() << "Open config file failed: " + path + ". " << __LINE__ << "@" << __FILENAME__;
        exit(0);
    }
    QByteArray data = file.readAll();
    file.close();
    QJsonParseError parseError;
    QJsonDocument document = QJsonDocument::fromJson(data, &parseError);
    if (parseError.error != QJsonParseError::NoError) {
        qDebug() << "Parse config file failed: " + path + ". " << __LINE__ << "@" << __FILENAME__;
        exit(0);
    }
    if (!document.isObject()) {
        qDebug() << "Parse json Object failed: " + path + ". " << __LINE__ << "@" << __FILENAME__;
        exit(0);
    }
    QJsonObject root_object = document.object();
    QJsonObject location_param = root_object.value("LocationParam").toObject();
    InvokeInterval = location_param.value("InvokeInterval").toInt();

#ifdef PLATFORM_WINDOWS
    WorkDir = location_param.value("WorkDirWindows").toString();
    DomPath = location_param.value("DomPathWindows").toString();
    DemPath = location_param.value("DemPathWindows").toString();
#elif PLATFORM_RK3588
    WorkDir = location_param.value("WorkDirRK3588").toString();
    DomPath = location_param.value("DomPathRK3588").toString();
    DemPath = location_param.value("DemPathRK3588").toString();
#elif PLATFORM_JETSON
    WorkDir = location_param.value("WorkDirJetson").toString();
    GuaranteedDataDir = location_param.value("GuaranteedDataDirJetson").toString();
    DomPath = location_param.value("DomPathJetson").toString();
    DemPath = location_param.value("DemPathJetson").toString();
#endif
}

// ${projectDir}\..\Build
int main(int argc, char *argv[]) {
#ifdef PLATFORM_WINDOWS
    std::cout << "PLATFORM_WINDOWS=================TargetTrack" << std::endl;
#elif PLATFORM_RK3588
    std::cout << "PLATFORM_RK3588==================TargetTrack" << std::endl;
#elif PLATFORM_JETSON
    std::cout << "PLATFORM_JETSON==================TargetTrack" << std::endl;
#endif
    QCoreApplication application(argc, argv);
    ReadProjectLocationConfig();
    Detector detector;
    detector.Initialize(GuaranteedDataDir, WorkDir);

    for (auto &i: filesystem::directory_iterator((WorkDir+"0.raw").toStdString())) {
        auto path = i.path().string();
        if (".raw" != path.substr(path.size() - 4)) {
            continue;
        }

        auto image_total = CameraImageCols * CameraImageRows * 2;
        static uint8_t* realtime_yuv_buffer;
        if (nullptr == realtime_yuv_buffer)
        {
            realtime_yuv_buffer = new uint8_t[image_total];
        }
        auto fp = fopen(path.c_str(), "rb");
        fread(realtime_yuv_buffer, sizeof(uint8_t), image_total, fp);
        fclose(fp);

        Mat realtime_image = cv::Mat(ImageRow,ImageCol, CV_8UC3);
        BigBufferUYVYToRGB(realtime_yuv_buffer, realtime_image.data, ImageCol,ImageRow);
        cv::cvtColor(realtime_image, realtime_image, cv::COLOR_BGR2RGB);
        auto longitude = 0.0;
        auto latitude = 0.0;
        auto relative_height = 0.0;
        auto roll = 0.0;
        auto pitch = 0.0;
        auto yaw = 0.0;
        auto index = path.rfind("/");
        string file_name = path.substr(index + 1);
        auto id = file_name.substr(0, file_name.size() - 4);
        ParseFileName(file_name, longitude, latitude, relative_height, roll, pitch, yaw);
        auto current_position = StdVector3(longitude, latitude, relative_height);
        auto current_posture = StdVector3(roll, pitch, yaw);
        detector.Detect(realtime_image, current_position, current_posture);
    }

    return application.exec();
}
