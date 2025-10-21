#include "ITracker.h"
#include "../../0.VisionGISAIEngine/1.Common/HeaderFiles.h"
#include "../../0.VisionGISAIEngine/1.Common/EngineFunction.h"
#include "../../0.VisionGISAIEngine/6.GIS/GISHelper.h"
#include "../0.Platform/PlatformFunctions.h"

using namespace VisionGISAIEngine;
using namespace UavVisionPlatform;

void ITracker::ReadPlatformLocationConfig() {
    QString path = QCoreApplication::applicationDirPath() + "/PlatformConfig.json";
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        qDebug() << "Open config file failed: " + path + ". " << __LINE__ << "@" << __FILENAME__;
        exit(0);
    }
    QByteArray data = file.readAll();
    file.close();
    QJsonParseError parseError;
    auto document = QJsonDocument::fromJson(data, &parseError);
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
    RunningStatus = (RunningStatusEnum) location_param.value("RunningStatus").toInt();
    MinWorkHeight = location_param.value("MinWorkHeight").toInt();
    MaxWorkHeight = location_param.value("MaxWorkHeight").toInt();
    auto need = location_param.value("NeedSaveDebugInfo").toBool();
    SetNeedSaveDebugInfo(need);
    DownSampleFactor = location_param.value("DownSampleFactor").toDouble();
}

void ITracker::Initialize(QString dom_path, QString dem_path, QString work_dir) {
    ReadPlatformLocationConfig();
    // LoadGuaranteedData(dom_path, dem_path);
#ifdef  WIN32
    //    RunningStatus = Debugging;
#endif
    if (RunningStatus == Debugging) {
        LogFolderPath = work_dir + "2.RunningLog/";
    } else {
        auto time = QDateTime::currentDateTime().toString("hhmmss");
        LogFolderPath = work_dir + "00Log_" + time + "/";
    }
    QDir dir(LogFolderPath);
    if (dir.exists()) {
        dir.removeRecursively();
    }
    dir.mkpath(LogFolderPath);
    WorkDir = work_dir;
    DomPath = dom_path;
}

void ITracker::Initialize(QString guaranteed_data_dir, QString work_dir) {
    ReadPlatformLocationConfig();
    // LoadGuaranteedData(guaranteed_data_dir);
#ifdef  WIN32
    //    RunningStatus = Debugging;
#endif
    if (RunningStatus == Debugging) {
        LogFolderPath = work_dir + "2.RunningLog/";
    } else {
        auto time = QDateTime::currentDateTime().toString("hhmmss");
        LogFolderPath = work_dir + "00Log_" + time + "/";
    }
    QDir dir(LogFolderPath);
    if (dir.exists()) {
        dir.removeRecursively();
    }
    dir.mkpath(LogFolderPath);
    WorkDir = work_dir;
}

void ITracker::SetParamDebugging(QString work_dir, string id, Mat &realtime_rgb_image, Mat &padding_reference_image, int &clip_left, int &clip_top, StdVector3 &position, StdVector3 &orientation) {
    // 实时图
    string realtime_file_name = (work_dir + "1.RealtimeRgbImages/").toLocal8Bit().data() + id + ".jpg";
    if (!filesystem::exists(realtime_file_name)) {
        cout << "File was not found: " << realtime_file_name << "  " << __LINE__ << "@" << __FILENAME__ << endl;
        exit(0);
    }
    realtime_rgb_image = imread(realtime_file_name);
//    auto image_total = CameraImageCols * CameraImageRows * 2;
//    if (nullptr == realtime_yuv_buffer) {
//        realtime_yuv_buffer = new uint8_t[image_total];
//    }
//    auto fp = fopen(realtime_file_name.c_str(), "rb");
//    fread(realtime_yuv_buffer, sizeof(uint8_t), image_total, fp);
//    fclose(fp);
//    GenerateRealtimeImage(id,realtime_yuv_buffer,position.Z);
    // 2. 基准图
    string reference_file_name = (work_dir + "1.ReferenceRgbImages/").toLocal8Bit().data() + id + ".jpg";
    if (!filesystem::exists(reference_file_name)) {
        cout << "File was not found: " << reference_file_name << "  " << __LINE__ << "@" << __FILENAME__ << endl;
        exit(0);
    }
    auto path = work_dir + "1.ReferenceRgbImages/" + QString::fromLocal8Bit(id.c_str()) + ".txt";
    QFile file(path);
    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream in(&file);
        clip_left = in.readLine().remove("ClipLeftInDom:").toInt();
        clip_top = in.readLine().remove("ClipTopInDom:").toInt();
        file.close();
    } else {
        qDebug() << "Read file failed: " << path << ". " << __LINE__ << "@" << __FILENAME__;
    }
    padding_reference_image = imread(reference_file_name);
    // 2. 位姿
    auto longitude = 0.0;
    auto latitude = 0.0;
    auto relative_height = 0.0;
    auto roll = 0.0;
    auto pitch = 0.0;
    auto yaw = 0.0;
    ParseFileName(id, longitude, latitude, relative_height, roll, pitch, yaw);
    position = StdVector3(longitude, latitude, relative_height);
    orientation = StdVector3(roll, pitch, yaw);
}

void ITracker::Track(Mat& realtime_image, StdVector3 position, StdVector3 orientation) {
    string id = QDateTime::currentDateTime().toString("hhmmss_zzz").toLocal8Bit().data();

}
