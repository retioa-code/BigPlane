#include "Locator.h"
#include "1.Common/HeaderFiles.h"
#include "1.Common/EngineFunction.h"
#include "6.GIS/GISHelper.h"
#include "3.Vision/ImageMatcher.h"
#include "../0.Platform/PlatformFunctions.h"

using namespace VisionGISAIEngine;
using namespace UavVisionPlatform;

void Locator::ReadPlatformLocationConfig() {
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

void Locator::Initialize(QString dom_path, QString dem_path, QString work_dir) {
    ReadPlatformLocationConfig();
    LoadGuaranteedData(dom_path, dem_path);
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

void Locator::ExecuteDebugging() {
    string realtime_dir = (WorkDir + "1.RealtimeRgbImages/").toLocal8Bit().data();
    string reference_dir = (WorkDir + "1.ReferenceRgbImages/").toLocal8Bit().data();
    CreateReferenceMultiple(realtime_dir, reference_dir, DomPath);
    if (!filesystem::exists(realtime_dir)) {
        cout << "Camera image dir was not found: " << realtime_dir << ". " << __LINE__ << "@" << __FILENAME__ << endl;
        return;
    }
    // 遍历文件夹中的文件，并按名称排序
    std::vector<filesystem::directory_entry> entries;
    for (const auto &entry: filesystem::directory_iterator(realtime_dir)) {
        entries.push_back(entry);
    }
    std::sort(entries.begin(), entries.end(), [](const filesystem::directory_entry &a, const filesystem::directory_entry &b) {
        return a.path().filename() < b.path().filename();
    });
    // 遍历读取实时图和基准图，并执行检测
    for (auto i = 0; i < entries.size(); i++) {
        auto path = entries.at(i).path().string();
        if (".jpg" != path.substr(path.size() - 4)) {
            continue;
        }
        auto id = path.substr(path.rfind("/") + 1);
        id = id.substr(0, id.size() - 4);
        QElapsedTimer elapsed_timer;
        elapsed_timer.start();
        Mat realtime_rgb_image;
        Mat padding_ref_rgb_image;
        int clip_left;
        int clip_top;
        StdVector3 position;
        StdVector3 orientation;
        SetParamDebugging(WorkDir, id, realtime_rgb_image, padding_ref_rgb_image, clip_left, clip_top, position, orientation);
        auto location = Locate(id, realtime_rgb_image, padding_ref_rgb_image, clip_left, clip_top, position.Z, orientation);
        qDebug().noquote().nospace() << "Location:(" + QString::number(location.X, 10, 6) + "," + QString::number(location.Y, 10, 6) + "," + QString::number(location.Z)
                                        + "), TimeSpend:" + QString::number(elapsed_timer.elapsed()) + "ms";
    }
}

void Locator::SetParamDebugging(QString work_dir, string id, Mat &realtime_rgb_image, Mat &padding_reference_image, int &clip_left, int &clip_top, StdVector3 &position, StdVector3 &orientation) {
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

StdVector3 Locator::Locate(cv::Mat realtime_image, double height, StdVector3 current_orientation) {
    if (RunningStatus == Debugging) {
        ExecuteDebugging();
        _Exit(EXIT_SUCCESS);  // exit(0); 执行这句会有段错误
    }
    cv::Mat reference_image;
    int clip_left = 0, clip_top = 0;
    auto result = CreateReferenceSingle(PreviousPosition, PreviousOrientation, reference_image, clip_left, clip_top);
    if (!result) {
        return StdVector3(0, 0, 0);
    }
    string id = QDateTime::currentDateTime().toString("hhmmss").toLocal8Bit().data();
    StdVector3 current_position = Locate(id, realtime_image, reference_image, clip_left, clip_top, height, current_orientation);
    if (GpsRtkIsOn) {
        // Todo 从GPS或RTK获取
        PreviousPosition = StdVector3(1, 1, 1);
    } else {
        PreviousPosition = current_position;
    }
    PreviousOrientation = current_orientation;
    return current_position;
}

// 俯仰角正负约定：前向为0，上正下负，横滚角正负约定：平行为0，右侧向下为正,偏航角正负约定：北向为0，左负右正.  注意： previousLlh 和 current_llh 变量的高度是相对地面高度
StdVector3 Locator::Locate(string id, cv::Mat &realtime_image, cv::Mat &reference_image, int clip_left, int clip_top, double height, StdVector3 current_orientation) {
    if (height < MinWorkHeight || height >= MaxWorkHeight) {
        auto text = "Relative ground height is " + QString::number(height, 10, 2) + ", but it mast be between " + QString::number(MinWorkHeight, 10, 2) + " and " + QString::number(MaxWorkHeight, 10, 2);
        std::cout << text.toLocal8Bit().data() << "  " << __LINE__ << "@" << __FILENAME__ << std::endl;
        return StdVector3(0.001, 0.001, 0.01);
    }
    // 3. 执行图像匹配
    cv::Point left_tof_point(0, 0);
    cv::Point center_point(0, 0);
    double confidence = 0;
    cv::Mat corr_surface;
    SaveTextLog("Match Started, " + std::to_string(__LINE__) + "@" + __FILENAME__ + ". ");
//    MatchByAkaze(reference_image, realtime_image, center_point);
//    MatchBySift(reference_image, realtime_image, center_point);
    MatchByTemplate(reference_image, realtime_image, left_tof_point, center_point, confidence, corr_surface, DownSampleFactor);
//    MatchByGradCorrelative(reference_image, realtime_image, left_tof_point, center_point, confidence);
//    MatchByGradIntensity(reference_image, realtime_image, left_tof_point, center_point, confidence, corr_surface);
    SaveTextLog("Match Stopped, " + std::to_string(__LINE__) + "@" + __FILENAME__ + ". ");
    // 4. 解算经度和纬度
    double match_gauss_x, match_gauss_y;
    PixelToGeography(clip_left + center_point.x, clip_top + center_point.y, match_gauss_x, match_gauss_y, DomGeoTransform);
    // 匹配后得到地面点经纬度，再计算wrj经纬度时，应加上这个地面点和无人机位置偏移量的影响
    auto current_offset = CalcGroundToAir(height, StdVector3(current_orientation.X, current_orientation.Y, current_orientation.Z));
    auto location_x = match_gauss_x + current_offset.X;
    auto location_y = match_gauss_y + current_offset.Y;
    double location_longitude, location_latitude;  // 不能被替换为 current_llh 的x和y分量，开发测试环境下，这两个分量用于在日志中输出参考位置的经纬度，统计定位误差
    ConvertGauss3ToLLH(location_x, location_y, location_longitude, location_latitude);
    SaveTextLog("Resolve Stopped, " + std::to_string(__LINE__) + "@" + __FILENAME__ + ". ");
////    // 5. 保存日志
////    double match_longitude, match_latitude;  // 不能被替换为 current_llh 的x和y分量，开发测试环境下，这两个分量用于在日志中输出参考位置的经纬度，统计定位误差
////    ConvertGauss3ToLLH(match_gauss_x, match_gauss_y, match_longitude, match_latitude);
////    SavePositioningLog(reference_image, realtime_image, left_tof_point, center_point, confidence, match_longitude, match_latitude, location_x, location_y, location_longitude, location_latitude, current_llh, current_orientation, previousLlh, previous_orientation, previous_velocity, current_velocity, time_point_list);
    return StdVector3(location_longitude, location_latitude, height);
}
