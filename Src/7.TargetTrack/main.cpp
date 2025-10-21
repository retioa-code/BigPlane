# pragma execution_character_set("utf-8")
#include <iostream>
#include <QtCore/QJsonParseError>
#include <QtCore/QCoreApplication>
#include <QtCore/QFile>
#include "1.Common/MacroEnumStruct.h"
#include "0.Platform/PlatformFunctions.h"
#include "7.Tracker/ITracker.h"

using namespace UavVisionPlatform;

int InvokeInterval = 40;
QString WorkDir = "";
QString GuaranteedDataDir = "";
QString DomPath = "";
QString DemPath = "";

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
    ITracker tracker;
    tracker.Initialize(GuaranteedDataDir, WorkDir);
    auto previous_dataTime = QDateTime::currentDateTime();
    QElapsedTimer elapsed_timer;
    while (true) {
        auto current_time = QDateTime::currentDateTime();
        auto milli_seconds = current_time.toMSecsSinceEpoch() - previous_dataTime.toMSecsSinceEpoch();
        if (milli_seconds < InvokeInterval) {
            std::this_thread::sleep_for(std::chrono::milliseconds(InvokeInterval - milli_seconds));
            continue;
        }
        previous_dataTime = current_time;  // ���Ӧ�ŵ���ͼ��仯����ǰ��
        if (!InitializeLiftOff()) {    // ��ɵ����ѹ�Ƹ߶Ⱥ͸̳߳�ʼ����Ҫд��ˢ�¹����ڴ��Ժ���ΪҪ�õ����Ⱥ�γ�Ȳ���
            continue;
        }
        // Todo ========================================================
        cv::Mat realtime_image;
        double height = 0;
        StdVector3 current_posture;
        elapsed_timer.start();
        auto location = tracker.Track(realtime_image, height, current_posture);
        qDebug().noquote().nospace() << current_time.toString("hhmmss") << ":(" + QString::number(location.X, 10, 6) + "," + QString::number(location.Y, 10, 6)
                                     << "," + QString::number(location.Z) + "), Spend:" + QString::number(elapsed_timer.elapsed()) + "ms";
    }

    return application.exec();
}
