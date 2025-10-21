#pragma once

#include <QtCore/QThread>
#include <opencv2/core/mat.hpp>

class LocationThread : public QThread {
    QString Id;
    cv::Mat RealtimeImage;
    cv::Mat ReferenceImage;
public:
    LocationThread();

    ~LocationThread();

    void SetParameter(QString id, cv::Mat &realtime_image, cv::Mat &reference_image);

    void run() override;

    void SaveMatchPoint(QList<QMap<QString, QString>> csv_result_list);

    void SaveMatchPolyline(QList<QMap<QString, QString>> csv_result_list);

    void SavePositioningPoint(QList<QMap<QString, QString>> csv_result_list);

    void SavePositioningPolyline(QList<QMap<QString, QString>> csv_result_list);

    void SaveGpsPoint(QList<QMap<QString, QString>> csv_result_list);

    void SaveGpsPolyline(QList<QMap<QString, QString>> csv_result_list);

    void SaveLogCsv(QString csv_file_path, double distance_error);
};

