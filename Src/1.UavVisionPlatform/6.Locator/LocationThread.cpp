# pragma execution_character_set("utf-8")

#include "../../0.VisionGISAIEngine/1.Common/HeaderFiles.h"
#include "LocationThread.h"
#include "shapefil.h"

LocationThread::LocationThread() {

}

LocationThread::~LocationThread() {

}

void LocationThread::SetParameter(QString id, cv::Mat &realtime_image, cv::Mat &reference_image) {
    Id = id;
    RealtimeImage = realtime_image;
    ReferenceImage = reference_image;
}

void LocationThread::run() {
//    if (RealtimeImage.empty() || ReferenceImage.empty()) return;
//    if (!NeedSaveDebugInfo) return;
//    // 保存Html和Csv
//    static int index = 0;
//    // 更新Html日志文件，默认毎100个图片在一个html文件中
//    auto temp = FillZeroAsPrefix(index / 100, 2);  // 标记当前帧在哪个html中
//    auto html_file_path = QString::fromLocal8Bit(LogFolderPath + "_" + temp + ".html");
//    auto distance_delta_x = positioning_x - psdk_gauss_x;
//    auto distance_delta_y = positioning_y - psdk_gauss_y;
//    auto distance_error = sqrt(distance_delta_x * distance_delta_x + distance_delta_y * distance_delta_y);
//    SaveLogHtml(html_file_path, index, distance_error);
//    // 更新Csv日志文件
//    SaveLogCsv(LogFolderPath + "_0000.csv", distance_error);
//    index++;
}

void LocationThread::SavePositioningPoint(QList<QMap<QString, QString>> csv_result_list) {
//    auto shp_file_path = LogFolderPath + "/PositioningPoint.shp";
//    QFile temp(shp_file_path);
//    if (temp.exists()) {
//        shp_file_path = LogFolderPath + "/PositioningPoint" + IntervalTimeStr + ".shp";
//    }
//    SHPHandle outShp = SHPCreate(shp_file_path.toLocal8Bit().data(), SHPT_POINT);
//    DBFHandle dbf_h = DBFCreate(shp_file_path.toLocal8Bit().data());
//    DBFAddField(dbf_h, "SysTime", FTString, 22, 0);
//    DBFAddField(dbf_h, "Longitude", FTDouble, 11, 6);
//    DBFAddField(dbf_h, "Latitude", FTDouble, 11, 6);
//    DBFAddField(dbf_h, "GroundHgt", FTDouble, 11, 2);
//    DBFAddField(dbf_h, "DistError", FTDouble, 11, 2);
//    for (int ir = 0; ir < csv_result_list.size(); ir++) {
//        auto *xCoords = new double[1];
//        auto *yCoords = new double[1];
//        xCoords[0] = csv_result_list[ir]["定位经度"].toDouble();
//        yCoords[0] = csv_result_list[ir]["定位纬度"].toDouble();
//        auto psShape = SHPCreateObject(SHPT_POINT, -1, 0, NULL, NULL, 1, xCoords, yCoords, NULL, NULL);
//        int id = SHPWriteObject(outShp, -1, psShape);
//        DBFWriteStringAttribute(dbf_h, id, 0, csv_result_list[ir]["当前系统时间"].toLocal8Bit().data());
//        DBFWriteDoubleAttribute(dbf_h, id, 1, csv_result_list[ir]["定位经度"].toDouble());
//        DBFWriteDoubleAttribute(dbf_h, id, 2, csv_result_list[ir]["定位纬度"].toDouble());
//        DBFWriteDoubleAttribute(dbf_h, id, 3, csv_result_list[ir]["当前点GPS高度"].toDouble());
//        DBFWriteDoubleAttribute(dbf_h, id, 4, csv_result_list[ir]["参考GPS误差"].toDouble());
//        SHPDestroyObject(psShape);
//        delete[] xCoords;
//        delete[] yCoords;
//    }
//    SHPClose(outShp);
//    DBFClose(dbf_h);
}

void LocationThread::SavePositioningPolyline(QList<QMap<QString, QString>> csv_result_list) {
//    auto size = csv_result_list.size();
//    if (size < 2) {
//        return;
//    }
//    auto shp_file_path = LogFolderPath + "/PositioningPolyline.shp";
//    QFile temp(shp_file_path);
//    if (temp.exists()) {
//        shp_file_path = LogFolderPath + "/PositioningPolyline" + IntervalTimeStr + ".shp";
//    }
//    SHPHandle outShp = SHPCreate(shp_file_path.toLocal8Bit().data(), SHPT_ARC);
//    DBFHandle dbf_h = DBFCreate(shp_file_path.toLocal8Bit().data());
//    DBFAddField(dbf_h, "Id", FTInteger, 22, 0);
//    for (int ir = 0; ir < 1; ir++) {  // 仅保存一条线
//        double *xCoords = new double[size];
//        double *yCoords = new double[size];
//        for (int ip = 0; ip < size; ip++) {
//            xCoords[ip] = csv_result_list[ip]["定位经度"].toDouble();
//            yCoords[ip] = csv_result_list[ip]["定位纬度"].toDouble();
//        }
//        auto psShape = SHPCreateObject(SHPT_ARC, -1, 0, NULL, NULL, size, xCoords, yCoords, NULL, NULL);
//        int id = SHPWriteObject(outShp, -1, psShape);
//        DBFWriteDoubleAttribute(dbf_h, id, 0, ir);
//        SHPDestroyObject(psShape);
//        delete[] xCoords;
//        delete[] yCoords;
//    }
//    SHPClose(outShp);
//    DBFClose(dbf_h);
}

void LocationThread::SaveMatchPoint(QList<QMap<QString, QString>> csv_result_list) {
//    auto shp_file_path = LogFolderPath + "/MatchPoint.shp";
//    QFile temp(shp_file_path);
//    if (temp.exists()) {
//        shp_file_path = LogFolderPath + "/MatchPoint" + IntervalTimeStr + ".shp";
//    }
//    SHPHandle outShp = SHPCreate(shp_file_path.toLocal8Bit().data(), SHPT_POINT);
//    DBFHandle dbf_h = DBFCreate(shp_file_path.toLocal8Bit().data());
//    DBFAddField(dbf_h, "SysTime", FTString, 22, 0);
//    DBFAddField(dbf_h, "Longitude", FTDouble, 11, 6);
//    DBFAddField(dbf_h, "Latitude", FTDouble, 11, 6);
//    for (int ir = 0; ir < csv_result_list.size(); ir++) {
//        auto *xCoords = new double[1];
//        auto *yCoords = new double[1];
//        xCoords[0] = csv_result_list[ir]["匹配点经度"].toDouble();
//        yCoords[0] = csv_result_list[ir]["匹配点纬度"].toDouble();
//        auto psShape = SHPCreateObject(SHPT_POINT, -1, 0, NULL, NULL, 1, xCoords, yCoords, NULL, NULL);
//        int id = SHPWriteObject(outShp, -1, psShape);
//        DBFWriteStringAttribute(dbf_h, id, 0, csv_result_list[ir]["当前系统时间"].toLocal8Bit().data());
//        DBFWriteDoubleAttribute(dbf_h, id, 1, csv_result_list[ir]["匹配点经度"].toDouble());
//        DBFWriteDoubleAttribute(dbf_h, id, 2, csv_result_list[ir]["匹配点纬度"].toDouble());
//        SHPDestroyObject(psShape);
//        delete[] xCoords;
//        delete[] yCoords;
//    }
//    SHPClose(outShp);
//    DBFClose(dbf_h);
}

void LocationThread::SaveMatchPolyline(QList<QMap<QString, QString>> csv_result_list) {
//    auto size = csv_result_list.size();
//    if (size < 2) {
//        return;
//    }
//    auto shp_file_path = LogFolderPath + "/MatchPolyline.shp";
//    QFile temp(shp_file_path);
//    if (temp.exists()) {
//        shp_file_path = LogFolderPath + "/MatchPolyline" + IntervalTimeStr + ".shp";
//    }
//    SHPHandle outShp = SHPCreate(shp_file_path.toLocal8Bit().data(), SHPT_ARC);
//    DBFHandle dbf_h = DBFCreate(shp_file_path.toLocal8Bit().data());
//    DBFAddField(dbf_h, "Id", FTInteger, 22, 0);
//    for (int ir = 0; ir < 1; ir++) {  // 仅保存一条线
//        double *xCoords = new double[size];
//        double *yCoords = new double[size];
//        for (int ip = 0; ip < size; ip++) {
//            xCoords[ip] = csv_result_list[ip]["匹配点经度"].toDouble();
//            yCoords[ip] = csv_result_list[ip]["匹配点纬度"].toDouble();
//        }
//        auto psShape = SHPCreateObject(SHPT_ARC, -1, 0, NULL, NULL, size, xCoords, yCoords, NULL, NULL);
//        int id = SHPWriteObject(outShp, -1, psShape);
//        DBFWriteDoubleAttribute(dbf_h, id, 0, ir);
//        SHPDestroyObject(psShape);
//        delete[] xCoords;
//        delete[] yCoords;
//    }
//    SHPClose(outShp);
//    DBFClose(dbf_h);
}

void LocationThread::SaveGpsPoint(QList<QMap<QString, QString>> csv_result_list) {
//    auto shp_file_path = LogFolderPath + "/GpsPoint.shp";
//    QFile temp(shp_file_path);
//    if (temp.exists()) {
//        shp_file_path = LogFolderPath + "/GpsPoint" + IntervalTimeStr + ".shp";
//    }
//    SHPHandle outShp = SHPCreate(shp_file_path.toLocal8Bit().data(), SHPT_POINT);
//    DBFHandle dbf_h = DBFCreate(shp_file_path.toLocal8Bit().data());
//    DBFAddField(dbf_h, "SysTime", FTString, 22, 0);
//    DBFAddField(dbf_h, "Longitude", FTDouble, 11, 6);
//    DBFAddField(dbf_h, "Latitude", FTDouble, 11, 6);
//    DBFAddField(dbf_h, "GroundHgt", FTDouble, 11, 2);
//    for (int ir = 0; ir < csv_result_list.size(); ir++) {
//        auto *xCoords = new double[1];
//        auto *yCoords = new double[1];
//        xCoords[0] = csv_result_list[ir]["当前点GPS经度"].toDouble();
//        yCoords[0] = csv_result_list[ir]["当前点GPS纬度"].toDouble();
//        auto psShape = SHPCreateObject(SHPT_POINT, -1, 0, NULL, NULL, 1, xCoords, yCoords, NULL, NULL);
//        int id = SHPWriteObject(outShp, -1, psShape);
//        DBFWriteStringAttribute(dbf_h, id, 0, csv_result_list[ir]["当前系统时间"].toLocal8Bit().data());
//        DBFWriteDoubleAttribute(dbf_h, id, 1, csv_result_list[ir]["当前点GPS经度"].toDouble());
//        DBFWriteDoubleAttribute(dbf_h, id, 2, csv_result_list[ir]["当前点GPS纬度"].toDouble());
//        DBFWriteDoubleAttribute(dbf_h, id, 3, csv_result_list[ir]["当前点GPS高度"].toDouble());
//        SHPDestroyObject(psShape);
//        delete[] xCoords;
//        delete[] yCoords;
//    }
//    SHPClose(outShp);
//    DBFClose(dbf_h);
}

void LocationThread::SaveGpsPolyline(QList<QMap<QString, QString>> csv_result_list) {
//    auto size = csv_result_list.size();
//    if (size < 2) {
//        return;
//    }
//    auto shp_file_path = LogFolderPath + "/GpsPolyline.shp";
//    QFile temp(shp_file_path);
//    if (temp.exists()) {
//        shp_file_path = LogFolderPath + "/GpsPolyline" + IntervalTimeStr + ".shp";
//    }
//    SHPHandle outShp = SHPCreate(shp_file_path.toLocal8Bit().data(), SHPT_ARC);
//    DBFHandle dbf_h = DBFCreate(shp_file_path.toLocal8Bit().data());
//    DBFAddField(dbf_h, "Id", FTInteger, 22, 0);
//    for (int ir = 0; ir < 1; ir++) {  // 仅保存一条线
//        double *xCoords = new double[size];
//        double *yCoords = new double[size];
//        for (int ip = 0; ip < size; ip++) {
//            xCoords[ip] = csv_result_list[ip]["当前点GPS经度"].toDouble();
//            yCoords[ip] = csv_result_list[ip]["当前点GPS纬度"].toDouble();
//        }
//        auto psShape = SHPCreateObject(SHPT_ARC, -1, 0, NULL, NULL, size, xCoords, yCoords, NULL, NULL);
//        int id = SHPWriteObject(outShp, -1, psShape);
//        DBFWriteDoubleAttribute(dbf_h, id, 0, ir);
//        SHPDestroyObject(psShape);
//        delete[] xCoords;
//        delete[] yCoords;
//    }
//    SHPClose(outShp);
//    DBFClose(dbf_h);
}

void LocationThread::SaveLogCsv(QString csv_file_path, double distance_error) {
    // QFile csv_file(csv_file_path);
    // auto content = "\n" + Id + "," + QString::number(distance_error, 10, 2);
    // if (!csv_file.open(QFile::WriteOnly | QFile::Append)) {
    //     return;
    // }
    // QTextStream write_text_stream(&csv_file);
    // write_text_stream.setEncoding(QStringConverter::Utf8);
    // if (csv_file.pos() == 0) {  // 如果是新文件，添加 UTF-8 BOM
    //     csv_file.write("\xEF\xBB\xBF");
    //     content = "系统时间,定位误差" + content;
    // }
    // write_text_stream << content;
    // csv_file.close();
}

