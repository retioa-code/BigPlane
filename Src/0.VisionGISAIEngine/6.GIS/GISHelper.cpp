# pragma execution_character_set("utf-8")

#include <shapefil.h>
#include "GISHelper.h"
#include "../1.Common/EngineFunction.h"

#define p0 206264.8062470963551564
//wgs84参考椭球参数
const double e = 0.00669438002290;
const double e1 = 0.00673949677548;
const double b = 6356752.3141;
const double a = 6378137.0;

// 根据飞行的相对地面高度和姿态，计算匹配点地面经纬度和无人机经纬度的位置偏移， pos_posture 的各元素依次为：横滚俯仰偏航
StdVector2 VisionGISAIEngine::CalcGroundToAir(double ground_height, StdVector3 pos_posture) {
    auto roll_angle = pos_posture.X * M_PI / 180; // 横滚角
    auto pitch_angle = pos_posture.Y * M_PI / 180; // 俯仰角
    auto yaw_angle = pos_posture.Z * M_PI / 180; // 偏航角
    auto roll_offset = ground_height * tan(roll_angle); // 以匹配点为原点的X轴分量
    auto pitch_offset = -ground_height * tan(pitch_angle); // 以匹配点为原点的Y轴分量
    auto x_offset = roll_offset * cos(yaw_angle) + pitch_offset * sin(yaw_angle);
    auto y_offset = pitch_offset * cos(yaw_angle) - roll_offset * sin(yaw_angle);
    return StdVector2(x_offset, y_offset);
}

void VisionGISAIEngine::PixelToGeography(int pixel_col, int pixel_row, double &geography_x, double &geography_y, double geo_transform[6]) {
    geography_x = geo_transform[0] + pixel_col * geo_transform[1];
    geography_y = geo_transform[3] + pixel_row * geo_transform[5];
}

void VisionGISAIEngine::GeographyToPixel(double geography_x, double geography_y, int &pixel_col, int &pixel_row, double geo_transform[6]) {
    pixel_col = (geography_x - geo_transform[0]) / geo_transform[1];
    pixel_row = (geography_y - geo_transform[3]) / geo_transform[5];
}

// 参考自:  https://blog.csdn.net/weixin_47156401/article/details/124363834
//大地坐标转高斯3度投影坐标，不要重构，可以和参考网页尽量保持一致
void VisionGISAIEngine::ConvertLLHToGauss3(double longitude, double latitude, double &gauss3_x, double &gauss3_y) {
    //把度转化为弧度
    latitude = latitude * M_PI / 180;
    longitude = longitude * M_PI / 180;

    double N, t, n, c, V, Xz, m1, m2, m3, m4, m5, m6, a0, a2, a4, a6, a8, M0, M2, M4, M6, M8, x0, y0, l;

    int L_num;
    double L_center;

    //中央子午线经度，6°带
    L_num = (int) ((longitude * 180 / M_PI + 1.5) / 3.0);
    L_center = 3 * L_num;

    //中央子午线经度，3°带
    //L_num = (int)(longitude * 180 / pi / 3.0 + 0.5);
    //L_center = 3 * L_num;

    l = (longitude / M_PI * 180 - L_center) * 3600; //求带号、中央经线、经差

    M0 = a * (1 - e);
    M2 = 3.0 / 2.0 * e * M0;
    M4 = 5.0 / 4.0 * e * M2;
    M6 = 7.0 / 6.0 * e * M4;
    M8 = 9.0 / 8.0 * e * M6;

    a0 = M0 + M2 / 2.0 + 3.0 / 8.0 * M4 + 5.0 / 16.0 * M6 + 35.0 / 128.0 * M8;
    a2 = M2 / 2.0 + M4 / 2 + 15.0 / 32.0 * M6 + 7.0 / 16.0 * M8;
    a4 = M4 / 8.0 + 3.0 / 16.0 * M6 + 7.0 / 32.0 * M8;
    a6 = M6 / 32.0 + M8 / 16.0;
    a8 = M8 / 128.0;

    Xz = a0 * latitude - a2 / 2.0 * sin(2 * latitude) + a4 / 4.0 * sin(4 * latitude) - a6 / 6.0 * sin(6 * latitude) + a8 / 8.0 * sin(8 * latitude);  //计算子午线弧长
    c = a * a / b;
    V = sqrt(1 + e1 * cos(latitude) * cos(latitude));
    N = c / V;
    t = tan(latitude);
    n = e1 * cos(latitude) * cos(latitude);

    m1 = N * cos(latitude);
    m2 = N / 2.0 * sin(latitude) * cos(latitude);
    m3 = N / 6.0 * pow(cos(latitude), 3) * (1 - t * t + n);
    m4 = N / 24.0 * sin(latitude) * pow(cos(latitude), 3) * (5 - t * t + 9 * n);
    m5 = N / 120.0 * pow(cos(latitude), 5) * (5 - 18 * t * t + pow(t, 4) + 14 * n - 58 * n * t * t);
    m6 = N / 720.0 * sin(latitude) * pow(cos(latitude), 5) * (61 - 58 * t * t + pow(t, 4));
    x0 = Xz + m2 * l * l / pow(p0, 2) + m4 * pow(l, 4) / pow(p0, 4) + m6 * pow(l, 6) / pow(p0, 6);
    y0 = m1 * l / p0 + m3 * pow(l, 3) / pow(p0, 3) + m5 * pow(l, 5) / pow(p0, 5);   //计算x y坐标

    gauss3_y = x0;
    gauss3_x = y0 + 500000 + L_num * 1000000;     //化为国家统一坐标
}

//高斯3度投影坐标转大地坐标，不要重构，可以和参考网页尽量保持一致
void VisionGISAIEngine::ConvertGauss3ToLLH(double gauss3_x, double gauss3_y, double &longitude, double &latitude) {
    auto zone = (int) gauss3_x / 1000000;
    gauss3_x = gauss3_x - zone * 1000000;
    //l0为中央经度
    double Bf, B00, FBf, M, N, V, t, n, c, y1, n1, n2, n3, n4, n5, n6, a0, a2, a4, a6, M0, M2, M4, M6, M8, l;

    int L_num, L_center;

    L_num = (int) (gauss3_y / 1000000.0);
    y1 = gauss3_x - 500000;
    //y1 = gauss3_x - 500000 - L_num * 1000000;

    //L_center = ((L_num + 1) * 6 - 3)*pi*180;		//中央子午线经度，6°带
    //cout<<"L_center="<<L_center<<endl;
    //L_center = L_num * 3;			//中央子午线经度，3°带

    M0 = a * (1 - e);
    M2 = 3.0 / 2.0 * e * M0;
    M4 = 5.0 / 4.0 * e * M2;
    M6 = 7.0 / 6.0 * e * M4;
    M8 = 9.0 / 8.0 * e * M6;

    a0 = M0 + M2 / 2.0 + 3.0 / 8.0 * M4 + 5.0 / 16.0 * M6 + 35.0 / 128.0 * M8;
    a2 = M2 / 2.0 + M4 / 2 + 15.0 / 32.0 * M6 + 7.0 / 16.0 * M8;
    a4 = M4 / 8.0 + 3.0 / 16.0 * M6 + 7.0 / 32.0 * M8;
    a6 = M6 / 32.0 + M8 / 16.0;

//    cout << "a0=" << a0 << endl;
//    cout << "a2=" << a2 << endl;
//    cout << "a4=" << a4 << endl;
//    cout << "a6=" << a6 << endl;

    Bf = gauss3_y / a0;
    B00 = Bf;
//    cout << "B00=" << B00 << endl;
//
//    cout << "sin(2 * B00)=" << sin(2 * B00) / 2 << endl;

    while ((fabs(Bf - B00) > 0.0000001) || (B00 == Bf)) {
        B00 = Bf;
        FBf = -a2 / 2.0 * sin(2 * B00) + a4 / 4.0 * sin(4 * B00) - a6 / 6.0 * sin(6 * B00);
        Bf = (gauss3_y - FBf) / a0;
    }    //迭代求数值为x坐标的子午线弧长对应的底点纬度

//    cout << "Bf=" << Bf << endl;

    t = tan(Bf);                            //一样
    c = a * a / b;
    V = sqrt(1 + e1 * cos(Bf) * cos(Bf));   //一样
    N = c / V;                              //一样
    M = c / pow(V, 3);                      //一样
    n = e1 * cos(Bf) * cos(Bf);             //一样(为n的平方)

    n1 = 1 / (N * cos(Bf));
    n2 = -t / (2.0 * M * N);
    n3 = -(1 + 2 * t * t + n) / (6.0 * pow(N, 3) * cos(Bf));
    n4 = t * (5 + 3 * t * t + n - 9 * n * t * t) / (24.0 * M * pow(N, 3));
    n5 = (5 + 28 * t * t + 24 * pow(t, 4) + 6 * n + 8 * n * t * t) / (120.0 * pow(N, 5) * cos(Bf));
    n6 = -t * (61 + 90 * t * t + 45 * pow(t, 4)) / (720.0 * M * pow(N, 5));

    //秒
    latitude = (Bf + n2 * y1 * y1 + n4 * pow(y1, 4) + n6 * pow(y1, 6)) / M_PI * 180;

    double L0 = zone * 3;

    l = n1 * y1 + n3 * pow(y1, 3) + n5 * pow(y1, 5);
    //double L = L_center + l / pi * 180;    //反算得大地经纬度
    longitude = L0 + l / M_PI * 180;
}

bool VisionGISAIEngine::ConvertGauss6ToLLH(double gauss_x, double gauss_y, double &out_longitude_angle, double &out_latitude_angle) {
    double sinB, cosB, t, t2, N, ng2, V, yN;
    double preB0, B00;
    double eta;
    double Elli_a, Elli_f, Elli_e2, Elli_e12; // 椭球体的长半轴，扁率，第一偏心率平方，第二偏心率平方
    double A1, A2, A3, A4, A5;
    double L0; // 中央子午线经度
    int iBand; // 投影带带号

    // 84系统
    Elli_a = 6378137;
    Elli_f = 298.257223563;


    Elli_e2 = 1 - (Elli_f - 1) / Elli_f * ((Elli_f - 1) / Elli_f);
    Elli_e12 = Elli_f / (Elli_f - 1) * (Elli_f / (Elli_f - 1)) - 1;


    // 计算中央经线以及投影带带号
    iBand = floor(gauss_x / 1000000);
    L0 = (iBand * 6 - 3) * M_PI / 180;


    //计算五个参数，以求X
    A1 = Elli_a * (1 - Elli_e2) * (1 + 3.0 / 4.0 * Elli_e2 + 45.0 / 64.0 * Elli_e2 * Elli_e2 + 350.0 / 512.0 * Elli_e2 * Elli_e2 * Elli_e2 + 11025.0 / 16384.0 * Elli_e2 * Elli_e2 * Elli_e2 * Elli_e2);


    A2 = -0.5 * Elli_a * (1 - Elli_e2) * (3.0 / 4.0 * Elli_e2 + 60.0 / 64.0 * Elli_e2 * Elli_e2 + 525.0 / 512.0 * Elli_e2 * Elli_e2 * Elli_e2 + 17640.0 / 16384.0 * Elli_e2 * Elli_e2 * Elli_e2 * Elli_e2);


    A3 = 0.25 * Elli_a * (1 - Elli_e2) * (15.0 / 64.0 * Elli_e2 * Elli_e2 + 210.0 / 512.0 * Elli_e2 * Elli_e2 * Elli_e2 + 8820.0 / 16384.0 * Elli_e2 * Elli_e2 * Elli_e2 * Elli_e2);


    A4 = -1.0 / 6.0 * Elli_a * (1 - Elli_e2) * (35.0 / 512.0 * Elli_e2 * Elli_e2 * Elli_e2 + 2520.0 / 16384.0 * Elli_e2 * Elli_e2 * Elli_e2 * Elli_e2);


    A5 = 0.125 * Elli_a * (1 - Elli_e2) * (315.0 / 16384.0 * Elli_e2 * Elli_e2 * Elli_e2 * Elli_e2);


    double x = gauss_x - iBand * 1000000 - 500000;
    double y = gauss_y;
    double B, L;


    B00 = y / A1;
    do {
        preB0 = B00;
        B00 = B00;
        B00 = (y - (A2 * sin(2 * B00) + A3 * sin(4 * B00) + A4 * sin(6 * B00) + A5 * sin(8 * B00))) / A1;
        eta = fabs(B00 - preB0);
    } while (eta > 0.000000001);
    B = B00;
    sinB = sin(B00);
    cosB = cos(B00);
    t = tan(B00);
    t2 = t * t;
    N = Elli_a / sqrt(1 - Elli_e2 * sinB * sinB);
    ng2 = cosB * cosB * Elli_e2 / (1 - Elli_e2);
    V = sqrt(1 + ng2);
    yN = x / N;
    B = B00 - (yN * yN - (5 + 3 * t2 + ng2 - 9 * ng2 * t2) * yN * yN * yN * yN / 12.0 + (61 + 90 * t2 + 45 * t2 * t2) * yN * yN * yN * yN * yN * yN / 360.0) * V * V * t / 2;
    L = L0 + (yN - (1 + 2 * t2 + ng2) * yN * yN * yN / 6.0 + (5 + 28 * t2 + 24 * t2 * t2 + 6 * ng2 + 8 * ng2 * t2) * yN * yN * yN * yN * yN / 120.0) / cosB;


    out_longitude_angle = L * 180 / M_PI;
    out_latitude_angle = B * 180 / M_PI;
    if (out_longitude_angle > 180) {
        out_longitude_angle -= 360;
    }
    return true;
}

bool VisionGISAIEngine::ConvertLLHToGauss6(double longitude_angle, double latitude_angle, double &out_gauss_x, double &out_gauss_y, int iBand) {
    double X, N, t, t2, m, m2, ng2;
    double sinB, cosB;
    double Elli_a, Elli_f, Elli_e2, Elli_e12; // 椭球体的长半轴，扁率，第一偏心率平方，第二偏心率平方
    double A1, A2, A3, A4, A5;
    double L0; // 中央子午线经度


    if (longitude_angle < 0.0) {
        longitude_angle += 360;
    }
    if (-1 == iBand) {
        iBand = (int) (longitude_angle / 6.0) + 1;
    }

    // 84系统
    Elli_a = 6378137;
    Elli_f = 298.257223563;

    Elli_e2 = 1 - (Elli_f - 1) / Elli_f * ((Elli_f - 1) / Elli_f);
    Elli_e12 = Elli_f / (Elli_f - 1) * (Elli_f / (Elli_f - 1)) - 1;

    L0 = iBand * 6 - 3;


    //计算五个参数，以求X
    A1 = Elli_a * (1 - Elli_e2) * (1 + 3.0 / 4.0 * Elli_e2 + 45.0 / 64.0 * Elli_e2 * Elli_e2 + 350.0 / 512.0 * Elli_e2 * Elli_e2 * Elli_e2 + 11025.0 / 16384.0 * Elli_e2 * Elli_e2 * Elli_e2 * Elli_e2);


    A2 = -0.5 * Elli_a * (1 - Elli_e2) * (3.0 / 4.0 * Elli_e2 + 60.0 / 64.0 * Elli_e2 * Elli_e2 + 525.0 / 512.0 * Elli_e2 * Elli_e2 * Elli_e2 + 17640.0 / 16384.0 * Elli_e2 * Elli_e2 * Elli_e2 * Elli_e2);


    A3 = 0.25 * Elli_a * (1 - Elli_e2) * (15.0 / 64.0 * Elli_e2 * Elli_e2 + 210.0 / 512.0 * Elli_e2 * Elli_e2 * Elli_e2 + 8820.0 / 16384.0 * Elli_e2 * Elli_e2 * Elli_e2 * Elli_e2);


    A4 = -1.0 / 6.0 * Elli_a * (1 - Elli_e2) * (35.0 / 512.0 * Elli_e2 * Elli_e2 * Elli_e2 + 2520.0 / 16384.0 * Elli_e2 * Elli_e2 * Elli_e2 * Elli_e2);


    A5 = 0.125 * Elli_a * (1 - Elli_e2) * (315.0 / 16384.0 * Elli_e2 * Elli_e2 * Elli_e2 * Elli_e2);


    //求取X(地球赤道置一纬度的经线长度)
    double a1 = A1 * latitude_angle * (M_PI / 180);
    double a2 = A2 * sin(2 * latitude_angle * M_PI / 180);
    double a3 = A3 * sin(4 * latitude_angle * M_PI / 180);
    double a4 = A4 * sin(6 * latitude_angle * M_PI / 180);
    double a5 = A5 * sin(8 * latitude_angle * M_PI / 180);
    X = a1 + a2 + a3 + a4 + a5;


    sinB = sin(latitude_angle * M_PI / 180);
    cosB = cos(latitude_angle * M_PI / 180);
    t = tan(latitude_angle * M_PI / 180);
    t2 = t * t;
    N = Elli_a / sqrt(1 - Elli_e2 * sinB * sinB);
    m = cosB * (longitude_angle - L0) * M_PI / 180.0;
    m2 = m * m;
    ng2 = cosB * cosB * Elli_e2 / (1 - Elli_e2);


    out_gauss_y = X + N * t * ((0.5 + ((5 - t2 + 9 * ng2 + 4 * ng2 * ng2) / 24.0 + (61 - 58 * t2 + t2 * t2 + 270 * ng2 - 330 * ng2 * t2) * m2 / 720.0) * m2) * m2);
    out_gauss_x = N * m * (1 + m2 * ((1 - t2 + ng2) / 6.0 + m2 * m2 * (5 - 18 * t2 + t2 * t2 + 14 * ng2 - 58 * ng2 * t2) / 120.0));
    out_gauss_x = out_gauss_x + 500000 + iBand * 1000000;

    return true;
}

double VisionGISAIEngine::GetElevation(double longitude, double latitude, Mat &dem_mat, double geo_transform[6]) {
    static double result = 333;  // 若(longitude,latitude)超出了Dem的范围，则使用上一次的高程结果
    int pixel_xx, pixel_yy;
    GeographyToPixel(longitude, latitude, pixel_xx, pixel_yy, geo_transform);
    if (pixel_xx < 0 || pixel_yy < 0 || pixel_xx >= dem_mat.cols || pixel_yy >= dem_mat.rows) {
//        auto text = "The extent of terrain data [" + "Need Improve" + "GuaranteedData/*Dem.tif" + "], can not cover current position. ";
        QString text = "The extent of terrain data can not cover current position. ";
        cout << text.toLocal8Bit().data() << endl;
        SaveTextLog(text.toLocal8Bit().data());
    } else {
        result = dem_mat.at<float>(pixel_yy, pixel_xx);
    }
    return result;
}

void VisionGISAIEngine::SavePositioningPoint(QString shape_file_path, QList<QMap<QString, QString>> point_list) {
    SHPHandle outShp = SHPCreate(shape_file_path.toLocal8Bit().data(), SHPT_POINT);
    DBFHandle dbf_h = DBFCreate(shape_file_path.toLocal8Bit().data());
    DBFAddField(dbf_h, "Id", FTString, 11, 0);
    DBFAddField(dbf_h, "Longitude", FTDouble, 11, 6);
    DBFAddField(dbf_h, "Latitude", FTDouble, 11, 6);
    for (int ir = 0; ir < point_list.size(); ir++) {
        auto *xCoords = new double[1];
        auto *yCoords = new double[1];
        xCoords[0] = point_list[ir]["Longitude"].toDouble();
        yCoords[0] = point_list[ir]["Latitude"].toDouble();
        auto psShape = SHPCreateObject(SHPT_POINT, -1, 0, NULL, NULL, 1, xCoords, yCoords, NULL, NULL);
        int id = SHPWriteObject(outShp, -1, psShape);
        DBFWriteStringAttribute(dbf_h, id, 0, point_list[ir]["Code"].toLocal8Bit().data());
        DBFWriteDoubleAttribute(dbf_h, id, 1, xCoords[0]);
        DBFWriteDoubleAttribute(dbf_h, id, 2, yCoords[0]);
        SHPDestroyObject(psShape);
        delete[] xCoords;
        delete[] yCoords;
    }
    SHPClose(outShp);
    DBFClose(dbf_h);
}

// 将地面采样距离(Ground Sample Distance，单位是米，默认为横纵向相等，也就是无人机实时图的对地分辨率)转换为经纬度单位
void VisionGISAIEngine::ConvertGSDToDegrees(double gsd, double latitude, double &gsdLongitude, double &gsdLatitude) {
    double radLat = latitude * M_PI / 180.0;

    // 计算给定纬度上1度经度对应的距离（单位：米）
    double numeratorLong = std::cos(radLat) * std::sqrt(std::pow(a * a * std::cos(radLat), 2) + std::pow(b * b * std::sin(radLat), 2));
    double denominatorLong = std::sqrt(std::pow(a * std::cos(radLat), 2) + std::pow(b * std::sin(radLat), 2));
    double metersPerDegreeLongitude = (M_PI / 180.0) * numeratorLong / denominatorLong;

    // 计算给定纬度上1度纬度对应的距离（单位：米）
    double numeratorLat = std::sqrt(std::pow(a * a * std::cos(radLat), 2) + std::pow(b * b * std::sin(radLat), 2));
    double denominatorLat = std::sqrt(std::pow(a * std::cos(radLat), 2) + std::pow(b * std::sin(radLat), 2));
    double metersPerDegreeLatitude = (M_PI / 180.0) * numeratorLat / denominatorLat;

    // 进行转换
    gsdLongitude = gsd / metersPerDegreeLongitude;
    gsdLatitude = gsd / metersPerDegreeLatitude;
}

// 判断偏航角 yaw 是否在横、纵两个坐标轴的正负 yaw_tolerance 度以内
bool VisionGISAIEngine::IsYawWithinAxesRange(double yaw, double yaw_tolerance) {
    // 先将角度归一化到 0 到 360 度之间
    yaw = std::fmod(yaw, 360.0);
    if (yaw < 0) {
        yaw += 360.0;
    }

    // 判断是否在横轴（0 度和 180 度）正负 yaw_tolerance 度以内
    bool withinXAxis = (yaw <= yaw_tolerance) || (yaw >= 360 - yaw_tolerance) || ((yaw >= 180 - yaw_tolerance) && (yaw <= 180 + yaw_tolerance));
    // 判断是否在纵轴（90 度和 270 度）正负 yaw_tolerance 度以内
    bool withinYAxis = ((yaw >= 90 - yaw_tolerance) && (yaw <= 90 + yaw_tolerance)) || ((yaw >= 270 - yaw_tolerance) && (yaw <= 270 + yaw_tolerance));

    return withinXAxis || withinYAxis;
}

void VisionGISAIEngine::PointListToShp(QList<QPair<QString, QPointF>> point_list, QString &shp_file_path) {
    QFile temp(shp_file_path);
    if (temp.exists()) {
        qDebug();
    }
    SHPHandle outShp = SHPCreate(shp_file_path.toLocal8Bit().data(), SHPT_POINT);
    DBFHandle dbf_h = DBFCreate(shp_file_path.toLocal8Bit().data());
    DBFAddField(dbf_h, "SysTime", FTString, 22, 0);
    DBFAddField(dbf_h, "Longitude", FTDouble, 11, 6);
    DBFAddField(dbf_h, "Latitude", FTDouble, 11, 6);
    for (auto &item: point_list) {
        auto id = item.first;
        auto longitude = item.second.x();
        auto latitude = item.second.y();
        auto object = SHPCreateObject(SHPT_POINT, -1, 0, NULL, NULL, 1, &longitude, &latitude, NULL, NULL);
        int result = SHPWriteObject(outShp, -1, object);
        DBFWriteStringAttribute(dbf_h, result, 0, id.toLocal8Bit().data());
        DBFWriteDoubleAttribute(dbf_h, result, 1, longitude);
        DBFWriteDoubleAttribute(dbf_h, result, 2, latitude);
        SHPDestroyObject(object);
    }
    SHPClose(outShp);
    DBFClose(dbf_h);
}

void VisionGISAIEngine::ConvertFileNameToShp(QString &dir_path, QString &file_suffix, QString &shp_file_path) {
    QList<QPair<QString, QPointF>> point_list;
    QDir directory(dir_path);
    auto file_name_list = directory.entryList({file_suffix}, QDir::Files);
    auto longitude = 0.0;
    auto latitude = 0.0;
    auto relative_height = 0.0;
    auto roll = 0.0;
    auto pitch = 0.0;
    auto yaw = 0.0;
    for (int index = 0; index < file_name_list.size(); ++index) {
        auto file_name=file_name_list[index];
        ParseFileName(file_name.toLocal8Bit().data(), longitude, latitude, relative_height, roll, pitch, yaw);
        auto geometry = QPointF(longitude, latitude);
        auto point = QPair<QString, QPointF>(file_name, geometry);
        point_list.append(point);
    }
    PointListToShp(point_list, shp_file_path);
}

bool VisionGISAIEngine::IsGaussKruger3Degree(GDALDataset* dataset) {
    const char* projection = dataset->GetProjectionRef();
    if(projection == nullptr || strlen(projection) == 0) {
        return false;
    }
    OGRSpatialReference srs(projection);
    const char* projName = srs.GetAttrValue("PROJCS");

    if(projName && strstr(projName, "GK3_M_WGS84") != nullptr) {
        return true;
    }
    return false;
}


bool VisionGISAIEngine::IsWGS84(GDALDataset* dataset)
{
    const char* projection = dataset->GetProjectionRef();
    if (projection == nullptr || strlen(projection) == 0)
    {
        return false;
    }
    OGRSpatialReference srs(projection);
    const char* geogcs = srs.GetAttrValue("GEOGCS");
    if (geogcs && (strstr(geogcs, "WGS 84") != nullptr ||
        strstr(geogcs, "World Geodetic System 1984") != nullptr))
    {
        return true;
    }
    return false;
}
