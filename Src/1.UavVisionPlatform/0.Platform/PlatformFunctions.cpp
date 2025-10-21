#include "PlatformFunctions.h"
#include "../0.VisionGISAIEngine/1.Common/EngineFunction.h"
#include "../0.VisionGISAIEngine/6.GIS/GISHelper.h"

using namespace VisionGISAIEngine;

bool UavVisionPlatform::LoadGuaranteedData(QString dom_path, QString dem_path) {
    GDALAllRegister();
    CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
    // 1. ��ȡ����Ӱ������
    auto dataset = static_cast<GDALDataset *>(GDALOpen(dom_path.toLocal8Bit().data(), GA_ReadOnly));
    if (dataset == nullptr) {
        qDebug() << "GDAL read dataset failed: " << dom_path << ". " << __LINE__ << "@" << __FILENAME__;
        exit(0);
    }
    dataset->GetGeoTransform(DomGeoTransform);
    GDALClose(dataset);
    if (FlyingTest == RunningStatus) {
        qDebug() << "Loading the dom file: " << dom_path << ". " << __LINE__ << "@" << __FILENAME__;
        BaseDomMat = imread(dom_path.toLocal8Bit().data());
    }
    // 2. ��ȡ�߳�����
    qDebug() << "Loading the dem file: " << dem_path << ". " << __LINE__ << "@" << __FILENAME__;
    // �˴�����ֱ����OpenCV��ȡ��BaseDemMat = imread(dem_path.toLocal8Bit().data()); OpenCV �� TIFF �������Դ�֧�����ޣ����ܱ�������
    // cv::TiffDecoder::readData OpenCV TIFF: TIFFRGBAImageOK: Sorry, can not handle images with 32-bit samples
    GDALDataset *poDataset = (GDALDataset *) GDALOpen(dem_path.toLocal8Bit().data(), GA_ReadOnly);
    if (poDataset == nullptr) {
        qDebug() << "GDAL read dataset failed: " << dem_path << ". " << __LINE__ << "@" << __FILENAME__;
        exit(0);
    }
    cv::Mat dem(poDataset->GetRasterYSize(), poDataset->GetRasterXSize(), CV_32F);
    poDataset->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, dem.cols, dem.rows, dem.data, dem.cols, dem.rows, GDT_Float32, 0, 0);
    poDataset->GetGeoTransform(DemGeoTransform);
    GDALClose(poDataset);
    BaseDemMat = dem.clone();
    return true;
}

bool UavVisionPlatform::LoadGuaranteedData(QString guaranteed_data_dir) {
    GDALAllRegister();
    CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
    string folder_path = guaranteed_data_dir.toLocal8Bit().data();
    if (!filesystem::exists(folder_path)) {
        cout << "Find file failed:  [" << folder_path << "] . " << __LINE__ << "@" << __FILENAME__ << endl;
        exit(0);
    }
    string dom_path = "";
    string dem_path = "";
    //
    for (auto &i: filesystem::directory_iterator(folder_path)) {
        auto file_path = QString::fromLocal8Bit(i.path().string().c_str());
        if (file_path.size() > 7 && "Dom.tif" == file_path.right(7)) {
            dom_path = file_path.toLocal8Bit().data();
        }
        if (file_path.size() > 7 && "Dem.tif" == file_path.right(7)) {
            dem_path = file_path.toLocal8Bit().data();
        }
    }
    if (dom_path.empty()) {
        cout << "In dir [" << folder_path << "] ��can not find DOM file ending with Dom.tif. " << __LINE__ << "@" << __FILENAME__ << endl;
        exit(0);
    }
    if (dem_path.empty()) {
        cout << "In dir [" << folder_path << "] , can not find DEM file ending with Dem.tif. " << __LINE__ << "@" << __FILENAME__ << endl;
        exit(0);
    }
    {
        auto dataset = static_cast<GDALDataset *>(GDALOpen(dom_path.c_str(), GA_ReadOnly));
        if (dataset == nullptr) {
            cout << "GDAL read dataset failed: " + dom_path + ". " << __LINE__ << "@" << __FILENAME__ << endl;
            exit(0);
        }
        if (!IsGaussKruger3Degree(dataset)) {
            string projection = dataset->GetProjectionRef();
            cout << "No detected coordinate system is a Gaussian system of 3 degree zone : " + projection + ". " << __LINE__ << "@" << __FILENAME__ << endl;
            exit(0);
        }
        dataset->GetGeoTransform(DomGeoTransform);
        qDebug() << "Loading the dom file: " << QString::fromStdString(dom_path) << ". " << __LINE__ << "@" << __FILENAME__;
        BaseDomMat = imread(dom_path);
    }
    {
        auto dataset = static_cast<GDALDataset *>(GDALOpen(dem_path.c_str(), GA_ReadOnly));
        if (dataset == nullptr) {
            cout << "GDAL read dataset failed: " + dem_path + ". " << __LINE__ << "@" << __FILENAME__ << endl;
            exit(0);
        }
        if (!IsWGS84(dataset)) {
            string projection = dataset->GetProjectionRef();
            cout << "No detected coordinate system is Wgs1984 : " + projection + ". " << __LINE__ << "@" << __FILENAME__ << endl;
            exit(0);
        }
        dataset->GetGeoTransform(DemGeoTransform);
        auto x_size = dataset->GetRasterXSize();
        auto y_size = dataset->GetRasterYSize();
        qDebug() << "Loading the dem file: " << QString::fromStdString(dem_path) << ". " << __LINE__ << "@" << __FILENAME__;
        BaseDemMat = Mat(y_size, x_size, CV_32FC1);
        auto count = dataset->GetRasterCount();
        dataset->GetRasterBand(count)->RasterIO(GF_Read, 0, 0, x_size, y_size, BaseDemMat.data, x_size, y_size, GDT_Float32, 0, 0);
        if (BaseDemMat.empty()) {
            cout << "OpenCV read dataset failed: " + dem_path + ". " << __LINE__ << "@" << __FILENAME__ << endl;
            exit(0);
        }
        GDALClose(dataset);
    }
    return true;
}

void UavVisionPlatform::CreateReferenceMultiple(string realtime_dir, string reference_dir, QString dom_path) {
    if (filesystem::directory_iterator(realtime_dir) == filesystem::directory_iterator{}) {
        return;
    }
    filesystem::create_directories(reference_dir);
    qDebug() << "Loading the dom file: " << dom_path << ". " << __LINE__ << "@" << __FILENAME__;
    BaseDomMat = imread(dom_path.toLocal8Bit().data());
    for (auto &i: filesystem::directory_iterator(realtime_dir)) {
        auto path = i.path().string();
        if (".raw" != path.substr(path.size() - 4) && ".jpg" != path.substr(path.size() - 4)) {
            continue;
        }
        // ������׼ͼ
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
        auto position = StdVector3(longitude, latitude, relative_height);
        auto orientation = StdVector3(roll, pitch, yaw);
        static Mat reference_image;
        int clip_left;
        int clip_top;
        CreateReferenceSingle(position, orientation, reference_image, clip_left, clip_top);
        imwrite(reference_dir + id + ".jpg", reference_image);
        string content = "ClipLeftInDom:" + to_string(clip_left) + "\n" + "ClipTopInDom:" + to_string(clip_top);
        auto f = fopen((reference_dir + id + ".txt").c_str(), "w+");
        fprintf(f, "%s", content.c_str());
        fclose(f);
    }
}

bool UavVisionPlatform::CreateReferenceSingle(StdVector3 position, StdVector3 orientation, cv::Mat &reference_image, int &clip_left, int &clip_top) {
    auto gauss_x = 0.0, gauss_y = 0.0;
    ConvertLLHToGauss3(position.X, position.Y, gauss_x, gauss_y);
    auto previous_offset = CalcGroundToAir(position.Z, StdVector3(orientation.X, orientation.Y, orientation.Z));
    // �����˻�����Ĺ���ָ��ĵ����Ϊ���ģ��ü���ͼӰ��õ���׼ͼ
    gauss_x -= previous_offset.X;
    gauss_y -= previous_offset.Y;
    int pixel_x, pixel_y;
    GeographyToPixel(gauss_x, gauss_y, pixel_x, pixel_y, DomGeoTransform);
    clip_left = pixel_x - PaddingImageCols / 2;
    clip_top = pixel_y - PaddingImageRows / 2;
    if (clip_left < 0 || clip_top < 0 || clip_left + PaddingImageCols >= BaseDomMat.cols || clip_top + PaddingImageRows >= BaseDomMat.rows) {
        auto text = "Realtime image extent is out of reference image extent. " + std::to_string(__LINE__) + "@" + __FILENAME__ + ". ";
        SaveTextLog(text);
        cout << text << endl;
        return false;
    }
    reference_image = BaseDomMat(cv::Rect(clip_left, clip_top, PaddingImageCols, PaddingImageRows));
    return true;
}

// ��ȡ��ɵ����ѹ�Ƹ߶Ⱥ͵��θ̵߳�����
bool UavVisionPlatform::InitializeLiftOff() {
#ifdef PLATFORM_JETSON
    //    if (RunningStatus == Debugging) {
//        LiftoffBarometerHeight = 68;
//        LiftoffElevation = 25;
//        return true;
//    }
//    static int num = 0;
//    if (num > 10) {
//        return true;
//    } else if (fabs(pAFTInfo->velocity.x) > 0.01 || fabs(pAFTInfo->velocity.y) > 0.01 || fabs(pAFTInfo->velocity.z) > 0.01) {
//        // ����������������ж����˻��ķ���״̬�Ƿ���û����ɣ�û�д������ԣ�ע����ȷ�ԣ���ͣ״̬�£�pAFTInfo->velocity ��֪���Ƿ�ȫΪ0
//        ifstream file((WorkDir + "LiftOffBarometerHeight.txt").toLocal8Bit().data());  // ����״̬�£�������ֹ�������ܣ�ֱ�Ӵ��ļ���ȡ
//        if (!file.is_open()) {
//            SaveSpdLog("Open liftoff barometer height txt file failed��" + to_string(__LINE__) + "@" + __FILENAME__ + ". ");
//            exit(0);
//        } else if (!(file >> LiftoffBarometerHeight)) {
//            SaveSpdLog("Read liftoff barometer height from txt file failed��" + to_string(__LINE__) + "@" + __FILENAME__ + ". ");
//            exit(0);
//        }
//        file.close();
//        auto text="Read liftoff barometer height from txt file��value is " + to_string(LiftoffBarometerHeight) + ". " + to_string(__LINE__) + "@" + __FILENAME__ + ". ";
//        SaveSpdLog(text);
//        qDebug() << QString::fromLocal8Bit(text) << __LINE__ << "@" << __FILENAME__;
//        num = 11;
//    } else if (num < 10) {  // ��1������������ȡ10����ɵ����ѹ�Ƹ߶Ⱥͺ��θ߶ȣ�ȡƽ��ֵ
//        QThread::msleep(100);
//        LiftoffBarometerHeight += pAFTInfo->altitude_barometer;
//        num++;
//        return false;
//    } else if (num == 10) {
//        LiftoffBarometerHeight /= 10;
//        auto f = fopen((WorkDir + "LiftOffBarometerHeight.txt").toLocal8Bit().data(), "w+");
//        fprintf(f, "%.2f", LiftoffBarometerHeight);
//        fclose(f);
//        num++;
//    }
//    LiftoffElevation = GetElevation(pAFTInfo->gps_info.longitude, pAFTInfo->gps_info.latitude, BaseDemMat, BaseDemGeoTransform);
#endif
    return true;
}

StdVector3 GetPosition()
{

}
