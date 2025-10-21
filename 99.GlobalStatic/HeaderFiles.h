#pragma once
// Windows平台：Qt5.11.2、QGIS3.8.3，Linux平台：Qt4.8.6、QGIS2.18.28
#define PlatformIsWindows  
//#define PlatformIsKylin
//#define PlatformIsNeoKylin

// QGIS头文件，按字母排序，方便查找
#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
//#include "qgis_core.h"
//#include "qgis_sip.h"
//#include <qgsabstractgeometry.h>
#include <qgsapplication.h>
//#include <qgsfeatureid.h>
//#include <qgsgeometry.h>
#include <qgsmapcanvas.h>
#include <qgsmapmouseevent.h>
#include <qgsrubberband.h>
#include <qgsmaptoolpan.h>
#include <qgsmaptooledit.h>
#include <qgsmaptoolidentify.h>
//#include <qgspoint.h>
//#include <qgspointxy.h>
#include <qgsproject.h>
#include <qgsrasterlayer.h>
#include <qgsvectorlayer.h>
#include <qgsrasteridentifyresult.h>
//#include <qgssymbol.h>
//#include <qgssinglesymbolrenderer.h>
//#include <qgsmarkersymbollayer.h>
#include <QPluginLoader>
#ifdef PlatformIsNeoKylin
#include <qgsmaplayerregistry.h>
#endif


// Gdal头文件，按字母排序，方便查找
#include <gdal_alg.h>
#include <gdal_pam.h>
#include <gdal_priv.h>
#include <gdalwarper.h>
#include <ogr_geometry.h>
#include <ogrsf_frmts.h>

// OpenCV头文件，按字母排序，方便查找
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>
//using namespace cv;  //引用了log4cpp，会有命名空间冲突，编译报错："ACCESS_MASK" ambiguous symbol

//// SqLite头文件，按字母排序，方便查找
//#include <QSqlDatabase>
//#include <QSqlQuery>
//#include <QSqlError>
//#include <QDir>

// Qt头文件，按字母排序，方便查找

#ifndef PlatformIsNeoKylin
#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#else
	#include <QtGui/QApplication>
	#include <QtGui/QMainWindow>
#endif
#include <QBitmap>
#include <QObject>
#include <QWidget>
#include <QApplication>
#include <QDomDocument>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QTextStream>
#include <QDebug>
#include <QDesktopServices>
#include <QHBoxLayout>
#include <QMovie>
#include <QMap>
#include <QList>
#include <QXmlStreamReader>
#include <QLabel>
#include <QLibrary>
#include <QTextCodec>
#include <QProcess>
#include <QThread>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QSqlError>
#include <QString>
#include <QDateTime>
//#include <QRegularExpression>
#include <QDesktopWidget>
#include <QMouseEvent>
#include <QListWidget>
#include <QSystemTrayIcon>
#include <QMenu>
#include <QPaintEvent>
#include <QColor>
#include <QPolygonF>
#include <QPainterPath>
//#include <QtConcurrent>
#include <QTableWidget>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <qevent.h>
#include <QPainter>
#include <QDialog>
#include <QButtonGroup>
#ifdef PlatformIsWindows
#include <qtdatavisualization/QtDataVisualization>
using namespace QtDataVisualization;
#endif

// C++通用头文件，按字母排序，方便查找，可能需要放在QGIS的后面
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
//#include <comutil.h>
#include <set>
#include <omp.h>
using namespace std;

// Log4cp头文件，按字母排序，方便查找
#include <log4cpp/FileAppender.hh>
#include <log4cpp/PatternLayout.hh>
#include <log4cpp/Category.hh>

#include "InfoWarningError.h"
#include "../0.ApplicationFrame/MainWindow/MainWindow.h"
#include "../0.ApplicationFrame/NotifyWidget/NotifyWidget.h"

enum ExecuteFlowEnum
{
	TrainFlow,
	PredictFlow,
	UpdateFlow
};
extern ExecuteFlowEnum ExecuteFlow;

enum WhichSystemEnum
{
	ShiErSuoNormEstimate, // 准则评估
	SiBuUpdate,   // 更新软件
	SanBuNormTrain   // 准则训练
};
extern WhichSystemEnum WhichSystem;

struct ListWidgetUserData : QObjectUserData
{
	QString DiskRelativePath;
};

extern QDomDocument AppDomDocument;
extern double TransitResolution;
extern double WorkResolution; // 勿删，客户调试机，降分辨率的代码用到了
extern int BigImageWidth;
extern int BigImageHeight;
extern int SmallImageStepX;
extern int SmallImageStepY;
extern int SmallImageWidth;
extern int SmallImageHeight;
extern double GaussianBlurKernelSize;
extern double GaussianBlurSigma;
extern double GaussianNoiseStdDev;
extern double RotateAngle;
extern double CompressionRatio;
extern bool ParallelExecutionOrNot;
extern QString FeatureKeyAll;
extern cv::Size BlockSize;
extern cv::Size BlockStride;
extern cv::Size CellSize;
extern int Bins;
extern int FeatureDimensionOfBlock; // 一个 Block 中包含的特征维度
extern QString ProgressText; // 整个程序只有一个：进度文字提示
extern int ProgressValue;
extern int ContourLengthThreshold;
extern double StretchDensityThreshold;
extern double ElevationStdDevThreshold;
extern double ElevationRangeThreshold;
extern bool UseSceneClassification;
extern double ZeroClassProportionThreshold;
extern double MainPeakValueThreshold;
extern double MainSubPeakRatioThreshold;
extern QMainWindow* MmainWindow;
extern NotifyWidget* NnotifyWidget;
extern bool HaveShowingDialog;
extern QgsMapCanvas* LeftMapCanvas;
extern QgsMapCanvas* RightMapCanvas;
extern QString SlidingColRowIndex;
extern long long TotalSlidingCount;
extern long long MatchedSlidingCount;
extern QString UnmatchedSlideWindow;
extern bool IsProductionEnvironment;
extern bool SaveIntermediateResultOrNot;
extern bool NeedStop;
extern double PredictConfidence;
