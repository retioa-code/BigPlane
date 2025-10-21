#pragma once
// Qt
#include <QtCore/QDateTime>
#include <QtCore/QDebug>
#include <QtCore/QDir>
#include <QtCore/QFile>
#include <QtCore/QLibrary>
#include <QtCore/QList>
#include <QtCore/QMap>
#include <QtCore/QObject>
#include <QtCore/QProcess>
#include <QtCore/QRegularExpression>
#include <QtCore/QString>
#include <QtCore/QTextStream>
#include <QtCore/QThread>
#include <QtCore/QXmlStreamReader>
#include <QtGui/QColor>
#include <QtGui/QMouseEvent>
#include <QtGui/QPainter>
#include <QtGui/QPainterPath>
#include <QtGui/QPaintEvent>
#include <QtGui/QPolygonF>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QSystemTrayIcon>
#include <QtWidgets/QWidget>
#include <QtXml/QDomDocument>
#include <QtWidgets/QInputDialog>
#include <QtCore/QJsonParseError>
#include <QtCore/QJsonObject>
#include <QtCore/QJsonArray>
#include <QtCore/QElapsedTimer>
#include <QtConcurrent/QtConcurrent>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <filesystem>

#ifdef WIN32

#include <windows.h>

#endif

#include "shapefil.h"

// Gdal
#include <gdal_alg.h>
#include <gdal_pam.h>
#include <gdal_priv.h>
#include <gdalwarper.h>
#include <ogr_core.h>
#include <ogr_geometry.h>
#include <ogrsf_frmts.h>
#include <spdlog/spdlog.h>
#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/basic_file_sink.h"

// 注意其他没引用HeaderFiles.h的文件，也不建议写下面的这句，可能会报错：error C2872: “byte”: 不明确的符号
using namespace std;
using namespace cv;
