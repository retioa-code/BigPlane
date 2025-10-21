// 跟界面操作和业务逻辑相关的函数
#ifndef ApplicationHelper_H
#define ApplicationHelper_H
#include "HeaderFiles.h"

void MoveCursorToButton(QPushButton* push_button);
void Log4cppPrintf(const QString& log_type, const QString& file_and_line, QString log_text);
void MakeMultilevelDir(QString folder_path);
void DeleteFilesHaveName(const QString& dir_path, const QString& name);
void DeleteFolderByQt5(const QString& dir_path);
bool DeleteFolderByQt4(const QString &dir_path);
bool CopyFolder(const QString& fromDir, const QString& toDir, bool coverFileIfExist = true);
void ReadApplicationConfigFile();
void ReloadImageTableWidget(QTableWidget* table_widget, QObject* receiver);
void ReloadNormTableWidget(QTableWidget* table_widget, QObject* receiver);
void RegisterContextMenu(QObject* parent, QObject* receiver, QString name_flag = "");
void TriggerContextMenu(QAction* action);
void TriggerContextMenu(QString command_str, QLabel* label);
bool PredictSingleFile(const QString csv_path, QStringList norm_file_content, QMap<QString, double> feature_mean_map, QMap<QString, double> feature_stddev_map);
double PredictSingleRecordByNormFile(QMap<QString, float> key_value_map, QStringList norm_file_content);
QString FindGotoId(QMap<QString, float> key_value_map, QString line_content);
bool CalcMatchOrPredictResult(QString csv_path, ExecuteFlowEnum execute_flow);
bool FilterByArtificial(QString csv_path, QString shp_path,QString match_result_path);
void OpenMatchResultImage(QString result_image_path,QTableWidget* table_widget);
void CloseMatchResultImage(QTableWidget* table_widget);
void SetCloseImageTableWidgetStyle(QTableWidget* table_widget);
void SetOpenImageTableWidgetStyle(QTableWidget* table_widget);
QString QuerySampleTypeByPixel(int pixel_x, int pixel_y, double geo_transform[6], QgsVectorLayer* vector_layer);
bool UpdateFeatureKeys(ExecuteFlowEnum execute_flow, QString norm_path);
#endif
