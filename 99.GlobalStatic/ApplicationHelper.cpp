# pragma execution_character_set("utf-8")
#include "ApplicationHelper.h"

#include <QHeaderView>

#include "GISHelper.h"
#include "OpenCVHelper.h"
#include "SQLiteHelper.h"
#include "qurl.h"

void MoveCursorToButton(QPushButton* push_button)
{
	QPoint point = push_button->mapToGlobal(QPoint(0, 0));
	QPoint button_center_point(point.x() + push_button->width() / 2, point.y() + push_button->height() / 2);
	QCursor::setPos(button_center_point);
	push_button->setFocus();
}

/**
*brief 写日志
*param log_type 日志类型（ERROR | ANALYZE）
*param log_text 间隔距离
*/
void Log4cppPrintf(const QString& log_type, const QString& file_and_line, QString log_text)
{
	// 暂不删除
	static int need_print_flag = -1;
	if (-1 == need_print_flag)
	{
		need_print_flag = AppDomDocument.documentElement().firstChildElement("NeedPrintLog").text().toInt();
	}
	if (need_print_flag != 1) return;
	QString file_path = QCoreApplication::applicationDirPath() + "/" + log_type + ".log";
	QString file_dir = file_path;
	file_dir.replace('\\', '/');
	file_dir = file_dir.left(file_dir.lastIndexOf("/"));
	QDir dir(file_dir);
	if (!dir.exists()) MakeMultilevelDir(file_path);
	const QByteArray file_path_array = file_path.toLocal8Bit();
	const char* file_path_data = file_path_array.data();

	log4cpp::FileAppender* appender = new log4cpp::FileAppender("fileAppender", file_path_data, true, 00644);
	log4cpp::PatternLayout* pLayout = new log4cpp::PatternLayout();
	pLayout->setConversionPattern("%d: %p %c %x: %m%n");
	appender->setLayout(pLayout);

	log4cpp::Category& root = log4cpp::Category::getRoot();
	log4cpp::Category& infoCategory = root.getInstance("");
	infoCategory.addAppender(appender);
	infoCategory.setPriority(log4cpp::Priority::INFO);

	log_text = file_and_line + log_text + "\n";
	const QByteArray text = log_text.toLocal8Bit();
	const char* data = text.data();
	if (log_type == "ERROR")
	{
		infoCategory.error(data);
	}
	else if (log_type == "ANALYZE")
	{
		infoCategory.info(data);
	}
	else if (log_type == "3DSCENEANALYZE")
	{
		infoCategory.info(data);
	}

	log4cpp::Category::shutdown();
}

/**
* \brief 如果多级文件目录不存在，就创建文件目录
*/
void MakeMultilevelDir(QString folder_path)
{
	folder_path.replace('\\', '/');
	QDir dir(folder_path);
	QStringList dir_list = dir.absolutePath().split('/');
	QString dir_path;
	for (int i = 0; i < dir_list.count(); i++)
	{
		dir_path += dir_list.at(i) + '/';
		QDir dir_temp;
		bool ret = dir_temp.exists(dir_path);
		if (!ret)
		{
			dir_temp.mkdir(dir_path);
		}
	}
}

/**
* \brief 删除文件夹中所有文件名包含给定字符串的文件
*/
void DeleteFilesHaveName(const QString& dir_path, const QString& name)
{
	QDir dir(dir_path);
	QStringList nameFilters("*.*");
	QStringList file_name_list = dir.entryList(nameFilters);
	for (int i = 0; i < file_name_list.size(); ++i)
	{
		QString file_path = file_name_list.at(i);
		if (!file_path.contains(name))
			continue;
		file_path = dir_path + file_path;
		QFile::remove(file_path);
	}
}

/**
* \brief 删除多级文件夹
*/
void DeleteFolderByQt5(const QString& dir_path)
{
#ifndef PlatformIsNeoKylin
	QDir dir(dir_path);
	if (!dir.exists())
	{
		return;
	}
	dir.removeRecursively();
#endif
}

//qt4递归删除文件夹
bool DeleteFolderByQt4(const QString& dir_path)
{
	bool result = true;
	QDir dir(dir_path);
	if (dir.exists(dir_path))
	{
		QFileInfoList infolist = dir.entryInfoList();
		for (int i = 0; i < infolist.count(); i++)
		{
			QFileInfo info = infolist.at(i);
			QString file_name = info.fileName();
			if ("." == file_name || file_name == "..") continue; //防止本层无限进归,这函数真坑
			if (info.isDir())
			{
				result = DeleteFolderByQt4(info.absoluteFilePath());
			}
			else
			{
				result = QFile::remove(info.absoluteFilePath());
			}
			if (!result)
			{
				return result;
			}
		}
		result = dir.rmdir(dir_path);
	}
	return result;
}

//************************************
// 方法名称:	copyFolder
// 概要:		复制文件夹
// 返回值:		bool
// 参数:		const QString & fromDir 原路径
// 参数:		const QString & toDir 新路径
// 参数:		bool coverFileIfExist 如果存在是否覆盖
//************************************
bool CopyFolder(const QString& fromDir, const QString& toDir, bool coverFileIfExist)
{
	QDir sourceDir(fromDir);
	QDir targetDir(toDir);

	if (!targetDir.exists())
	{    //如果目标目录不存在，则进行创建 
		if (!targetDir.mkdir(targetDir.absolutePath())) return false;
	}

	QFileInfoList fileInfoList = sourceDir.entryInfoList();
	foreach(QFileInfo fileInfo, fileInfoList)
	{
		if (fileInfo.fileName() == "." || fileInfo.fileName() == "..") continue;

		if (fileInfo.isDir())
		{    // 当为目录时，递归的进行copy 
			if (!CopyFolder(fileInfo.filePath(),
				targetDir.filePath(fileInfo.fileName()),
				coverFileIfExist))
				return false;
		}
		else
		{   //当允许覆盖操作时，将旧文件进行删除操作
			if (coverFileIfExist && targetDir.exists(fileInfo.fileName()))
			{
				targetDir.remove(fileInfo.fileName());
			}

			// 进行文件拷贝
			if (!QFile::copy(fileInfo.filePath(), targetDir.filePath(fileInfo.fileName())))
			{
				return false;
			}
		}
	}
	return true;
}

void ReadApplicationConfigFile()
{
	QDomElement dom_element = AppDomDocument.documentElement();
	TransitResolution = dom_element.firstChildElement("TransitResolution").text().toDouble();
	WorkResolution = dom_element.firstChildElement("WorkResolution").text().toDouble();
	BigImageWidth = dom_element.firstChildElement("BigImageWidth").text().toDouble();
	BigImageHeight = dom_element.firstChildElement("BigImageHeight").text().toDouble();
	SmallImageStepX = dom_element.firstChildElement("SmallImageStepX").text().toDouble();
	SmallImageStepY = dom_element.firstChildElement("SmallImageStepY").text().toDouble();
	SmallImageWidth = dom_element.firstChildElement("SmallImageWidth").text().toDouble();
	SmallImageHeight = dom_element.firstChildElement("SmallImageHeight").text().toDouble();
	int block_size_width = dom_element.firstChildElement("BlockSizeWidth").text().toInt();
	int block_size_height = dom_element.firstChildElement("BlockSizeHeight").text().toInt();
	BlockSize = cv::Size(block_size_width, block_size_height);
	int block_stride_width = dom_element.firstChildElement("BlockStrideWidth").text().toInt();
	int block_stride_height = dom_element.firstChildElement("BlockStrideHeight").text().toInt();
	BlockStride = cv::Size(block_stride_width, block_stride_height);
	int cell_size_width = dom_element.firstChildElement("CellSizeWidth").text().toInt();
	int cell_size_height = dom_element.firstChildElement("CellSizeHeight").text().toInt();
	CellSize = cv::Size(cell_size_width, cell_size_height);
	Bins = dom_element.firstChildElement("Bins").text().toInt();
	FeatureDimensionOfBlock = (block_size_width / cell_size_width) * (block_size_height / cell_size_height) * Bins;
	ContourLengthThreshold = dom_element.firstChildElement("ContourLengthThreshold").text().toInt();
	ElevationStdDevThreshold = dom_element.firstChildElement("ElevationStdDevThreshold").text().toDouble();
	ElevationRangeThreshold = dom_element.firstChildElement("ElevationRangeThreshold").text().toDouble();
	UseSceneClassification = dom_element.firstChildElement("UseSceneClassification").text().toInt() == 1;
	ZeroClassProportionThreshold = dom_element.firstChildElement("ZeroClassProportionThreshold").text().toDouble();
	MainPeakValueThreshold = dom_element.firstChildElement("MainPeakValueThreshold").text().toDouble();
	MainSubPeakRatioThreshold = dom_element.firstChildElement("MainSubPeakRatioThreshold").text().toDouble();
	SlidingColRowIndex = dom_element.firstChildElement("SlidingColRowIndex").text();
	SlidingColRowIndex.remove("\t").remove("\n").remove(" ");
	QString title = dom_element.firstChildElement("SystemName").text();
	if("SPXFXandGXPG" == title)
	{
		WhichSystem = SiBuUpdate;
	}
	else if("FEandNE" == title)
	{
		WhichSystem = ShiErSuoNormEstimate;
	}
	else if("NormTrain" == title)
	{
		WhichSystem = SanBuNormTrain;
	}
	else
	{
		QString text = "系统名称配置不正确，请修改“/ConfigurationFiles/ApplicationConfig.xml”里面的“SystemName”节点后，重试。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowWindow(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		exit(0);
	}
	IsProductionEnvironment = dom_element.firstChildElement("IsProductionEnvironment").text().toInt() == 1;
	ParallelExecutionOrNot = dom_element.firstChildElement("ParallelExecutionOrNot").text().toInt() == 1;
	SaveIntermediateResultOrNot = dom_element.firstChildElement("SaveIntermediateResultOrNot").text().toInt() == 1;
	PredictConfidence = dom_element.firstChildElement("PredictConfidence").text().toDouble();
}

void ReloadImageTableWidget(QTableWidget* table_widget, QObject* receiver)
{
	// 先清空，再插入
	while(table_widget->rowCount()>0)
	{
		table_widget->removeRow(0);
	}
	QVariantMap where_clause_equal;
	QString type = table_widget->toolTip();
	if(SiBuUpdate == WhichSystem) type = "影像训练集";
	where_clause_equal.insert("Type", type);
	QSqlQuery sql_query = SelectRow("Image", &where_clause_equal, "Name");
	while (sql_query.next())
	{
		table_widget->insertRow(table_widget->rowCount());
		QString id = sql_query.value(0).toString();  // Id，在Qt5中可写成：sql_query.value("Id").toString();
		QString relative_path = sql_query.value(2).toString(); // DiskRelativePath

		QTableWidgetItem* image_name_item = new QTableWidgetItem(sql_query.value(1).toString()); // Name
		image_name_item->setData(1, id); // Id，在Qt5中可写成：sql_query.value("Id").toString();
		image_name_item->setData(11, relative_path); // DiskRelativePath
		image_name_item->setFlags(image_name_item->flags() & (~Qt::ItemIsEditable));
		table_widget->setItem(table_widget->rowCount() - 1, 0, image_name_item);

		if("样本修改-影像训练集" != table_widget->statusTip())
		{
			QLabel* open_label = new QLabel("<a href = " + id + " >打 开</a>", table_widget);
			open_label->setStatusTip(id);
			open_label->setToolTip(relative_path);
			open_label->setAlignment(Qt::AlignCenter);
			open_label->setFont(QFont("song", 12));
			QObject::connect(open_label, SIGNAL(linkActivated(QString)), receiver, SLOT(OpenFilePositonSlot(QString)));
			table_widget->setCellWidget(table_widget->rowCount() - 1, table_widget->columnCount()-2, open_label);

			QLabel* delete_label = new QLabel("<a href = " + id + " >删 除</a>", table_widget);
			delete_label->setFont(QFont("song", 12));
			delete_label->setAlignment(Qt::AlignCenter);
			QObject::connect(delete_label, SIGNAL(linkActivated(QString)), receiver, SLOT(DeleteImageSlot(QString)));
			table_widget->setCellWidget(table_widget->rowCount() - 1, table_widget->columnCount()-1, delete_label);
		}
		else
		{
			QLabel* open_result = new QLabel("<a href = " + id + " >打开结果</a>", table_widget);
			open_result->setStatusTip(id);
			open_result->setToolTip(relative_path);
			open_result->setAlignment(Qt::AlignCenter);
			open_result->setFont(QFont("song", 12));
			QObject::connect(open_result, SIGNAL(linkActivated(QString)), receiver, SLOT(OpenMatchResultSlot(QString)));
			table_widget->setCellWidget(table_widget->rowCount() - 1, 1, open_result);

			QLabel* close_result = new QLabel("<a href = " + id + " >关闭结果</a>", table_widget);
			close_result->setStatusTip(id);
			close_result->setToolTip(relative_path);
			close_result->setAlignment(Qt::AlignCenter);
			close_result->setFont(QFont("song", 12));
			QObject::connect(close_result, SIGNAL(linkActivated(QString)), receiver, SLOT(CloseMatchResultSlot(QString)));
			table_widget->setCellWidget(table_widget->rowCount() - 1, 2, close_result);

			QLabel* open_location_label = new QLabel("<a href = " + id + " >打开位置</a>", table_widget);
			open_location_label->setStatusTip(id);
			open_location_label->setToolTip(relative_path);
			open_location_label->setAlignment(Qt::AlignCenter);
			open_location_label->setFont(QFont("song", 12));
			QObject::connect(open_location_label, SIGNAL(linkActivated(QString)), receiver, SLOT(OpenFilePositonSlot(QString)));
			table_widget->setCellWidget(table_widget->rowCount() - 1, 3, open_location_label);

			QLabel* delete_label = new QLabel("<a href = " + id + " >删 除</a>", table_widget);
			delete_label->setFont(QFont("song", 12));
			delete_label->setAlignment(Qt::AlignCenter);
			QObject::connect(delete_label, SIGNAL(linkActivated(QString)), receiver, SLOT(DeleteImageSlot(QString)));
			table_widget->setCellWidget(table_widget->rowCount() - 1, 4, delete_label);
		}
	}
	// 样式修改
	if("样本修改-影像训练集" != table_widget->statusTip())
	{
		table_widget->setColumnWidth(0, 333);
		table_widget->setColumnWidth(1, 115);
		table_widget->setColumnWidth(2, 80);
	}
	else
	{
		table_widget->setColumnWidth(0, 333);
		table_widget->setColumnWidth(1, 115);
		table_widget->setColumnWidth(2, 115);
		table_widget->setColumnWidth(3, 115);
		table_widget->setColumnWidth(4, 80);
	}
	table_widget->horizontalHeader()->setFont(QFont("song", 12));
	table_widget->setStyleSheet("selection-background-color: rgb(166,166,166)");
	// 启用右键菜单
	table_widget->setContextMenuPolicy(Qt::CustomContextMenu);
	QObject::connect(table_widget, SIGNAL(customContextMenuRequested(const QPoint &)),
		receiver, SLOT(ShowImageTableWidgetMenuSlot(const QPoint &)));
}

void ReloadNormTableWidget(QTableWidget* table_widget, QObject* receiver)
{
	// 先清空，再插入
	while(table_widget->rowCount()>0)
	{
		table_widget->removeRow(0);
	}
	QVariantMap where_clause_equal;
	QString type = table_widget->toolTip();
	QSqlQuery sql_query = SelectRow("Norm", &where_clause_equal, "Name");
	while (sql_query.next())
	{
		table_widget->insertRow(table_widget->rowCount());

		QTableWidgetItem* norm_name_item = new QTableWidgetItem(sql_query.value(1).toString()); // Name
		norm_name_item->setFlags(norm_name_item->flags() & (~Qt::ItemIsEditable));
		table_widget->setItem(table_widget->rowCount() - 1, 0, norm_name_item);

		QString relative_path = sql_query.value(2).toString(); // DiskRelativePath
		QFile file(QCoreApplication::applicationDirPath() + relative_path + "_5.ModelEvaluation.csv");
		if (file.exists() && file.open(QFile::ReadOnly))
		{
			QTextStream stream(&file);
			QString read_line = stream.readLine();
			QString text = read_line.split(",")[1];
			QTableWidgetItem* accuracy_item = new QTableWidgetItem(text);
			accuracy_item->setFlags(accuracy_item->flags() & (~Qt::ItemIsEditable));
			table_widget->setItem(table_widget->rowCount() - 1, 1, accuracy_item);
			file.close();
		}

		QString id = sql_query.value(0).toString();  // Id，在Qt5中可写成：sql_query.value("Id").toString();
		QLabel* open_label = new QLabel("<a href = " + id + " >打 开</a>", table_widget);
		open_label->setStatusTip(id);
		open_label->setToolTip(relative_path);
		open_label->setAlignment(Qt::AlignCenter);
		open_label->setFont(QFont("song", 12));
		QObject::connect(open_label, SIGNAL(linkActivated(QString)), receiver, SLOT(OpenFilePositonSlot(QString)));
		table_widget->setCellWidget(table_widget->rowCount() - 1, 2, open_label);

		QLabel* delete_label = new QLabel("<a href = " + id + " >删 除</a>", table_widget);
		delete_label->setFont(QFont("song", 12));
		delete_label->setAlignment(Qt::AlignCenter);
		QObject::connect(delete_label, SIGNAL(linkActivated(QString)), receiver, SLOT(DeleteNormSlot(QString)));
		table_widget->setCellWidget(table_widget->rowCount() - 1, 3, delete_label);
	}
	// 样式修改
	table_widget->setColumnWidth(0, 333);
	table_widget->setColumnWidth(1, 90);
	table_widget->setColumnWidth(2, 115);
	table_widget->setColumnWidth(3, 80);
	table_widget->horizontalHeader()->setFont(QFont("song", 12));
	table_widget->setStyleSheet("selection-background-color:  rgb(166,166,166)");
	// 启用右键菜单
	table_widget->setContextMenuPolicy(Qt::CustomContextMenu);
	QObject::connect(table_widget, SIGNAL(customContextMenuRequested(const QPoint &)),
		receiver, SLOT(ShowNormTableWidgetMenuSlot(const QPoint &)));
	table_widget->selectRow(0);
}

void RegisterContextMenu(QObject* sender, QObject* receiver, QString name_flag)
{
	if ("数据管理-影像训练集" == name_flag)
	{
		QMenu image_menu;
		QAction action("打开文件位置", sender);
		image_menu.addAction(&action);
		QAction action11("删除影像", sender);
		image_menu.addAction(&action11);
		QObject::connect(&image_menu, SIGNAL(triggered(QAction*)), receiver, SLOT(TriggerContextMenuSlot(QAction*)));
		image_menu.exec(QCursor::pos());
	}
	else if ("样本集生成-影像训练集" == name_flag)
	{
		QMenu image_menu;
		QAction action("打开文件位置", sender);
		image_menu.addAction(&action);
		QAction action11("删除影像", sender);
		image_menu.addAction(&action11);
		QObject::connect(&image_menu, SIGNAL(triggered(QAction*)), receiver, SLOT(TriggerContextMenuSlot(QAction*)));
		image_menu.exec(QCursor::pos());
	}
	else if ("样本修改-影像训练集" == name_flag)
	{
		QMenu image_menu;
		QAction action("打开匹配结果", sender);
		image_menu.addAction(&action);
		QAction action00("打开文件位置", sender);
		image_menu.addAction(&action00);
		QAction action11("删除影像", sender);
		image_menu.addAction(&action11);
		QObject::connect(&image_menu, SIGNAL(triggered(QAction*)), receiver, SLOT(TriggerContextMenuSlot(QAction*)));
		image_menu.exec(QCursor::pos());
	}
	else if ("准则挖掘-影像训练集" == name_flag)
	{
		QMenu image_menu;
		QAction action("打开文件位置", sender);
		image_menu.addAction(&action);
		QAction action11("删除影像", sender);
		image_menu.addAction(&action11);
		QObject::connect(&image_menu, SIGNAL(triggered(QAction*)), receiver, SLOT(TriggerContextMenuSlot(QAction*)));
		image_menu.exec(QCursor::pos());
	}
	else if ("准则挖掘-已有预测准则" == name_flag)
	{
		QMenu image_menu;
		QAction action("打开文件位置", sender);
		image_menu.addAction(&action);
		QAction action11("删除准则", sender);
		image_menu.addAction(&action11);
		QObject::connect(&image_menu, SIGNAL(triggered(QAction*)), receiver, SLOT(TriggerContextMenuSlot(QAction*)));
		image_menu.exec(QCursor::pos());
	}
	else if ("适配区筛选-影像测试集" == name_flag)
	{
		QMenu image_menu;
		QAction action("打开文件位置", sender);
		image_menu.addAction(&action);
		QAction action11("删除影像", sender);
		image_menu.addAction(&action11);
		QObject::connect(&image_menu, SIGNAL(triggered(QAction*)), receiver, SLOT(TriggerContextMenuSlot(QAction*)));
		image_menu.exec(QCursor::pos());
	}
	else if ("适配区筛选-预测准则模型" == name_flag)
	{
		QMenu image_menu;
		QAction action("打开文件位置", sender);
		image_menu.addAction(&action);
		QAction action11("删除准则", sender);
		image_menu.addAction(&action11);
		QObject::connect(&image_menu, SIGNAL(triggered(QAction*)), receiver, SLOT(TriggerContextMenuSlot(QAction*)));
		image_menu.exec(QCursor::pos());
	}
}

void TriggerContextMenu(QAction* action)
{
	QTableWidget* table_widget = static_cast<QTableWidget*>(action->parent());
	if (table_widget->rowCount() == 0) return;
	QWidget* open_folder_widget = table_widget->cellWidget(table_widget->currentRow(), table_widget->columnCount() - 2); // 列序号为1，表示打开文件位置那一列
	QString id = open_folder_widget->statusTip();
	QString file_path = QCoreApplication::applicationDirPath() + open_folder_widget->toolTip();
	QString folder_path = file_path.left(file_path.lastIndexOf("/"));
	if ("打开文件位置" == action->text())
	{
		QDir dir(folder_path);
		if (dir.exists())
		{
			QDesktopServices::openUrl(QUrl("file:///" + folder_path));
		}
		else
		{
			QString text = "打开目录失败：" + folder_path + "。";
			Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
			InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		}
	}
	else if ("删除影像" == action->text())
	{
		QMessageBox message_box(QMessageBox::Question, "提 示", "删除后将无法恢复，是否继续？", NULL);
		message_box.addButton(" 是 ", QMessageBox::AcceptRole);
		message_box.addButton(" 否 ", QMessageBox::RejectRole);
		if (0 != message_box.exec()) return;
		// 磁盘文件中删除
		DeleteFolderByQt4(folder_path);
		// 数据库中删除
		QVariantMap field_value_map;
		field_value_map.insert("Id", id);
		DeleteRow("Image", &field_value_map);
		// 界面上删除
		table_widget->removeRow(table_widget->currentRow());
	}
	else if ("打开匹配结果" == action->text())
	{
		OpenMatchResultImage(file_path,table_widget);
	}
	else if ("关闭匹配结果" == action->text())
	{
		CloseMatchResultImage(table_widget);
	}
	else if ("删除准则" == action->text())
	{
		QMessageBox message_box(QMessageBox::Question, "提 示", "删除后将无法恢复，是否继续？", NULL);
		message_box.addButton(" 是 ", QMessageBox::AcceptRole);
		message_box.addButton(" 否 ", QMessageBox::RejectRole);
		if (0 != message_box.exec()) return;
		// 磁盘文件中删除
		DeleteFolderByQt4(folder_path);
		// 数据库中删除
		QVariantMap field_value_map;
		field_value_map.insert("Id", id);
		DeleteRow("Norm", &field_value_map);
		// 界面上删除
		table_widget->removeRow(table_widget->currentRow());
	}
}

void TriggerContextMenu(QString command_str, QLabel* label)
{
	QTableWidget* table_widget = (QTableWidget*)(label->parent()->parent());
	if (table_widget->rowCount() == 0) return;
	QWidget* widget = table_widget->cellWidget(table_widget->currentRow(), table_widget->columnCount() - 2); // 倒是第2列是打开文件位置那一列
	QString id = widget->statusTip();
	QString file_path = QCoreApplication::applicationDirPath() + widget->toolTip();
	QString folder_path = file_path.left(file_path.lastIndexOf("/"));
	if ("打开文件位置" == command_str)
	{
		QString relative_path = label->toolTip();
		QString path = QCoreApplication::applicationDirPath() + relative_path.left(relative_path.lastIndexOf("/"));
		QDir dir(path);
		if (dir.exists())
		{
			QDesktopServices::openUrl(QUrl("file:///" + path));
		}
		else
		{
			QString text = "打开目录失败：" + path + "。";
			Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
			InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		}
	}
	else if ("删除影像" == command_str)
	{
		QMessageBox message_box(QMessageBox::Question, "提 示", "删除后将无法恢复，是否继续？", NULL);
		message_box.addButton(" 是 ", QMessageBox::AcceptRole);
		message_box.addButton(" 否 ", QMessageBox::RejectRole);
		if (0 != message_box.exec()) return;
		// 磁盘文件中删除
		DeleteFolderByQt4(folder_path);
		// 数据库中删除
		QVariantMap field_value_map;
		field_value_map.insert("Id", id);
		DeleteRow("Image", &field_value_map);
		// 界面上删除
		table_widget->removeRow(table_widget->currentRow());
	}
	else if ("打开匹配结果" == command_str)
	{
		OpenMatchResultImage(file_path,table_widget);
	}
	else if ("关闭匹配结果" == command_str)
	{
		CloseMatchResultImage( table_widget);
	}
	else if ("删除准则" == command_str)
	{
		QMessageBox message_box(QMessageBox::Question, "提 示", "删除后将无法恢复，是否继续？", NULL);
		message_box.addButton(" 是 ", QMessageBox::AcceptRole);
		message_box.addButton(" 否 ", QMessageBox::RejectRole);
		if (0 != message_box.exec()) return;
		// 磁盘文件中删除
		DeleteFolderByQt4(folder_path);
		// 数据库中删除
		QVariantMap field_value_map;
		field_value_map.insert("Id", id);
		DeleteRow("Norm", &field_value_map);
		// 界面上删除
		table_widget->removeRow(table_widget->currentRow());
	}
}

bool PredictSingleFile(const QString csv_path, QStringList norm_file_content, QMap<QString, double> feature_mean_map, QMap<QString, double> feature_stddev_map)
{
	QFile csv_file(csv_path);
	if (!csv_file.open(QFile::ReadOnly))
	{
		QString text = "打开文件失败：" + csv_path + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return false;
	}
	QTextStream read_text_stream(&csv_file);
	int total_line_count = read_text_stream.readAll().split("\n").size() - 1;
	read_text_stream.seek(0);
	QString new_csv_text = read_text_stream.readLine(); // 第一行是列头
	QMap<QString, float> key_value_map;
	QStringList column_name_split = new_csv_text.split(",");
	int increasing_number = 0;
	while (!read_text_stream.atEnd())
	{
		QString old_line_str = read_text_stream.readLine();
		QStringList old_line_split = old_line_str.split(",");
		for (int index = 0; index < old_line_split.size(); ++index)
		{
			if(index>=column_name_split.size()) break;
			key_value_map[column_name_split.at(index)] = old_line_split.at(index).toFloat(); // 更新若干个特征名和值的键值对，读取自csv文件
		}
		// 读取原始的图像特征指标值后，按训练时的数据集中的特征均值和标准差，做归一化
		for (int index = 0; index < feature_mean_map.size(); ++index)
		{
			QString key = key_value_map.keys().at(index);
			key_value_map[key] = (key_value_map[key] - feature_mean_map[key]) / feature_stddev_map[key];
		}
		QString new_line_str = old_line_str.left(old_line_str.lastIndexOf(",") + 1);
		double predict_result = PredictSingleRecordByNormFile(key_value_map, norm_file_content);
		new_csv_text += "\n" + new_line_str + QString::number(predict_result,10,6);
		ProgressValue = ++increasing_number * 100.0 / total_line_count;
	}
	csv_file.close();
	if (!csv_file.open(QFile::WriteOnly | QFile::Truncate))
	{
		QString text = "以写入方式打开文件失败：" + csv_path + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return false;
	}
	QTextStream write_text_stream(&csv_file);
	write_text_stream << new_csv_text;
	csv_file.close();
	return true;
}

// key_value_map 和 norm_file_content 中的特征个数,不需要完全一样，按特征名称从 key_value_map 中按需读取
double PredictSingleRecordByNormFile(QMap<QString, float> key_value_map, QStringList norm_file_content)
{
	double score_summation = 0.0;
	QString goto_id = "";
	for (int index = 0; index < norm_file_content.size(); ++index)
	{
		QString line_content = norm_file_content.at(index);
		if ("0:[" == line_content.left(3)) // 每颗小树的根节点索引号是"0"
		{
			goto_id = FindGotoId(key_value_map, line_content);
			while (true)
			{
				line_content = norm_file_content.at(++index);
				if (goto_id != line_content.split(":")[0]) continue;
				if (line_content.contains("leaf"))
				{
					score_summation += line_content.split("=")[1].toDouble();
					break;
				}
				goto_id = FindGotoId(key_value_map, line_content);
			}
		}
	}
	double predict_score = 1.0 / (1 + exp(-score_summation));  // 使用 Sigmoid 公式预测分类概率
	return predict_score;
}

// line_content 的内容示例一：3:[FourierBW2<-0.0704023167] yes=7,no=8,missing=7
// 根据 key_value_map 特征列表中的 FourierBW2 的值的大小，查找到是走yes分支还是no分支
QString FindGotoId(QMap<QString, float> key_value_map, QString line_content)
{
	QStringList goto_where = line_content.split(" ")[1].split(",");
	QString goto_yes_id = goto_where[0].split("=")[1];
	QString goto_no_id = goto_where[1].split("=")[1];
	QStringList feature_compare = line_content.split("]").at(0).split("[").at(1).split("<");
	QString feature_name = feature_compare.at(0);
	QString feature_value = feature_compare.at(1);
	if (key_value_map[feature_name] < feature_value.toDouble())
	{
		return goto_yes_id;
	}
	return goto_no_id;
}

// 通过第一层筛选的几个图像特征的阈值，筛选一遍
bool CalcMatchOrPredictResult(QString csv_path, ExecuteFlowEnum execute_flow)
{
	QFile csv_file(csv_path);
	if (!csv_file.open(QFile::ReadOnly))
	{
		QString text = "打开文件失败：" + csv_path + "。请提取特征后，重试。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return false;
	}
	// 区分训练流程和测试流程，读取用图像特征进行筛选的映射表
	QString node_name = "FeatureFilterMapListPredict";
	if (TrainFlow == execute_flow)
	{
		node_name = "FeatureFilterMapListTrain";
	}
	// QList表示多个特征值的比较阈值列表，QString表示特征值名称，QString表示大于还是小于，double表示阈值数字
	QList<QPair<QString, QPair<QString, float> > > feature_filter_map_list11;
	QList<QPair<QString, QPair<QString, float> > > feature_filter_map_list22;
	QDomElement dom_element = AppDomDocument.documentElement();
	QDomElement element11 = dom_element.firstChildElement(node_name+"11").firstChildElement("Element");
	while (!element11.isNull())
	{
		QString feature_name = element11.attribute("FeatureName");
		QString great_or_less = element11.attribute("GreatOrLess");
		float threshold = element11.attribute("threshold").toFloat();
		QPair<QString, QPair<QString, float> > pair = QPair<QString, QPair<QString, float> >(feature_name, QPair<QString, float>(great_or_less, threshold));
		feature_filter_map_list11.append(pair);
		element11 = element11.nextSiblingElement();
	}
	QDomElement element22 = dom_element.firstChildElement(node_name+"22").firstChildElement("Element");
	while (!element22.isNull())
	{
		QString feature_name = element22.attribute("FeatureName");
		QString great_or_less = element22.attribute("GreatOrLess");
		float threshold = element22.attribute("threshold").toFloat();
		QPair<QString, QPair<QString, float> > pair = QPair<QString, QPair<QString, float> >(feature_name, QPair<QString, float>(great_or_less, threshold));
		feature_filter_map_list22.append(pair);
		element22 = element22.nextSiblingElement();
	}
	QTextStream read_text_stream(&csv_file);
	QString new_csv_text = read_text_stream.readLine(); // 第一行是列头
	QMap<QString, float> key_value_map;
	QStringList column_name_split = new_csv_text.split(",");
	while (!read_text_stream.atEnd())
	{
		QString old_line_str = read_text_stream.readLine();
		QStringList old_line_split = old_line_str.split(",");
		for (int index = 0; index < old_line_split.size(); ++index)
		{
			key_value_map[column_name_split.at(index)] = old_line_split.at(index).toFloat(); // 更新若干个特征名和值的键值对，读取自csv文件
		}
		QString new_line_str = old_line_str.left(old_line_str.lastIndexOf(",") + 1);
		if (PredictFlow == execute_flow)
		{
			double predict_value = key_value_map["PredictProbability"];
			if(predict_value < PredictConfidence)
			{
				new_csv_text += "\n" + new_line_str + QString::number(predict_value, 10, 6);
				continue; // 如个是预测流程，并且预测结果小于0.5，则直接返回
			}
		}
		double match_or_predict = 1.0;  
		if(key_value_map.keys().contains("PredictProbability"))
		{
			match_or_predict = key_value_map["PredictProbability"];
		}
		bool need_update11 = false;
		bool need_update22 = false;
		for (int index = 0; index < feature_filter_map_list11.size(); ++index)
		{
			QPair<QString, QPair<QString, float> > pair = feature_filter_map_list11.at(index);
			QString feature_name = pair.first;
			if (!key_value_map.contains(feature_name))
			{
				continue;
			}
			float feature_value = key_value_map[feature_name];
			QString great_or_less = pair.second.first;
			float threshold = pair.second.second;
			if (great_or_less == "GreaterEqual" && feature_value < threshold || great_or_less == "LessEqual" && feature_value > threshold)
			{
				need_update11 = true;
				break;
			}
		}
		for (int index = 0; index < feature_filter_map_list22.size(); ++index)
		{
			QPair<QString, QPair<QString, float> > pair = feature_filter_map_list22.at(index);
			QString feature_name = pair.first;
			if (!key_value_map.contains(feature_name))
			{
				continue;
			}
			float feature_value = key_value_map[feature_name];
			QString great_or_less = pair.second.first;
			float threshold = pair.second.second;
			if (great_or_less == "GreaterEqual" && feature_value < threshold || great_or_less == "LessEqual" && feature_value > threshold)
			{
				need_update22 = true;
				break;
			}
		}
		if(need_update11 && need_update22)
		{
			match_or_predict = 0;
		}
		new_csv_text += "\n" + new_line_str + QString::number(match_or_predict,10,6);
	}
	csv_file.close();
	if (!csv_file.open(QFile::WriteOnly | QFile::Truncate))
	{
		QString text = "以写入方式打开文件失败：" + csv_path + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return false;
	}
	QTextStream write_text_stream(&csv_file);
	write_text_stream << new_csv_text;
	csv_file.close();
	return true;
}

// 样本集生成时，提取特征和用算法匹配后，再人工对样本修改一遍
bool FilterByArtificial(QString csv_path, QString shp_path, QString match_result_path)
{
	QFile shp_file(shp_path);
	if (!shp_file.exists())
	{
		QString text = "打开文件失败：" + shp_path + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		return false;
	}
	QgsVectorLayer* vector_layer = new QgsVectorLayer(shp_path, "样本修改图层", "ogr");
	if (vector_layer->featureCount() == 0)
	{
		delete vector_layer;
		QString text = "尚未绘制多边形，用以指定训练样本结果类型。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return false;
	}
	QFile csv_file(csv_path);
	if (!csv_file.open(QFile::ReadOnly))
	{
		delete vector_layer;
		QString text = "打开文件失败：" + csv_path + "。请提取特征后，重试。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return false;
	}
	double geo_transform[6];
	QString projection_ref = "";
	GISHelper::GetCoordinateSystemInfo(match_result_path, geo_transform, projection_ref);
	QTextStream read_text_stream(&csv_file);
	int total_count = read_text_stream.readAll().split("\n").size() - 1;
	double increasing_number = 0;
	read_text_stream.seek(0);
	QString new_csv_text = read_text_stream.readLine(); // 第一行是列头
	QStringList column_name_split = new_csv_text.split(",");
	while (!read_text_stream.atEnd())
	{
		QString old_line_str = read_text_stream.readLine();
		QString new_line_str = old_line_str.left(old_line_str.lastIndexOf(",") + 1);
		QStringList ltwh = old_line_str.split(",")[2].split(";"); // 左上宽高
		int pixel_x = ltwh[1].toInt() + ltwh[2].toInt() / 2;
		int pixel_y = ltwh[0].toInt() + ltwh[3].toInt() / 2;
		QString sample_type = QuerySampleTypeByPixel(pixel_x, pixel_y, geo_transform, vector_layer);
		if ("有效样本" == sample_type)
		{
			new_csv_text += "\n" + old_line_str;
		}
		else if ("正样本" == sample_type)
		{
			new_csv_text += "\n" + new_line_str + "1";
		}
		else if ("负样本" == sample_type)
		{
			new_csv_text += "\n" + new_line_str + "0";
		}
		else if ("无效样本" == sample_type)
		{
			new_csv_text += "\n" + new_line_str + "-9999";
		}
		ProgressValue = increasing_number++ / total_count * 100;
		if (NeedStop) break;
	}
	csv_file.close();
	// 先备份后再修改
	QString backup_path = csv_path.left(csv_path.size() - 4) + "_Backup.csv";
	QFile backup_file(backup_path);
	if (!backup_file.exists() && !QFile::copy(csv_path, backup_path))
	{
		delete vector_layer;
		csv_file.close();
		QString text = "备份文件失败：" + backup_path + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return false;
	}
	if (!csv_file.open(QFile::WriteOnly | QFile::Truncate))
	{
		delete vector_layer;
		QString text = "以写入方式打开文件失败：" + csv_path + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return false;
	}
	QTextStream write_text_stream(&csv_file);
	write_text_stream << new_csv_text;
	csv_file.close();
	delete vector_layer;
	InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), "训练样本修改完成。");
	return true;
}

void OpenMatchResultImage(QString result_image_path,QTableWidget* table_widget)
{
	QString match_result_image = result_image_path.left(result_image_path.size() - 4) + "_Train.tif";
	QFile file(match_result_image);
	if (!file.exists())
	{
		QString text = "未找到匹配结果文件：" + match_result_image + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return;
	}
	GISHelper::OpenImage(match_result_image, LeftMapCanvas);
	QString sar_path = result_image_path.left(result_image_path.size() - 4) + "_雷达_Train.tif";
	QFile temp_file(sar_path);
	if (temp_file.exists())
	{
		GISHelper::OpenImage(sar_path, RightMapCanvas);
	}
	QString shp_path = result_image_path.left(result_image_path.size() - 4) + ".shp";
	QFile shp_file(shp_path);
	if(!shp_file.exists())
	{
		// 复制过去一组空的shp文件
		QString blank_dbf_path = QCoreApplication::applicationDirPath() + "/BaseMap/空白.dbf";
		QString blank_shp_path = QCoreApplication::applicationDirPath() + "/BaseMap/空白.shp";
		QString blank_shx_path = QCoreApplication::applicationDirPath() + "/BaseMap/空白.shx";
		QFile dbf_file(blank_dbf_path);
		QFile shp_file(blank_shp_path);
		QFile shx_file(blank_shx_path);
		if (!dbf_file.exists() || !shp_file.exists() || !shx_file.exists())
		{
			QString text = "未找到文件：\n" + blank_dbf_path + ",或\n" + blank_shp_path + ",或\n" + blank_shx_path + "。";
			Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
			InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		}
		else
		{
			QFile::copy(blank_dbf_path, shp_path.left(shp_path.size() - 4) + ".dbf");
			QFile::copy(blank_shp_path, shp_path);
			QFile::copy(blank_shx_path, shp_path.left(shp_path.size() - 4) + ".shx");
		}
	}
	const char* base_name = "样本修改图层";
	//// 代码设置Polygon的符号，不删保留
	//uto left_fill_symbol = new QgsFillSymbol();
	//left_fill_symbol->setColor(Qt::red);
	//left_fill_symbol->setOpacity(.3);
	//left_layer->setRenderer(new QgsSingleSymbolRenderer(left_fill_symbol));
	// 在左侧中打开
	QgsVectorLayer* left_layer = new QgsVectorLayer(shp_path, base_name, "ogr");
	if (!left_layer->isValid()) return;
	bool result = false;
	left_layer->loadSldStyle(QCoreApplication::applicationDirPath() + "/ConfigurationFiles/样本修改图层符号样式.sld", result);
	//left_layer->setOpacity(0.75); // 暂时不删， 已通过加载文件中的符号样式控制透明度

	QList<QgsMapLayer*> left_layers = LeftMapCanvas->layers();
	left_layers.insert(0, left_layer);
	// 在右侧中打开
	QgsVectorLayer* right_layer = new QgsVectorLayer(shp_path, base_name, "ogr");
	if (!right_layer->isValid()) return;
	right_layer->loadSldStyle(QCoreApplication::applicationDirPath() + "/ConfigurationFiles/样本修改图层符号样式.sld", result);
	//right_layer->setOpacity(0.75); // 暂时不删， 已通过加载文件中的符号样式控制透明度
	QList<QgsMapLayer*> right_layers = RightMapCanvas->layers();
	right_layers.insert(0, right_layer);

#ifndef PlatformIsNeoKylin
	LeftMapCanvas->setLayers(left_layers);
	RightMapCanvas->setLayers(right_layers);
#else
	QgsMapLayerRegistry::instance()->addMapLayer(left_layer);
	QgsMapLayerRegistry::instance()->addMapLayer(right_layer);
	LeftMapCanvas->setLayerSet(QList<QgsMapCanvasLayer>() << QgsMapCanvasLayer(left_layer));
	RightMapCanvas->setLayerSet(QList<QgsMapCanvasLayer>() << QgsMapCanvasLayer(right_layer));
#endif
	SetCloseImageTableWidgetStyle(table_widget);
	SetOpenImageTableWidgetStyle(table_widget);
}

void CloseMatchResultImage(QTableWidget* table_widget)
{
	GISHelper::CloseImage(LeftMapCanvas);
	GISHelper::CloseImage(RightMapCanvas);
	QString path = QCoreApplication::applicationDirPath() + "/BaseMap/WorldImage.tif";
	GISHelper::OpenImage(path, LeftMapCanvas);
	GISHelper::OpenImage(path, RightMapCanvas);
	SetCloseImageTableWidgetStyle(table_widget);
}

void SetCloseImageTableWidgetStyle(QTableWidget* table_widget)
{
	for (int row_index = 0; row_index < table_widget->rowCount(); ++row_index)
	{
		for (int column_index = 0; column_index < table_widget->columnCount(); ++column_index)
		{
			QWidget* widget = table_widget->cellWidget(row_index, column_index);
			if (NULL == widget)
			{
				QTableWidgetItem* item = table_widget->item(row_index, column_index);
				item->setFont(QFont("song", 12));
			}
			else
			{
				widget->setFont(QFont("song", 12));
			}
		}
	}
}

void SetOpenImageTableWidgetStyle(QTableWidget* table_widget)
{
	for (int column_index = 0; column_index < table_widget->columnCount(); ++column_index)
	{
		int row_index = table_widget->currentRow();
		QWidget* widget = table_widget->cellWidget(row_index, column_index);
		if (NULL == widget)
		{
			QTableWidgetItem* item = table_widget->item(row_index, column_index);
			item->setFont(QFont("Microsoft YaHei", 12, 75));
		}
		else
		{
			widget->setFont(QFont("Microsoft YaHei", 12, 75));
		}
	}
}

QString QuerySampleTypeByPixel(int pixel_x, int pixel_y, double geo_transform[6], QgsVectorLayer* vector_layer)
{
	double geography_x, geography_y;
	GISHelper::PixelPointToGeographyPoint(pixel_x, pixel_y, geography_x, geography_y, geo_transform);
	// Todo，暂时放弃，未找到办法，待改进为用点查询图层 注意这里用一个很小的矩形表示一个点
	QgsRectangle rect(geography_x, geography_y, geography_x + 0.1, geography_y + 0.1);
	vector_layer->selectByRect(rect);
	int size = vector_layer->selectedFeatures().size();
	if (size == 0) return "有效样本";
	QgsFeature feature = vector_layer->selectedFeatures().at(0);
	QString sample_type = feature.attribute("SampleType").toString();
	return sample_type;
}

bool UpdateFeatureKeys(ExecuteFlowEnum execute_flow, QString norm_path)
{
	if(execute_flow == TrainFlow)
	{
		QDomElement dom_element = AppDomDocument.documentElement();
		FeatureKeyAll = dom_element.firstChildElement("FeatureKeyAll").text();
		FeatureKeyAll.remove("\t").remove("\n").remove(" ");
		if (SanBuNormTrain != WhichSystem) // 移除针对两个图对匹配的特征，代码有冗余
		{
			FeatureKeyAll.remove(",SiftMatchedPointCount").remove(",SurfMatchedPointCount").remove(",OrbMatchedPointCount");
			FeatureKeyAll.remove(",CrossEntropy").remove(",DeviationIndex").remove(",Distortion").remove(",GradientCovariance");
			FeatureKeyAll.remove(",GradientError").remove(",HogError").remove(",SiftError").remove(",SurfError");
			FeatureKeyAll.remove(",OrbError").remove(",AkazaError").remove(",BriskError").remove(",TwoImagePSNR");
			FeatureKeyAll.remove(",EdgeKeepingIndex").remove(",GradientSSIM").remove(",GraySSIM").remove(",TwoImageMSE");
			FeatureKeyAll.remove(",MaxPeakValue").remove(",SecondPeakValue").remove(",MaxSecondPeakRatio");
		}
		return true;
	}
	QString feature_map_path = norm_path + "_99.FeatureMap.txt";
	QFile file(feature_map_path);
	if (!file.exists())
	{
		QString log_text = "未找到特征名称文件："+ feature_map_path +"。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", log_text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), log_text);
		return false;
	}
	if (!file.open(QFile::ReadOnly | QIODevice::Text))
	{
		QString log_text = "打开特征名称文件：" + feature_map_path + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", log_text);
		return false;
	}
	QTextStream text_stream(&file);
	QStringList read_all = text_stream.readAll().split("\n");
	if (read_all.isEmpty()) return false;
	FeatureKeyAll.clear();
	for (int index = 0; index < read_all.size(); ++index)
	{
		QStringList read_line = read_all.at(index).split("\t");
		if(read_line.size() == 3)
		{
			FeatureKeyAll += read_line.at(1)+",";
		}
	}
	// 预测时，FeatureKeyAll读取自预测准则的输出文件：*_99.FeatureMap.txt，此处把排除的特征(如主峰值、主次比)加回来，提取全部特征
	// 用于第一层适配区筛选，按特征过滤，实际预测时是读取*.csv文件中的特征值并转换为了QMap对象，有冗余特征不影响
	FeatureKeyAll += AppDomDocument.documentElement().firstChildElement("ExcludeKey").text().remove("\t").remove("\n").remove(" ");
	//FeatureKeyAll = FeatureKeyAll.left(FeatureKeyAll.size() - 1);
	if (FeatureKeyAll.right(1) == ",") FeatureKeyAll = FeatureKeyAll.left(FeatureKeyAll.size() - 1);
	file.close();
	return  true;
}
