# pragma execution_character_set("utf-8")
#include "GISHelper.h"

#include "ApplicationHelper.h"

void GISHelper::OpenImage(QString image_file_path, QgsMapCanvas* map_canvas)
{
	QFile file(image_file_path);
	if(!file.exists())
	{
		QString text = "未读取到栅格图层：" + image_file_path + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return;
	}
	CloseImage(map_canvas);
	CreateCache(image_file_path, map_canvas);
	QFileInfo mapFileInfo(image_file_path);
	QgsRasterLayer* raster_layer = new QgsRasterLayer(image_file_path, mapFileInfo.baseName());
	Q_ASSERT(raster_layer);
	if (!raster_layer->isValid())
	{
		QString text = "栅格图层无效：" + image_file_path + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return;
	}
	//layers.insert(layers.size() - 2, raster_layer);
#ifndef PlatformIsNeoKylin
	map_canvas->setLayers(QList<QgsMapLayer*>() << raster_layer);
#else
	QgsMapLayerRegistry::instance()->addMapLayer(raster_layer);
	map_canvas->setLayerSet(QList<QgsMapCanvasLayer>() << QgsMapCanvasLayer(raster_layer));
#endif 

	// 设置地图的显示范围 暂不删除， 以后可能用到
	//QgsRectangle extent(-270, -100, 190, 100);
	//if (raster_layer->crs().authid() != "EPSG:4326")
	//{
	//QgsCoordinateReferenceSystem wgs1984_crs;
	//wgs1984_crs.createFromSrid(4326);
	//uto coordinate_transform = QgsCoordinateTransform(raster_layer->crs(), wgs1984_crs, QgsProject::instance());
	//uto extent = coordinate_transform.transform(raster_layer->extent());
	//}

	QgsRectangle extent = raster_layer->extent();
#ifndef PlatformIsNeoKylin
	map_canvas->setExtent(extent.buffered(extent.width() / 22));
#else
	map_canvas->setExtent(extent);
#endif
}

// 删除影像
void GISHelper::CloseImage(QgsMapCanvas* map_canvas)
{
	QList<QgsMapLayer*> layers = map_canvas->layers();
	for (QgsMapLayer* layer : layers)
	{
		layers.removeOne(layer);
		delete layer;
	}
}

void GISHelper::OpenShapeFile(QString shp_file_path, QgsMapCanvas* map_canvas)
{
}

void GISHelper::CreateCache(QString file_path, QgsMapCanvas* map_canvas)
{
	GDALDataset* dataset = static_cast<GDALDataset*>(GDALOpen(file_path.toLocal8Bit().data(), GA_ReadOnly));
	if (dataset == NULL)
	{
		QString text = "GDAL读取影像数据集失败：" + file_path + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return;
	}
	int total_cell_count = dataset->GetRasterXSize() * dataset->GetRasterYSize();
	GDALClose(dataset);
	QFile image(file_path);
	long long image_size = image.size();
	// 对于大于33M的影像，才做切片缓存
	if (image_size <= 33 * 1024 * 1024) return;
	QFile cache_file(file_path + ".ovr");
	if (cache_file.exists()) return;
	// 生成切片缓存金字塔文件，参考自： https://zhuanlan.zhihu.com/p/135359426
	int n = 0;
	while (total_cell_count > 4096 * pow(4, n))
	{
		n++;
	}
	QString command = "";
#ifdef PlatformIsWindows
	QDir dir(QCoreApplication::applicationDirPath());
	command = "\"" + dir.absolutePath() + "/gdaladdo.exe\"";
	command += " -ro --config COMPRESS_OVERVIEW DEFLATE";
	command += " \"" + file_path + "\"";
#else
	command += "gdaladdo -ro --config COMPRESS_OVERVIEW DEFLATE " + file_path;
#endif
	
	for (int i = 1; i < n; ++i)
	{
		command += " " + QString::number(pow(2, i));
	}
	QProcess process;
	process.start(command);
	QEventLoop event_loop;
	QObject::connect(&process, SIGNAL(finished(int, QProcess::ExitStatus)), &event_loop, SLOT(quit()));
	QWidget widget;
	widget.setFixedWidth(388);
	// 不删，放在地图的 map_canvas 中间
	widget.setParent(map_canvas);
	int left = map_canvas->x() + (map_canvas->width() - widget.width()) / 2;
	int top = map_canvas->y() + (map_canvas->height() - widget.height()) / 2;
	// 放在地图的 MmainWindow 中间
	//widget.setParent(MmainWindow);
	//uto left = MmainWindow->x() + (MmainWindow->width() - widget.width()) / 2;
	//uto top = MmainWindow->y() + (MmainWindow->height() - widget.height()) / 2;

	widget.move(left, top);
	widget.show();
	// 菊花转提示
	QMovie* movie = new QMovie(QCoreApplication::applicationDirPath() + "/ResourcesFiles/Waiting.gif");
	movie->setScaledSize(QSize(64, 64));
	movie->start();
	QLabel label_movie;
	label_movie.setMovie(movie);
	label_movie.setVisible(true);
	// 文字提示
	QLabel label_text;
	label_text.setText("正在构建影像金字塔，请稍后……");
	label_text.setFont(QFont("微软雅黑", 14));
	label_text.setStyleSheet("color: rgb(255, 0, 0);");
	QHBoxLayout layout;
	widget.setLayout(&layout);
	layout.addWidget(&label_movie);
	layout.addWidget(&label_text);
	event_loop.exec();
}

void GISHelper::GetCoordinateSystemInfo(QString image_path, double geo_transform[6], QString& projection_ref)
{
	QFile file(image_path);
	if (!file.exists()) return;
	GDALDataset* optical_dataset = static_cast<GDALDataset*>(GDALOpen(image_path.toLocal8Bit().data(), GA_ReadOnly));
	if (optical_dataset == NULL)
	{
		QString text = "GDAL读取影像数据集失败：" + image_path + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return;
	}
	optical_dataset->GetGeoTransform(geo_transform);
	const char* temp_str = optical_dataset->GetProjectionRef();
	projection_ref = QString::fromLocal8Bit(temp_str);
	GDALClose(optical_dataset);
}

void GISHelper::PixelPointToGeographyPoint(int pixel_col, int pixel_row, double& geography_x, double& geography_y, double geo_transform[6])
{
	geography_x = geo_transform[0] + pixel_col * geo_transform[1];
	geography_y = geo_transform[3] + pixel_row * geo_transform[5];
}

void GISHelper::GeographyPointToPixelPoint(double geography_x, double geography_y, int& pixel_col, int& pixel_row, double geo_transform[6])
{
	pixel_col = (geography_x - geo_transform[0]) / geo_transform[1];
	pixel_row = (geography_y - geo_transform[3]) / geo_transform[5];
}

void GISHelper::ReadImage(QString optical_image_path, cv::Mat& optical_image_mat, cv::Mat& sar_image_mat,
                          GDALDataset*& optical_gdal_dataset, GDALDataset*& sar_gdal_dataset, ExecuteFlowEnum execute_flow, QString sar_image_path)
{
	// 用GDAL读取影像及其参数
	optical_gdal_dataset = static_cast<GDALDataset*>(GDALOpen(optical_image_path.toLocal8Bit().data(), GA_ReadOnly));
	if (NULL == optical_gdal_dataset)
	{
		QString text = "读取文件失败：" + optical_image_path + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return;
	}
	if(sar_image_path.isEmpty())
	{
		sar_image_path = optical_image_path.left(optical_image_path.size() - 4) + "_雷达.tif";
	}
	QFile sar_file(sar_image_path);
	if (sar_file.exists())
	{
		sar_gdal_dataset = static_cast<GDALDataset*>(GDALOpen(sar_image_path.toLocal8Bit().data(), GA_ReadOnly));
		if (NULL == sar_gdal_dataset)
		{
			QString text = "读取文件失败：" + sar_image_path + "。";
			Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
			InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
			return;
		}
	}
	double geo_transform[6];
	optical_gdal_dataset->GetGeoTransform(geo_transform);
	QFile optical_image(optical_image_path);
	int image_width = optical_gdal_dataset->GetRasterXSize();
	int image_height = optical_gdal_dataset->GetRasterYSize();
	long long total_cell_count = static_cast<long long>(image_width) * image_height;
	// 对于小于200M 或 像素数小于20000 * 20000的图像，直接读取
	if (optical_image.size() <= 111 * 1024 * 1024 || total_cell_count < 30000 * 30000)
	{
		optical_image_mat = imread(optical_image_path.toLocal8Bit().data(), cv::IMREAD_GRAYSCALE);
		if (optical_image_mat.empty())
		{
			QString log_text = "读取文件失败：" + optical_image_path + "。";
			Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", log_text);
		}
		if(SiBuUpdate != WhichSystem || execute_flow != PredictFlow)
		{
			sar_image_mat = imread(sar_image_path.toLocal8Bit().data(), cv::IMREAD_GRAYSCALE);
		}
	}
}

// 裁剪图像，输出参数中的四至范围是地理坐标， source_dataset，没有在函数内部执行GDALClose(source_dataset);
void GISHelper::ClipImage(QString source_path, double geo_left, double geo_bottom, double geo_right, double geo_top, QString& result_path)
{
	GDALDataset* source_dataset = static_cast<GDALDataset*>(GDALOpen(source_path.toLocal8Bit().data(), GA_ReadOnly));
	if (source_dataset == NULL)
	{
		QString text = "GDAL读取影像数据集失败：" + source_path + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return;
	}

	GDALDataType data_type = source_dataset->GetRasterBand(1)->GetRasterDataType();
	int raster_count = source_dataset->GetRasterCount();
	// 创建目标文件
	GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
	QByteArray target_file = result_path.toLocal8Bit();
	int pixel_left = 0, pixel_bottom = 0, pixel_right = 0, pixel_top = 0;
	double input_geo_transform[6];
	source_dataset->GetGeoTransform(input_geo_transform);
	GeographyPointToPixelPoint(geo_left, geo_bottom,pixel_left,pixel_bottom, input_geo_transform); // Y方向上，像素坐标和地理坐标是相反的
	GeographyPointToPixelPoint(geo_right, geo_top, pixel_right, pixel_top, input_geo_transform);// Y方向上，像素坐标和地理坐标是相反的
	pixel_left = max(0, pixel_left);
	pixel_bottom = min(pixel_bottom, source_dataset->GetRasterYSize());
	pixel_right = min(pixel_right, source_dataset->GetRasterXSize());
	pixel_top = max(0, pixel_top);
	int x_size = pixel_right - pixel_left;
	int y_size = pixel_bottom - pixel_top;
	QString folder_path = result_path.left(result_path.lastIndexOf("/"));
	QDir dir(folder_path);
	if (!dir.exists()) dir.mkdir(folder_path);
	GDALDataset* target_dataset = driver->Create(target_file.data(), x_size, y_size, raster_count, data_type, NULL);
	if (target_dataset == NULL)
	{
		GDALClose(source_dataset);
		QString text = "GDAL创建影像数据集失败：" + result_path + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return;
	}
	double geo_transform[6];
	geo_transform[0] = geo_left;
	geo_transform[1] = input_geo_transform[1];
	geo_transform[2] = input_geo_transform[2];
	geo_transform[3] = geo_top;
	geo_transform[4] = input_geo_transform[4];
	geo_transform[5] = input_geo_transform[5];
	
	target_dataset->SetGeoTransform(geo_transform);
	target_dataset->SetProjection(source_dataset->GetProjectionRef());
	double no_data = source_dataset->GetRasterBand(1)->GetNoDataValue();
	target_dataset->GetRasterBand(1)->SetNoDataValue(no_data); // 设置无效值
	// 分块写入
	int block_size = 2048;
	int horizontal_count = static_cast<int>(ceil(x_size / static_cast<float>(block_size)));
	const int vertical_count = static_cast<int>(ceil(y_size / static_cast<float>(block_size)));
	for (int index = 1; index < raster_count + 1; ++index)
		for (int i = 0; i < horizontal_count; i++)
		{
			for (int j = 0; j < vertical_count; j++)
			{
				int x_offset = block_size * i;
				int y_offset = block_size * j;
				int x_sub_size = block_size;
				int y_sub_size = block_size;
				if (block_size * (i + 1) > x_size) x_sub_size = x_size - block_size * i;
				if (block_size * (j + 1) > y_size) y_sub_size = y_size - block_size * j;
				float* source_buffer = new float[x_sub_size * y_sub_size];
				// 读
				int xx_offset = x_offset + pixel_left;
				int yy_offset = y_offset + pixel_top;
				source_dataset->GetRasterBand(index)->RasterIO(GF_Read, xx_offset, yy_offset, x_sub_size, y_sub_size, source_buffer, x_sub_size, y_sub_size, GDT_Float32, 0, 0);
				// 写
				target_dataset->GetRasterBand(index)->RasterIO(GF_Write, x_offset, y_offset, x_sub_size, y_sub_size, source_buffer, x_sub_size, y_sub_size, GDT_Float32, 0, 0);
				delete[] source_buffer;
			}
		}
	GDALClose(source_dataset);
	GDALClose(target_dataset);
}

/***
* 遥感影像重采样，参考自： https://blog.csdn.net/sunj92/article/details/51787856
* @param pszSrcFile        输入文件的路径
* @param pszOutFile        写入的结果图像的路径
* @param eResample         采样模式，有五种，具体参见GDALResampleAlg定义，默认为双线性内插
							GRA_NearestNeighbour=0      最近邻法，算法简单并能保持原光谱信息不变；缺点是几何精度差，灰度不连续，边缘会出现锯齿状
							GRA_Bilinear=1              双线性法，计算简单，图像灰度具有连续性且采样精度比较精确；缺点是会丧失细节；
							GRA_Cubic=2                 三次卷积法，计算量大，图像灰度具有连续性且采样精度高；
							GRA_CubicSpline=3           三次样条法，灰度连续性和采样精度最佳；
							GRA_Lanczos=4               分块兰索斯法，由匈牙利数学家、物理学家兰索斯法创立，实验发现效果和双线性接近；
* @param fResX             X转换采样比，默认大小为1.0，大于1图像变大，小于1表示图像缩小。数值上等于采样后图像的宽度和采样前图像宽度的比
* @param fResY             Y转换采样比，默认大小为1.0，大于1图像变大，小于1表示图像缩小。数值上等于采样后图像的高度和采样前图像高度的比
* @retrieve     0   成功
* @retrieve     -1  打开源文件失败
* @retrieve     -2  创建新文件失败
* @retrieve     -3  处理过程中出错
*/
int GISHelper::ResampleImage(const char* pszSrcFile, const char* pszOutFile, float fResX, float fResY, GDALResampleAlg eResample)
{
	GDALAllRegister();
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	GDALDataset *pDSrc = (GDALDataset *)GDALOpen(pszSrcFile, GA_ReadOnly);
	if (pDSrc == NULL)
	{
		return -1;
	}

	GDALDriver *pDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
	if (pDriver == NULL)
	{
		GDALClose((GDALDatasetH)pDSrc);
		return -2;
	}
	int width = pDSrc->GetRasterXSize();
	int height = pDSrc->GetRasterYSize();
	int nBandCount = pDSrc->GetRasterCount();
	GDALDataType dataType = pDSrc->GetRasterBand(1)->GetRasterDataType();

	char *pszSrcWKT = NULL;
	pszSrcWKT = const_cast<char *>(pDSrc->GetProjectionRef());

	double dGeoTrans[6] = { 0 };
	int nNewWidth = width, nNewHeight = height;
	pDSrc->GetGeoTransform(dGeoTrans);

	bool bNoGeoRef = false;
	double dOldGeoTrans0 = dGeoTrans[0];
	//如果没有投影，人为设置一个  
	if (strlen(pszSrcWKT) <= 0)
	{
		//OGRSpatialReference oSRS;
		//oSRS.SetUTM(50,true);	//北半球  东经120度
		//oSRS.SetWellKnownGeogCS("WGS84");
		//oSRS.exportToWkt(&pszSrcWKT);
		//pDSrc->SetProjection(pszSrcWKT);
		//////////////////////////////////////////////////////////////////////////
		dGeoTrans[0] = 1.0;
		pDSrc->SetGeoTransform(dGeoTrans);
		//////////////////////////////////////////////////////////////////////////
		bNoGeoRef = true;
	}

	//adfGeoTransform[0] /* top left x */
	//adfGeoTransform[1] /* w-e pixel resolution */
	//adfGeoTransform[2] /* rotation, 0 if image is "north up" */
	//adfGeoTransform[3] /* top left y */
	//adfGeoTransform[4] /* rotation, 0 if image is "north up" */
	//adfGeoTransform[5] /* n-s pixel resolution */

	dGeoTrans[1] = dGeoTrans[1] / fResX;
	dGeoTrans[5] = dGeoTrans[5] / fResY;
	nNewWidth = static_cast<int>(nNewWidth*fResX + 0.5);
	nNewHeight = static_cast<int>(nNewHeight*fResY + 0.5);

	//创建结果数据集
	GDALDataset *pDDst = pDriver->Create(pszOutFile, nNewWidth, nNewHeight, nBandCount, dataType, NULL);
	if (pDDst == NULL)
	{
		GDALClose((GDALDatasetH)pDSrc);
		return -2;
	}

	pDDst->SetProjection(pszSrcWKT);
	pDDst->SetGeoTransform(dGeoTrans);

	void *hTransformArg = NULL;
	hTransformArg = GDALCreateGenImgProjTransformer2((GDALDatasetH)pDSrc, (GDALDatasetH)pDDst, NULL); //GDALCreateGenImgProjTransformer((GDALDatasetH) pDSrc,pszSrcWKT,(GDALDatasetH) pDDst,pszSrcWKT,FALSE,0.0,1);

	if (hTransformArg == NULL)
	{
		GDALClose((GDALDatasetH)pDSrc);
		GDALClose((GDALDatasetH)pDDst);
		return -3;
	}

	GDALWarpOptions *psWo = GDALCreateWarpOptions();

	psWo->papszWarpOptions = CSLDuplicate(NULL);
	psWo->eWorkingDataType = dataType;
	psWo->eResampleAlg = eResample;
	psWo->hSrcDS = (GDALDatasetH)pDSrc;
	psWo->hDstDS = (GDALDatasetH)pDDst;
	psWo->pfnTransformer = GDALGenImgProjTransform;
	psWo->pTransformerArg = hTransformArg;
	psWo->nBandCount = nBandCount;
	psWo->panSrcBands = (int *)CPLMalloc(nBandCount * sizeof(int));
	psWo->panDstBands = (int *)CPLMalloc(nBandCount * sizeof(int));
	for (int i = 0; i < nBandCount; i++)
	{
		psWo->panSrcBands[i] = i + 1;
		psWo->panDstBands[i] = i + 1;
	}

	GDALWarpOperation oWo;
	if (oWo.Initialize(psWo) != CE_None)
	{
		GDALClose((GDALDatasetH)pDSrc);
		GDALClose((GDALDatasetH)pDDst);
		return -3;
	}

	oWo.ChunkAndWarpImage(0, 0, nNewWidth, nNewHeight);

	GDALDestroyGenImgProjTransformer(hTransformArg);
	GDALDestroyWarpOptions(psWo);
	if (bNoGeoRef)
	{
		dGeoTrans[0] = dOldGeoTrans0;
		pDDst->SetGeoTransform(dGeoTrans);
		//pDDst->SetProjection("");
	}
	GDALFlushCache(pDDst);
	GDALClose((GDALDatasetH)pDSrc);
	GDALClose((GDALDatasetH)pDDst);
	return 0;
}

/**
*brief 保存影像到磁盘，建议不放在系统磁盘目录
*/
void GISHelper::SaveImageByGDAL(const QString &file_name, void* image, int width, int height, int band)
{
	GDALDriver* pDriver = static_cast<GDALDriver*>(GDALGetDriverByName("GTiff"));
	int nLineCount = width * band;
	unsigned char* ptr1 = static_cast<unsigned char*>(image);
	// 注意下面这几个变量要显式实例化
	QByteArray byte_array = file_name.toLocal8Bit();
	char* psz_name = byte_array.data();
	GDALDataset* pDst = pDriver->Create(psz_name, width, height, band, GDT_Byte, NULL);
	for (int i = 1; i <= band; i++)
	{
		GDALRasterBand* pBand = pDst->GetRasterBand(band - i + 1);
		pBand->RasterIO(GF_Write,
			0,
			0,
			width,
			height,
			ptr1 + i - 1,
			width,
			height,
			GDT_Byte,
			band,
			nLineCount);
	}
	GDALClose(pDst);
}
