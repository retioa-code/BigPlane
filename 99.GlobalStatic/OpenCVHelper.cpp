# pragma execution_character_set("utf-8")
#include "OpenCVHelper.h"

#include "ApplicationHelper.h"
#include "FeatureExtract.h"
#include "GISHelper.h"
#include "MatchAlgorithm.h"
#include "TestAndValidate.h"
#include "qmutex.h"

// 参考自：https://www.it1352.com/1702387.html
cv::Mat ShuffleRows(const cv::Mat& image)
{
	std::vector<int> seeds;
	for (int cont = 0; cont < image.rows; cont++)
		seeds.push_back(cont);

	cv::randShuffle(seeds);

	cv::Mat output;
	for (int cont = 0; cont < image.rows; cont++)
		output.push_back(image.row(seeds[cont]));

	return output;
}

cv::Mat MeanFilter(cv::Mat& image, int nBlurSize)
{
	if (image.empty()) return image;
	cv::Mat new_image = image.clone();
	blur(new_image, new_image, cv::Size(nBlurSize, nBlurSize));
	return new_image;
}

// 添加高斯加性噪声，暂不删除
cv::Mat AddGaussAddNoise(cv::Mat& image)
{
	if (image.empty()) return image;
	cv::Mat gauss_noise_mat(image.rows, image.cols, CV_16SC1);
	// 添加高斯斑点噪声，第2个参数是均值，第3个参数是标准差
	randn(gauss_noise_mat, cv::Scalar::all(0), cv::Scalar::all(1));
	cv::Mat noise_image = image.clone();
	noise_image.convertTo(noise_image, CV_16SC1);
	noise_image += gauss_noise_mat;
	gauss_noise_mat = ShuffleRows(gauss_noise_mat);
	noise_image += gauss_noise_mat;
	noise_image.convertTo(noise_image, CV_8UC1);
	return noise_image;
}

// 计算4个角上的n*n的4个小矩形的方差，若有等于0的则认为是黑边
bool ImageHaveBlackArea(cv::Mat& src_mat)
{
	if (src_mat.empty()) return false;
	cv::Scalar stMean, stdDev;
	int nMargin = 4;
	int nRegionWidth = 6, nRegionHeight = 6;
	cv::Mat lefttop_mat = src_mat(cv::Rect(nMargin, nMargin, nRegionWidth, nRegionHeight));
	meanStdDev(lefttop_mat, stMean, stdDev);
	if (stdDev.val[0] == 0)
	{
		return true;
	}
	cv::Mat righttop_mat = src_mat(cv::Rect(src_mat.cols - nMargin - nRegionWidth, nMargin, nRegionWidth, nRegionHeight));
	meanStdDev(righttop_mat, stMean, stdDev);
	if (stdDev.val[0] == 0)
	{
		return true;
	}
	cv::Mat leftbottom_mat = src_mat(cv::Rect(nMargin, src_mat.rows - nMargin - nRegionHeight, nRegionWidth, nRegionHeight));
	meanStdDev(leftbottom_mat, stMean, stdDev);
	if (stdDev.val[0] == 0)
	{
		return true;
	}
	cv::Mat rightbottom_mat = src_mat(cv::Rect(src_mat.cols - nMargin - nRegionWidth, src_mat.rows - nMargin - nRegionHeight, nRegionWidth, nRegionHeight));
	meanStdDev(rightbottom_mat, stMean, stdDev);
	if (stdDev.val[0] == 0)
	{
		return true;
	}
	return false;
}

QMap<QString, QString> CalcImageFeature(cv::Mat& realtime_image,cv::Mat& big_image, cv::Mat& small_image,int real_row,int real_col)
{
	//// 不删，快速写出特征名称字符串：复制提特征的代码，查找并选出包含“insert”的行，用excel分列数据
	//QFile csv_file("C:/Users/hqsc18/Desktop/xxx.cpp");
	//if (!csv_file.open(QFile::ReadOnly | QIODevice::Text))
	//{
	//	return 5;
	//}
	//QTextStream text_stream(&csv_file);
	//uto read_line = text_stream.readAll();
	//uto read_line_split = read_line.split("\n");
	//QString resulttt = "";
	//for (int index = 0; index < read_line_split.size(); ++index)
	//{
	//	if (read_line_split.at(index).contains("insert"))
	//	{
	//		resulttt += read_line_split.at(index) + "\n";
	//	}
	//}
	//exit(0);
	QStringList feature_key_list = FeatureKeyAll.split(",");
	QMap<QString, QString> feature_key_value;
	if(feature_key_list.contains("GrayStdDev11"))
	{
		double mmean = 0, dStdDev = 0;
		CalcMeanStdDev(realtime_image, mmean, dStdDev); // 对梯度强度图，计算标准差
		feature_key_value.insert("GrayStdDev11", QString::number(dStdDev, 10, 6));
	}
	cv::Mat realtime_image_grad = CalcGradientIntensity(realtime_image);
	if (feature_key_list.contains("GradientStdDev11"))
	{
		double mmean = 0, dStdDev = 0;
		CalcMeanStdDev(realtime_image_grad, mmean, dStdDev); // 对梯度强度图，计算标准差
		feature_key_value.insert("GradientStdDev11", QString::number(dStdDev, 10, 6));
	}
	if (feature_key_list.contains("AverageGradient"))
	{
		feature_key_value.insert("AverageGradient", QString::number(mean(realtime_image_grad)[0], 10, 6));
	}
	if (feature_key_list.contains("LaplacianEdgeMean"))
	{
		feature_key_value.insert("LaplacianEdgeMean", QString::number(CalcLaplacianEdgeMean(realtime_image_grad), 10, 6));
	}
	if (feature_key_list.contains("RobertGradientMean"))
	{
		feature_key_value.insert("RobertGradientMean", QString::number(CalcRobertGradientMean(realtime_image), 10, 6));
	}
	double max_std_dev;
	double min_std_dev;
	double min_local_stddev = CalcMaxMinLocalStdDev(realtime_image_grad, max_std_dev, min_std_dev);
	if(feature_key_list.contains("MinLocalStdDev"))
	{
		feature_key_value.insert("MinLocalStdDev", QString::number(min_local_stddev, 10, 6));
	}
	if (!feature_key_list.contains("SignalNoiseRotio"))
	{
	}
	else
	{
		double signal_noise_rotio = 6.66666; // 信噪比默认等于 6.66666 ，大约是平均值
		if (min_std_dev != 0)
		{
			signal_noise_rotio = max_std_dev / min_std_dev;
		}
		feature_key_value.insert("SignalNoiseRotio", QString::number(signal_noise_rotio, 10, 6));
	}
	if (feature_key_list.contains("FriedenEntropy"))
	{
		feature_key_value.insert("FriedenEntropy", QString::number(CalcFriedenEntropy(realtime_image), 10, 6));
	}
	if(feature_key_list.contains("DirectionGradEntropy"))
	{
		feature_key_value.insert("DirectionGradEntropy", QString::number(CalcDirectionGradEntropy(realtime_image), 10, 6));
	}
	if(feature_key_list.contains("GrayEntropy"))
	{
		feature_key_value.insert("GrayEntropy", QString::number(CalcGrayEntropy_2(realtime_image), 10, 6));
	}
	if(feature_key_list.contains("AbsoluteRough"))
	{
		feature_key_value.insert("AbsoluteRough", QString::number(CalcAbsoluteRough(realtime_image), 10, 6));
	}
	if(feature_key_list.contains("EdgeDensity"))
	{
		feature_key_value.insert("EdgeDensity", QString::number(CalcEdgeDensity(realtime_image,realtime_image_grad, ContourLengthThreshold), 10, 6));
	}
	if(feature_key_list.contains("EdgeDensity2"))
	{
		feature_key_value.insert("EdgeDensity2", QString::number(CalcEdgeDensity2(realtime_image, ContourLengthThreshold), 10, 6));
	}
	if(feature_key_list.contains("MinLocalEdgeDensity"))
	{
		feature_key_value.insert("MinLocalEdgeDensity", QString::number(CalcMinLocalEdgeDensity(realtime_image,realtime_image_grad, ContourLengthThreshold), 10, 6));
	}
	if(feature_key_list.contains("IndependPixNum"))
	{
		feature_key_value.insert("IndependPixNum", QString::number(CalcIndependentPixels(realtime_image), 10, 6));
	}
	if(feature_key_list.contains("SingleImagePSNR"))
	{
		feature_key_value.insert("SingleImagePSNR", QString::number(CalcSingleImagePSNR(realtime_image), 10, 6));
	}
	if(feature_key_list.contains("HogEntropy"))
	{
		double hog_entropy = CalcHogEntropy(realtime_image, BlockStride, BlockSize, CellSize, Bins);
		feature_key_value.insert("HogEntropy", QString::number(hog_entropy, 10, 6));
	}
	if (feature_key_list.contains("SingleImageMSE"))
	{
		feature_key_value.insert("SingleImageMSE", QString::number(CalcSingleImageMSE(realtime_image), 10, 6));
	}
	if (feature_key_list.contains("ZeroCrossDensity1"))
	{
		feature_key_value.insert("ZeroCrossDensity1", QString::number(CalcZeroCrossDensity1(realtime_image), 10, 6));
	}
	if (feature_key_list.contains("ZeroCrossDensity2"))
	{
		feature_key_value.insert("ZeroCrossDensity2", QString::number(CalcZeroCrossDensity2(realtime_image), 10, 6));
	}
	if (feature_key_list.contains("CornerPointsDensity"))
	{
		feature_key_value.insert("CornerPointsDensity", QString::number(CalCornerPointsDensity(realtime_image), 10, 6));
	}
	if (feature_key_list.contains("KeyPointIntensityMean") || feature_key_list.contains("KeyPointSizeMean"))
	{
		map<string, float> key_value;
		CalcKeyPoint(realtime_image, key_value);
		if (feature_key_list.contains("KeyPointIntensityMean"))
		{
			feature_key_value.insert("KeyPointIntensityMean", QString::number(key_value["KeyPointIntensityMean"], 10, 6));
		}
		if (feature_key_list.contains("KeyPointSizeMean"))
		{
			feature_key_value.insert("KeyPointSizeMean", QString::number(key_value["KeyPointSizeMean"], 10, 6));
		}
	}
	if (feature_key_list.contains("TextureComplexity"))
	{
		feature_key_value.insert("TextureComplexity", QString::number(CalcTextureComplexity(realtime_image), 10, 6));
	}
	if (feature_key_list.contains("Definition"))
	{
		feature_key_value.insert("Definition", QString::number(CalcDefinition(realtime_image), 10, 6));
	}
	if (feature_key_list.contains("GrayContrast"))
	{
		feature_key_value.insert("GrayContrast", QString::number(CalcGrayContrast(realtime_image), 10, 6));
	}
	if (feature_key_list.contains("OrbKeyPointCount"))
	{
		feature_key_value.insert("OrbKeyPointCount", QString::number(CalcOrbKeyPointCount(realtime_image), 10, 6));
	}
	if (feature_key_list.contains("LbpImageMean") || feature_key_list.contains("LbpGradientMean") || feature_key_list.contains("LbpEdgeMean"))
	{
		map<string, double> key_value;
		CalcLbp(realtime_image, key_value);
		if (feature_key_list.contains("LbpImageMean"))
		{
			feature_key_value.insert("LbpImageMean", QString::number(key_value["LbpImageMean"], 10, 6));
		}
		if (feature_key_list.contains("LbpGradientMean"))
		{
			feature_key_value.insert("LbpGradientMean", QString::number(key_value["LbpGradientMean"], 10, 6));
		}
		if (feature_key_list.contains("LbpEdgeMean"))
		{
			feature_key_value.insert("LbpEdgeMean", QString::number(key_value["LbpEdgeMean"], 10, 6));
		}
	}
	if(feature_key_list.contains("HuMoment0")||feature_key_list.contains("HuMoment1")||feature_key_list.contains("HuMoment2")
		||feature_key_list.contains("HuMoment3")||feature_key_list.contains("HuMoment4")||feature_key_list.contains("HuMoment5")
		||feature_key_list.contains("HuMoment6"))
	{
		map<string, double> key_value;
		CalcHuMoment(realtime_image, key_value);
		if (feature_key_list.contains("HuMoment0"))
		{
			feature_key_value.insert("HuMoment0", QString::number(key_value["HuMoment0"], 10, 6));
		}
		if (feature_key_list.contains("HuMoment1"))
		{
			feature_key_value.insert("HuMoment1", QString::number(key_value["HuMoment1"], 10, 6));
		}
		if (feature_key_list.contains("HuMoment2"))
		{
			feature_key_value.insert("HuMoment2", QString::number(key_value["HuMoment2"], 10, 6));
		}
		if (feature_key_list.contains("HuMoment3"))
		{
			feature_key_value.insert("HuMoment3", QString::number(key_value["HuMoment3"], 10, 6));
		}
		if (feature_key_list.contains("HuMoment4"))
		{
			feature_key_value.insert("HuMoment4", QString::number(key_value["HuMoment4"], 10, 6));
		}
		if (feature_key_list.contains("HuMoment5"))
		{
			feature_key_value.insert("HuMoment5", QString::number(key_value["HuMoment5"], 10, 6));
		}
		if (feature_key_list.contains("HuMoment6"))
		{
			feature_key_value.insert("HuMoment6", QString::number(key_value["HuMoment6"], 10, 6));
		}
	}
	if (feature_key_list.contains("GradientWeightEntropy")||feature_key_list.contains("GradientEnergy")
		||feature_key_list.contains("GradientContrast")||feature_key_list.contains("GradientEvenness")
		||feature_key_list.contains("GradientProportionEntropy"))
	{
		map<string, double> key_value;
		CalcGradientFeature(realtime_image_grad, key_value);
		if (feature_key_list.contains("GradientWeightEntropy"))
		{
			feature_key_value.insert("GradientWeightEntropy", QString::number(key_value["GradientWeightEntropy"], 10, 6));
		}
		if (feature_key_list.contains("GradientEnergy"))
		{
			feature_key_value.insert("GradientEnergy", QString::number(key_value["GradientEnergy"], 10, 6));
		}
		if (feature_key_list.contains("GradientContrast"))
		{
			feature_key_value.insert("GradientContrast", QString::number(key_value["GradientContrast"], 10, 6));
		}
		if (feature_key_list.contains("GradientEvenness"))
		{
			feature_key_value.insert("GradientEvenness", QString::number(key_value["GradientEvenness"], 10, 6));
		}
		if (feature_key_list.contains("GradientProportionEntropy"))
		{
			feature_key_value.insert("GradientProportionEntropy", QString::number(key_value["GradientProportionEntropy"], 10, 6));
		}
	}
	if (feature_key_list.contains("Consistency") || feature_key_list.contains("ThirdOrderMoment") || feature_key_list.contains("FourthOrderMoment"))
	{
		map<string, double> key_value;
		CalcGrayHistogramMoment(realtime_image, key_value);
		if (feature_key_list.contains("Consistency"))
		{
			feature_key_value.insert("Consistency", QString::number(key_value["Consistency"], 10, 6));
		}
		if (feature_key_list.contains("ThirdOrderMoment"))
		{
			feature_key_value.insert("ThirdOrderMoment", QString::number(key_value["ThirdOrderMoment"], 10, 6));
		}
		if (feature_key_list.contains("FourthOrderMoment"))
		{
			feature_key_value.insert("FourthOrderMoment", QString::number(key_value["FourthOrderMoment"], 10, 6));
		}
	}	
	CalcFourierFeature(realtime_image, feature_key_list, feature_key_value);
	CalcGLCMFeature(realtime_image, feature_key_list, feature_key_value);
	CalcGGCMFeature(realtime_image, feature_key_list, feature_key_value);
	cv::Mat big_image_grad = CalcGradientIntensity(big_image);
	cv::Mat small_image_grad = CalcGradientIntensity(small_image);
	CalcCorrelativeSurfaceFeature(realtime_image, big_image_grad, small_image_grad, real_row, real_col, feature_key_list, feature_key_value);
	if (small_image.empty())
	{
		return feature_key_value;
	}
	// 在小图在大图的真实位置处，截取和小图相同大小的子图，提取子图和小图的互信息特征
	cv::Mat sub_big_image = big_image(cv::Rect(real_col, real_row, small_image.cols, small_image.rows));
	if (feature_key_list.contains("CrossEntropy"))
	{
		feature_key_value.insert("CrossEntropy", QString::number(CalcCrossEntropy(sub_big_image, small_image), 10, 6));
	}
	if (feature_key_list.contains("DeviationIndex"))
	{
		feature_key_value.insert("DeviationIndex", QString::number(CalcDeviationIndex(sub_big_image, small_image), 10, 6));
	}
	if (feature_key_list.contains("GradientCovariance"))
	{
		feature_key_value.insert("GradientCovariance", QString::number(CalcGradientCovariance(sub_big_image, small_image), 10, 6));
	}
	if (feature_key_list.contains("EdgeKeepingIndex"))
	{
		feature_key_value.insert("EdgeKeepingIndex", QString::number(CalcEdgeKeepingIndex(sub_big_image, small_image), 10, 6));
	}
	if (feature_key_list.contains("GraySSIM"))
	{
		feature_key_value.insert("GraySSIM", QString::number(CalcGraySSIM(sub_big_image, small_image), 10, 6));
	}
	if (feature_key_list.contains("GradientSSIM"))
	{
		feature_key_value.insert("GradientSSIM", QString::number(CalcGradientSSIM(sub_big_image, small_image), 10, 6));
	}
	if (feature_key_list.contains("TwoImagePSNR"))
	{
		feature_key_value.insert("TwoImagePSNR", QString::number(CalcTwoImagePSNR(sub_big_image, small_image), 10, 6));
	}
	if (feature_key_list.contains("Distortion"))
	{
		feature_key_value.insert("Distortion", QString::number(CalcDistortion(sub_big_image, small_image), 10, 6));
	}
	//// 以下是计算两个图的角点匹配的相关特征。适用于：进行准则分析时输入是两个图组成的图对。
	//if (feature_key_list.contains("SiftMatchedPointCount"))
	//{
	//	feature_key_value.insert("SiftMatchedPointCount", QString::number(CalcSiftMatchedPointCount(big_image, small_image), 10, 6));
	//}
	//if (feature_key_list.contains("SurfMatchedPointCount"))
	//{
	//	feature_key_value.insert("SurfMatchedPointCount", QString::number(CalcSurfMatchedPointCount(big_image, small_image), 10, 6));
	//}
	//if (feature_key_list.contains("OrbMatchedPointCount"))
	//{
	//	feature_key_value.insert("OrbMatchedPointCount", QString::number(CalcOrbMatchedPointCount(big_image, small_image), 10, 6));
	//}
	//if (feature_key_list.contains("GradientError"))
	//{
	//	uto matched_point = MatchAlgorithm::CrossCoGradXY(big_image, small_image);
	//	uto delta_x = matched_point.x - real_col;
	//	uto delta_y = matched_point.y - real_row;
	//	uto error = sqrt(delta_x * delta_x + delta_y * delta_y);
	//	feature_key_value["GradientError"] = QString::number(error, 10, 6);
	//}
	//// 以下是把真实点和匹配点的像素误差，作为特征。 适用于：进行准则分析时输入是两个图组成的图对，且知道真实点位置
	//if (feature_key_list.contains("HogError"))
	//{
	//	cv::Mat small_hog_mat = MatchAlgorithm::CalculateHogFeature(small_image);
	//	cv::Mat big_hog_mat = MatchAlgorithm::CalculateHogFeature(big_image);
	//	uto matched_point = MatchAlgorithm::MatchByHogAndTemplate(big_image, big_hog_mat, small_image, small_hog_mat);
	//	uto delta_x = matched_point.x - real_col;
	//	uto delta_y = matched_point.y - real_row;
	//	uto error = sqrt(delta_x * delta_x + delta_y * delta_y);
	//	feature_key_value["HogError"] = QString::number(error, 10, 6);
	//}
	//if (feature_key_list.contains("SiftError"))
	//{
	//	uto matched_point = MatchAlgorithm::MatchBySift(big_image, small_image);
	//	uto delta_x = matched_point.x - real_col;
	//	uto delta_y = matched_point.y - real_row;
	//	uto error = sqrt(delta_x * delta_x + delta_y * delta_y);
	//	feature_key_value["SiftError"] = QString::number(error, 10, 6);
	//}
	//if (feature_key_list.contains("SurfError"))
	//{
	//	uto matched_point = MatchAlgorithm::MatchBySurf(big_image, small_image);
	//	uto delta_x = matched_point.x - real_col;
	//	uto delta_y = matched_point.y - real_row;
	//	uto error = sqrt(delta_x * delta_x + delta_y * delta_y);
	//	feature_key_value["SurfError"] = QString::number(error, 10, 6);
	//}
	//if (feature_key_list.contains("OrbError"))
	//{
	//	uto matched_point = MatchAlgorithm::MatchByOrb(big_image, small_image);
	//	uto delta_x = matched_point.x - real_col;
	//	uto delta_y = matched_point.y - real_row;
	//	uto error = sqrt(delta_x * delta_x + delta_y * delta_y);
	//	feature_key_value["OrbError"] = QString::number(error, 10, 6);
	//}
	//if (feature_key_list.contains("AkazaError"))
	//{
	//	uto matched_point = MatchAlgorithm::MatchByAkaza(big_image, small_image);
	//	uto delta_x = matched_point.x - real_col;
	//	uto delta_y = matched_point.y - real_row;
	//	uto error = sqrt(delta_x * delta_x + delta_y * delta_y);
	//	feature_key_value["AkazaError"] = QString::number(error, 10, 6);
	//}
	//if (feature_key_list.contains("BriskError"))
	//{
	//	uto matched_point = MatchAlgorithm::MatchByBrisk(big_image, small_image);
	//	uto delta_x = matched_point.x - real_col;
	//	uto delta_y = matched_point.y - real_row;
	//	uto error = sqrt(delta_x * delta_x + delta_y * delta_y);
	//	feature_key_value["BriskError"] = QString::number(error, 10, 6);
	//}
	return feature_key_value;	
}

void CalcFourierFeature(cv::Mat& big_image, QStringList& feature_key, QMap<QString, QString> &feature_key_value)
{
	if (feature_key.contains("FourierBW2") || feature_key.contains("FourierBW4") || feature_key.contains("FourierBW8") || feature_key.contains("FourierBW16")
		|| feature_key.contains("HighFrequencyPercent80") || feature_key.contains("HighFrequencyPercent112") || feature_key.contains("HighFrequencyPercent144")
		|| feature_key.contains("HighFrequencyPercent176") || feature_key.contains("MinLocalHighFrequencyMean") || feature_key.contains("FrequencyDomainEntropy"))
	{
		map<string, float> key_value;
		CalcFourierVeinsPercent(big_image, key_value);
		if (feature_key.contains("FourierBW2"))
		{
			feature_key_value.insert("FourierBW2", QString::number(key_value["FourierBW2"], 10, 6));
		}
		if (feature_key.contains("FourierBW4"))
		{
			feature_key_value.insert("FourierBW4", QString::number(key_value["FourierBW4"], 10, 6));
		}
		if (feature_key.contains("FourierBW8"))
		{
			feature_key_value.insert("FourierBW8", QString::number(key_value["FourierBW8"], 10, 6));
		}
		if (feature_key.contains("FourierBW16"))
		{
			feature_key_value.insert("FourierBW16", QString::number(key_value["FourierBW16"], 10, 6));
		}
		if (feature_key.contains("HighFrequencyPercent80"))
		{
			feature_key_value.insert("HighFrequencyPercent80", QString::number(key_value["HighFrequencyPercent80"], 10, 6));
		}
		if (feature_key.contains("HighFrequencyPercent112"))
		{
			feature_key_value.insert("HighFrequencyPercent112", QString::number(key_value["HighFrequencyPercent112"], 10, 6));
		}
		if (feature_key.contains("HighFrequencyPercent144"))
		{
			feature_key_value.insert("HighFrequencyPercent144", QString::number(key_value["HighFrequencyPercent144"], 10, 6));
		}
		if (feature_key.contains("HighFrequencyPercent176"))
		{
			feature_key_value.insert("HighFrequencyPercent176", QString::number(key_value["HighFrequencyPercent176"], 10, 6));
		}
		if (feature_key.contains("MinLocalHighFrequencyMean"))
		{
			feature_key_value.insert("MinLocalHighFrequencyMean", QString::number(key_value["MinLocalHighFrequencyMean"], 10, 6));
		}
		if (feature_key.contains("FrequencyDomainEntropy"))
		{
			feature_key_value.insert("FrequencyDomainEntropy", QString::number(key_value["FrequencyDomainEntropy"], 10, 6));
		}
	}
}

void CalcGLCMFeature(cv::Mat& big_image, QStringList& feature_key, QMap<QString, QString>& feature_key_value)
{
	if (feature_key.contains("GLCMASM") || feature_key.contains("GLCMEntropy") || feature_key.contains("GLCMContrast") 
		|| feature_key.contains("GLCMHomogeneity") || feature_key.contains("GLCMCorrelation"))
	{
		map<string, double> key_value;
		CalcGLCM(big_image, key_value);
		if (feature_key.contains("GLCMASM"))
		{
			feature_key_value.insert("GLCMASM", QString::number(key_value["GLCMASM"], 10, 6));
		}
		if (feature_key.contains("GLCMEntropy"))
		{
			feature_key_value.insert("GLCMEntropy", QString::number(key_value["GLCMEntropy"], 10, 6));
		}
		if (feature_key.contains("GLCMContrast"))
		{
			feature_key_value.insert("GLCMContrast", QString::number(key_value["GLCMContrast"], 10, 6));
		}
		if (feature_key.contains("GLCMHomogeneity"))
		{
			feature_key_value.insert("GLCMHomogeneity", QString::number(key_value["GLCMHomogeneity"], 10, 6));
		}
		if (feature_key.contains("GLCMCorrelation"))
		{
			feature_key_value.insert("GLCMCorrelation", QString::number(key_value["GLCMCorrelation"], 10, 6));
		}
	}
}

void CalcGGCMFeature(cv::Mat& big_image, QStringList& feature_key, QMap<QString, QString>& feature_key_value)
{
	if (feature_key.contains("GGCMSmallGradientDominance") || feature_key.contains("GGCMBigGradientDominance") || feature_key.contains("GGCMGrayAsymmetry")
		|| feature_key.contains("GGCMGradientAsymmetry") || feature_key.contains("GGCMEnergy") || feature_key.contains("GGCMMixedEntropy")
		|| feature_key.contains("GGCMInertia") || feature_key.contains("GGCMDifferMoment") || feature_key.contains("GGCMGrayMean")
		|| feature_key.contains("GGCMGradientMean") || feature_key.contains("GGCMGrayEntropy") || feature_key.contains("GGCMGradientEntropy")
		|| feature_key.contains("GGCMGrayStddev") || feature_key.contains("GGCMGradientStddev") || feature_key.contains("GGCMCorrelation"))
	{
		map<string, double> key_value;
		CalcGGCM(big_image, key_value);
		if (feature_key.contains("GGCMSmallGradientDominance"))
		{
			feature_key_value.insert("GGCMSmallGradientDominance", QString::number(key_value["SmallGradientDominance"], 10, 6));
		}
		if (feature_key.contains("GGCMBigGradientDominance"))
		{
			feature_key_value.insert("GGCMBigGradientDominance", QString::number(key_value["BigGradientDominance"], 10, 6));
		}
		if (feature_key.contains("GGCMGrayAsymmetry"))
		{
			feature_key_value.insert("GGCMGrayAsymmetry", QString::number(key_value["GrayAsymmetry"], 10, 6));
		}
		if (feature_key.contains("GGCMGradientAsymmetry"))
		{
			feature_key_value.insert("GGCMGradientAsymmetry", QString::number(key_value["GradientAsymmetry"], 10, 6));
		}
		if (feature_key.contains("GGCMEnergy"))
		{
			feature_key_value.insert("GGCMEnergy", QString::number(key_value["Energy"], 10, 6));
		}
		if (feature_key.contains("GGCMMixedEntropy"))
		{
			feature_key_value.insert("GGCMMixedEntropy", QString::number(key_value["MixedEntropy"], 10, 6));
		}
		if (feature_key.contains("GGCMInertia"))
		{
			feature_key_value.insert("GGCMInertia", QString::number(key_value["Inertia"], 10, 6));
		}
		if (feature_key.contains("GGCMDifferMoment"))
		{
			feature_key_value.insert("GGCMDifferMoment", QString::number(key_value["DifferMoment"], 10, 6));
		}
		if (feature_key.contains("GGCMGrayMean"))
		{
			feature_key_value.insert("GGCMGrayMean", QString::number(key_value["GrayMean"], 10, 6));
		}
		if (feature_key.contains("GGCMGradientMean"))
		{
			feature_key_value.insert("GGCMGradientMean", QString::number(key_value["GradientMean"], 10, 6));
		}
		if (feature_key.contains("GGCMGrayEntropy"))
		{
			feature_key_value.insert("GGCMGrayEntropy", QString::number(key_value["GrayEntropy"], 10, 6));
		}
		if (feature_key.contains("GGCMGradientEntropy"))
		{
			feature_key_value.insert("GGCMGradientEntropy", QString::number(key_value["GradientEntropy"], 10, 6));
		}
		if (feature_key.contains("GGCMGrayStddev"))
		{
			feature_key_value.insert("GGCMGrayStddev", QString::number(key_value["GrayStddev"], 10, 6));
		}
		if (feature_key.contains("GGCMGradientStddev"))
		{
			feature_key_value.insert("GGCMGradientStddev", QString::number(key_value["GradientStddev"], 10, 6));
		}
		if (feature_key.contains("GGCMCorrelation"))
		{
			feature_key_value.insert("GGCMCorrelation", QString::number(key_value["Correlation"], 10, 6));
		}
	}
}

void CalcCorrelativeSurfaceFeature(cv::Mat& realtime_image, cv::Mat& big_image_grad, cv::Mat& small_image_grad, int real_row, int real_col, QStringList& feature_key, QMap<QString, QString>& feature_key_value)
{
	if (feature_key.contains("MainPeakValue") || feature_key.contains("SubPeakValue") || feature_key.contains("MainSubPeakRatio")
		|| feature_key.contains("MainSubPeakDifference") || feature_key.contains("MainPeakSharpness") || feature_key.contains("RepeatMode"))
	{
		map<string, float> key_value;
		if(small_image_grad.empty())
		{
			CalcGradCorrelativeSurface(realtime_image, key_value);  // 单个图像匹配时，要对其加噪，所以传过去的是原图，而不是梯度图
		}
		else
		{
			CalcGradCorrelativeSurface(big_image_grad, small_image_grad, real_row, real_col, key_value);
		}
		if (feature_key.contains("MainPeakValue"))
		{
			feature_key_value.insert("MainPeakValue", QString::number(key_value["MainPeakValue"], 10, 6));
		}
		if (feature_key.contains("SubPeakValue"))
		{
			feature_key_value.insert("SubPeakValue", QString::number(key_value["SubPeakValue"], 10, 6));
		}
		if (feature_key.contains("MainSubPeakRatio"))
		{
			feature_key_value.insert("MainSubPeakRatio", QString::number(key_value["MainSubPeakRatio"], 10, 6));
		}
		if (feature_key.contains("MainSubPeakDifference"))
		{
			feature_key_value.insert("MainSubPeakDifference", QString::number(key_value["MainSubPeakDifference"], 10, 6));
		}
		if (feature_key.contains("MainPeakSharpness"))
		{
			feature_key_value.insert("MainPeakSharpness", QString::number(key_value["MainPeakSharpness"], 10, 6));
		}
		if (feature_key.contains("RepeatMode"))
		{
			feature_key_value.insert("RepeatMode", QString::number(key_value["RepeatMode"], 10, 6));
		}
	}
}

void OutputResultImage(QString image_path, QString csv_path, ExecuteFlowEnum execute_flow)
{
	cv::Mat result_image = imread(image_path.toLocal8Bit().data(), cv::IMREAD_GRAYSCALE);
	if (result_image.empty())
	{
		QString log_text = "读取文件失败：" + image_path + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", log_text);
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", log_text);
		return;
	}
	QFile csv_file(csv_path);
	if (!csv_file.open(QFile::ReadOnly | QIODevice::Text))
	{
		QString log_text = "打开文件失败：" + csv_path + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", log_text);
		return;
	}
	QTextStream text_stream(&csv_file);
	//text_stream.setCodec("UTF-8");
	QString read_line = text_stream.readLine();
	if (read_line.isEmpty()) return;
	QStringList read_line_split = read_line.split(",");
	int tlwh_index = 2;
	int predict_probability_index = read_line_split.size() - 1;
	for (int index = 0; index < read_line_split.size(); ++index)
	{
		if (read_line_split.at(index).contains("TopLeftWidthHeight"))
		{
			tlwh_index = index;
		}
		if ("PredictProbability" == read_line_split.at(index))
		{
			predict_probability_index = index;
		}
	}
	// 生成热力图，并给热力图的每个元素赋值
	QStringList rows_cols = read_line.split(",")[1].split("(")[1].replace(")", "").split(";");
	cv::Mat color_map = cv::Mat::zeros(rows_cols[0].toInt(), rows_cols[1].toInt(), CV_8UC1);
	while (!text_stream.atEnd())
	{
		read_line = text_stream.readLine();
		if (read_line.isEmpty()) continue;
		QStringList line_split = read_line.split(",");
		QStringList row_col = line_split[1].split(";");
		double probability = line_split[predict_probability_index].toDouble();
		if (probability < PredictConfidence || PredictConfidence >= 1)
			continue;
		color_map.at<uchar>(row_col[0].toInt(), row_col[1].toInt()) = probability * 255;
	}
	int temp_col = rows_cols[2].toInt() * color_map.cols;
	int temp_row = rows_cols[3].toInt() * color_map.rows;
	resize(color_map, color_map, cv::Size(temp_col, temp_row));
	applyColorMap(color_map, color_map, cv::COLORMAP_JET);
	// 缩放热力图后，调整到原图的宽高
	int top = (result_image.rows - color_map.rows) / 2;
	int bottom = result_image.rows - color_map.rows - top;
	int left = (result_image.cols - color_map.cols) / 2;
	int right = result_image.cols - color_map.cols - left;
	copyMakeBorder(color_map, color_map, top, bottom, left, right, cv::BORDER_REPLICATE);
	OutputHeatMapResult(image_path, color_map, text_stream, tlwh_index, predict_probability_index, execute_flow);
	if (TrainFlow == execute_flow) // 对于训练流程，如果同时有光学和雷达影像，也在雷达影像上输出热力图
	{
		QString sar_path = image_path.left(image_path.size() - 4) + "_雷达.tif";
		QFile sar_file(sar_path);
		if (sar_file.exists())
		{
			OutputHeatMapResult(sar_path, color_map, text_stream, tlwh_index, predict_probability_index, execute_flow);
		}
	}
	cv::Mat color_binary_image;
	if (SiBuUpdate == WhichSystem)
	{
		// 输出彩色二值结果图，红色代表可识别，蓝色代表不可识别
		color_binary_image = cv::Mat::zeros(result_image.rows, result_image.cols, CV_8UC3);
		add(color_binary_image, cv::Scalar(255, 0, 0), color_binary_image); // 蓝色背景大图片
		text_stream.seek(0);
		text_stream.readLine();
		while (!text_stream.atEnd())
		{
			read_line = text_stream.readLine();
			QStringList temp_split = read_line.split(",");
			QStringList tlwh = temp_split[tlwh_index].split(";"); // 左上宽高
			int width = tlwh[2].toInt(); // 默认滑动步长是窗口的一半，下面的代码才有效
			int height = tlwh[3].toInt();
			int temp_left = tlwh[1].toInt();
			int temp_top = tlwh[0].toInt();
			cv::Rect rect(temp_left, temp_top, width, height);
			double value = temp_split[predict_probability_index].toDouble();
			if (value > PredictConfidence)
			{
				static cv::Mat red_mask;
				if (red_mask.empty())
				{
					red_mask = cv::Mat::zeros(height, width, CV_8UC3);
					add(red_mask, cv::Scalar(0, 0, 255), red_mask); // 蓝色掩膜小图片	
				}
				// 和图像的ROI加权求和
				cv::Mat roi_image(color_binary_image(rect));
				addWeighted(roi_image, 0, red_mask, 1, 0, roi_image);
			}
		}
	}
	else if (ShiErSuoNormEstimate == WhichSystem)
	{
		// 暂不删除，输出黑白二值结果图，白色代表可识别，黑色代表不可识别
		color_binary_image = cv::Mat::zeros(result_image.rows, result_image.cols, CV_8UC3);
		add(color_binary_image, cv::Scalar(0, 0, 0), color_binary_image);
		text_stream.seek(0);
		text_stream.readLine();
		cv::Mat mask255;
		while (!text_stream.atEnd())
		{
			read_line = text_stream.readLine();
			QStringList temp_split = read_line.split(",");
			QStringList tlwh = temp_split[tlwh_index].split(";"); // 左上宽高
												  // 默认滑动步长是窗口的一半，下面的代码才有效
			int width = tlwh[2].toInt();
			int height = tlwh[3].toInt();
			int temp_left = tlwh[1].toInt();
			int temp_top = tlwh[0].toInt();
			cv::Rect rect(temp_left, temp_top, width, height);
			double value = temp_split[predict_probability_index].toDouble();
			if (value > PredictConfidence)
			{
				if (mask255.empty())
				{
					mask255 = cv::Mat::ones(rect.size(), CV_8UC3);
					add(mask255, cv::Scalar(255,255,255), mask255);
				}
				// 和图像的ROI加权求和
				cv::Mat roi_image(color_binary_image(rect));
				addWeighted(roi_image, 0, mask255, 1, 0, roi_image);
			}
		}
	}
	QString binary_image_path = image_path.left(image_path.size() - 4);
	if (TrainFlow == execute_flow)
	{
		binary_image_path += "_Train_Binary.tif";
	}
	else if (PredictFlow == execute_flow)
	{
		binary_image_path += "_Predict_Binary.tif";
		OutputRemoveSmallPieces(image_path, color_binary_image);
	}
	cv::imwrite(binary_image_path.toLocal8Bit().data(), color_binary_image);
	AddCoordinateReference(image_path, binary_image_path);
	OutputSuitabilityVector(image_path, &text_stream, tlwh_index, predict_probability_index, execute_flow);
	csv_file.close();
}

void OutputRemoveSmallPieces(QString original_image_path, cv::Mat& color_binary_image)
{
	cv::Mat image_filter = cv::Mat::zeros(color_binary_image.size(), CV_8UC3);
	cv::Mat binary_image;
	cv::cvtColor(color_binary_image, binary_image, cv::COLOR_BGR2GRAY);
	threshold(binary_image, binary_image, 0, 255, cv::THRESH_OTSU);
	binary_image = ConnectiveFilter(binary_image, 10 * SmallImageWidth * SmallImageHeight);
	cv::Mat red_blue_image, heat_image;
	//cv::cvtColor(binary_image, red_blue_image, cv::COLOR_GRAY2BGR);
	cv::resize(binary_image, binary_image, cv::Size(binary_image.cols/100, binary_image.rows / 100));

	cv::Mat original_image = cv::imread(original_image_path.toLocal8Bit().data(), cv::IMREAD_COLOR);
	cv::resize(binary_image, binary_image, cv::Size(original_image.cols, original_image.rows));
	applyColorMap(binary_image, heat_image, cv::COLORMAP_JET);
	cv::addWeighted(original_image, 0.92, heat_image, 0.08, 0, heat_image);
	QString heat_image_path = original_image_path.left(original_image_path.size() - 4) + "_热力图.tif";
	cv::imwrite(heat_image_path.toLocal8Bit().data(), heat_image);
	AddCoordinateReference(original_image_path, heat_image_path);
}

// 使用连通域过滤掉细碎结果
cv::Mat ConnectiveFilter(cv::Mat &image, long pixel_count_threshold)
{
	cv::Mat labels, centroids, stats;
	int nccomps = connectedComponentsWithStats(image, labels, stats, centroids, 4, 4);
	vector<uchar> color_value(nccomps + 1);;
	color_value[0] = 0;
	for (int i = 1; i <= nccomps; i++)
	{
		color_value[i] = stats.at<int>(i, cv::CC_STAT_AREA) < pixel_count_threshold ? 0 : 255;
	}
	// 生成过滤掉小面积连通域的二值图
	cv::Mat image_filter = cv::Mat::zeros(image.size(), CV_8UC1);
	for (int y = 0; y < image_filter.rows; y++)
	{
		for (int x = 0; x < image_filter.cols; x++)
		{
			int label = labels.at<int>(y, x);
			CV_Assert(0 <= label && label <= nccomps);
			image_filter.at<uchar>(y, x) = color_value[label];
		}
	}
	return image_filter;
}

void OutputHeatMapResult(QString original_image_path, cv::Mat& color_map, QTextStream& text_stream, int tlwh_index,int predict_probability_index,ExecuteFlowEnum execute_flow)
{
	cv::Mat original_image = imread(original_image_path.toLocal8Bit().data(), cv::IMREAD_GRAYSCALE);
	// 把单通道的灰度图像，合并为三通道彩色图像
	vector<cv::Mat> three_channels;
	for (int channel_index = 0; channel_index < 3; ++channel_index)
	{
		three_channels.push_back(original_image);
	}
	cv::Mat heat_image;
	merge(three_channels, heat_image);
	// 把热力图和光学原图融合
	if(WhichSystem==SiBuUpdate)
	{
		addWeighted(heat_image, 0.92, color_map, 0.08, 0, heat_image);
	}
	int line_width = AppDomDocument.documentElement().firstChildElement("LineWidth").text().toInt();
	if (line_width > 0)
	{
		// 绘制灰色矩形
		text_stream.seek(0);
		text_stream.readLine();
		while (!text_stream.atEnd())
		{
			QString read_line = text_stream.readLine();
			QStringList tlwh = read_line.split(",")[2].split(";"); // 左上宽高
			cv::Rect rect(tlwh[1].toInt(), tlwh[0].toInt(), tlwh[2].toDouble(), tlwh[3].toDouble());
			rectangle(heat_image, rect, cv::Scalar(128, 128, 128), line_width, 8);
		}
		// 绘制矩形中心点
		text_stream.seek(0);
		text_stream.readLine();
		while (!text_stream.atEnd())
		{
			QString read_line = text_stream.readLine();
			QStringList split = read_line.split(",");
			QStringList tlwh = split[tlwh_index].split(";"); // 左上宽高
			cv::Point point = cv::Point(tlwh[1].toInt() + tlwh[2].toInt() / 2, tlwh[0].toInt() + tlwh[3].toInt() / 2);
			circle(heat_image, point, 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA); // 客户电脑用这个参数
																				 //circle(heat_image, point, 22, cv::Scalar(0, 0, 255), 22, cv::LINE_AA); // 笔记本用这个参数
																				 // 绘制匹配比例或预测概率
			double font_scale = 0.8; // 客户电脑用这个参数
								   //uto font_scale = 3; // 笔记本用这个参数
			QString text = QString::number(split[predict_probability_index].toDouble(), 10, 2);
			putText(heat_image, text.toLocal8Bit().data(), point + cv::Point(0, 10), // 客户电脑用这个参数
				1, font_scale, cv::Scalar(255, 255, 255), 1, 8, false);
			QString position = split.at(1);
			position.replace(";", ", ");
			putText(heat_image, position.toLocal8Bit().data(), point + cv::Point(0, 20), // 客户电脑用这个参数
				1, font_scale, cv::Scalar(255, 255, 255), 1, 8, false);
			//putText(heat_image, text.toLocal8Bit().data(), point + cv::Point(33, 66),
			//        4, font_scale, cv::Scalar(255, 255, 255), 2, 8, false); // 笔记本用这个参数
		}
	}
	QString heat_image_path = original_image_path.left(original_image_path.size() - 4);
	if (TrainFlow == execute_flow)
	{
		heat_image_path += "_Train.tif";
	}
	else if (PredictFlow == execute_flow)
	{
		heat_image_path += "_Predict.tif";
	}
	bool delete_result = QFile::remove(heat_image_path + ".ovr");
	// 这种方式也可以，暂不删除
	//GISHelper::SaveImageByGDAL(heat_image_path, heat_image.data, heat_image.cols, heat_image.rows, 3);
	imwrite(heat_image_path.toLocal8Bit().data(), heat_image);
	AddCoordinateReference(original_image_path, heat_image_path);
}

// 输出*.shp格式的矢量文件，是否适合匹配(0/1)写到了属性表里面了
void OutputSuitabilityVector(QString image_path, QTextStream* text_stream, int tlwh_index, int predict_probability_index, ExecuteFlowEnum execute_flow)
{
#ifndef PlatformIsNeoKylin
	QFileInfo file_info(image_path);
	QString vector_result_path = file_info.absolutePath() + "/SuitabilityVector";
	DeleteFolderByQt4(vector_result_path);
	QDir dir(vector_result_path);
	if(dir.exists())
	{
		QString text = "删除目录失败：" + vector_result_path + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return;
	}
	bool result = dir.mkdir(vector_result_path);
	// 使用GDAL读取栅格文件的坐标系统
	QFile image_file(image_path);
	if (!image_file.exists())
	{
		QString text = "影像文件不存在：" + image_path + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return ;
	}
	GDALDataset* image_dataset = static_cast<GDALDataset*>(GDALOpen(image_path.toLocal8Bit().data(), GA_ReadOnly));
	OGRSpatialReference spatial_reference;
	const char* projection_ref = image_dataset->GetProjectionRef();
	QByteArray ba = QString::fromLocal8Bit(projection_ref).toLatin1(); // must
	char* temp_str = ba.data();
	spatial_reference.importFromWkt(&temp_str);
	double geo_transform[6];
	image_dataset->GetGeoTransform(geo_transform);
	GDALClose(image_dataset);
	GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");
	// 使用GDAL创建带坐标系统的shp矢量文件，并向属性表中插入字段，保存全部结果(适配和不适配)
	QString yesno_name = "YesNo.shp";
	GDALDataset* yesno_dataset = driver->Create((vector_result_path + "/" + yesno_name).toLocal8Bit().data(), 0, 0, 0,
	                                            GDT_Unknown, NULL);
	yesno_dataset->SetProjection(projection_ref);  // 指定shp矢量文件的坐标系统
	OGRLayer * yesno_ogr_layer = yesno_dataset->CreateLayer(yesno_name.toLocal8Bit().data(), &spatial_reference, wkbMultiPolygon);
	OGRFieldDefn field11("suitabilit", OFTString); // 适配性suitability字段
	field11.SetWidth(1);
	yesno_ogr_layer->CreateField(&field11);
	OGRFieldDefn field22("probabilit", OFTReal); // 匹配概率probability字段
	field22.SetPrecision(6);
	yesno_ogr_layer->CreateField(&field22);
	GDALClose(yesno_dataset);
	// 使用GDAL创建带坐标系统的shp矢量文件，并向属性表中插入字段，保存全部结果(适配和不适配)
	QString yes_name = "Yes.shp";
	GDALDataset* yes_dataset = driver->Create((vector_result_path + "/" + yes_name).toLocal8Bit().data(), 0, 0, 0,
	                                          GDT_Unknown, NULL);
	yes_dataset->SetProjection(projection_ref);  // 指定shp矢量文件的坐标系统
	OGRLayer * yes_ogr_layer = yes_dataset->CreateLayer(yes_name.toLocal8Bit().data(), &spatial_reference, wkbMultiPolygon);
	yes_ogr_layer->CreateField(&field11);
	yes_ogr_layer->CreateField(&field22);
	GDALClose(yes_dataset);
	// 使用QGIS向shp矢量文件中插入矩形要素
	text_stream->seek(0);
	text_stream->readLine();
	QgsVectorLayer* yes_layer = new QgsVectorLayer(vector_result_path + "/" + yes_name, yes_name, "ogr");
	QgsVectorLayer* yesno_layer = new QgsVectorLayer(vector_result_path + "/" + yesno_name, yesno_name, "ogr");
	int buffer_size = 3333; // 写入硬盘的分块大小，若对每个多边形都写入硬盘一次，耗时太多
	QgsFeatureList yes_feature_list;
	QgsFeatureList yesno_feature_list;
	while (!text_stream->atEnd())
	{
		QString read_line = text_stream->readLine();
		QStringList temp_split = read_line.split(",");
		QStringList tlwh = temp_split[tlwh_index].split(";"); // 左上宽高
		int pixel_left = tlwh[1].toInt();
		int pixel_top = tlwh[0].toInt();
		int pixel_right = tlwh[1].toInt() + tlwh[2].toInt();
		int pixel_bottom = tlwh[0].toInt() + tlwh[3].toInt();
		double geography_top, geography_left, geography_bottom, geography_right;
		GISHelper::PixelPointToGeographyPoint(pixel_left, pixel_top, geography_left, geography_top, geo_transform);
		GISHelper::PixelPointToGeographyPoint(pixel_right, pixel_bottom, geography_right, geography_bottom, geo_transform);
		QVector<QgsPointXY> point_vector;
		point_vector.append(QgsPointXY(geography_left, geography_top));
		point_vector.append(QgsPointXY(geography_right, geography_top));
		point_vector.append(QgsPointXY(geography_right, geography_bottom));
		point_vector.append(QgsPointXY(geography_left, geography_bottom));
		QgsPolygonXY polygon_xy;
		polygon_xy.append(point_vector);
		QgsFeature polygon_feature;
		polygon_feature.setGeometry(QgsGeometry::fromPolygonXY(polygon_xy));
		double value = temp_split[predict_probability_index].toDouble();
		polygon_feature.setAttributes(QgsAttributes() << QVariant(value > PredictConfidence?"1":"0") << QVariant(QString::number(value,10,6)));
		yesno_feature_list.append(polygon_feature);
		if (value > PredictConfidence)
		{
			yes_feature_list.append(polygon_feature);
		}
		if(yesno_feature_list.size() == buffer_size)
		{
			bool result11 = yesno_layer->dataProvider()->addFeatures(yesno_feature_list);
			bool result22 = yes_layer->dataProvider()->addFeatures(yes_feature_list);
			yes_feature_list.clear();
			yesno_feature_list.clear();
		}
	}
	if(!yesno_feature_list.empty())
	{
		bool result33 = yesno_layer->dataProvider()->addFeatures(yesno_feature_list);
		bool result44 = yes_layer->dataProvider()->addFeatures(yes_feature_list);
	}
	delete yes_layer;
	delete yesno_layer;
	// 转换到WGS1984坐标系
	QString env_variant = "GDAL_DATA=" + QCoreApplication::applicationDirPath() + "/GDALData";
	putenv(env_variant.toLocal8Bit().data()); // 使用ogr2ogr.exe，需要读取GDAL的gcs.csv等文件，把其目录添加到环境变量
	QString ogr2ogr_path = "ogr2ogr"; // 写相对路径且不加后缀名，Windows系统上是调用的EXE路径下的，Kylin系统上是调用的在环境变量指定的目录下寻找的
	QString output_path = vector_result_path + "/WGS1984.shp";
	QString input_path = vector_result_path + "/" + yesno_name;
	const char* other_params = " -f \"ESRI Shapefile\" -t_srs EPSG:4326 -lco ENCODING=UTF-8";
	QString command = "\"" + ogr2ogr_path + "\" \"" + output_path + "\" \"" + input_path + "\"" + other_params;
	// "C:/Project/ogr2ogr.exe" "C:/Project/WGS1984.shp" "C:/Project/QieMoDomClip.tif/SuitabilityVector/YesNo.shp" -f "ESRI Shapefile" -t_srs EPSG:4326 -lco ENCODING=UTF-8
	QProcess process;
	process.start(command);
	QEventLoop event_loop;
	QObject::connect(&process, SIGNAL(finished(int, QProcess::ExitStatus)), &event_loop, SLOT(quit()));
	event_loop.exec(QEventLoop::ExcludeUserInputEvents);
#endif
}

// 用GDAL读取源影像的坐标参数，赋给目标影像
void AddCoordinateReference(QString from_image, QString to_image)
{
	double geo_transform[6];
	QString projection_ref = "";
	GISHelper::GetCoordinateSystemInfo(from_image, geo_transform, projection_ref);
	GDALDataset* to_dataset = static_cast<GDALDataset*>(GDALOpen(to_image.toLocal8Bit().data(), GA_Update));
	if (NULL == to_dataset)
	{
		QString text = "读取文件失败：" + to_image + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return;
	}
	QByteArray ba = projection_ref.toLatin1(); // must
	char* temp_str = ba.data();
	 to_dataset->SetProjection(temp_str);
	to_dataset->SetGeoTransform(geo_transform);
	GDALClose(to_dataset);
}

// 输出十字交叉线结果：第一层蓝色框，第二层绿色半透明掩模，第三层红色十字交叉线。保留不删
//void OutputCrossWire(cv::Mat& input_image, QMap<QPair<int, int>, QString> first_result,
//	QMap<QPair<int, int>, QString> second_result, QMap<QPair<int, int>, QString> third_result)
//{
//	// 输出单张结果图，不删，调试用
//	vector<cv::Mat> three_channels;
//	for (int channel_index = 0; channel_index < 3; ++channel_index)
//	{
//		three_channels.push_back(input_image);
//	}
//	cv::Mat output_image;
//	merge(three_channels, output_image);
//	// 绘制第2层筛选结果
//	uto iterator2 = second_result.begin();
//	while (iterator2 != second_result.end())
//	{
//		QPair<int, int> pair = iterator2.key();
//		Rect rect(pair.first, pair.second, m_nReferenceWidth, m_nReferenceHeight);
//		AddMaskRectangleToImage(output_image, rect, Scalar(0, 255, 0));
//		iterator2++;
//	}
//	// 绘制第3层筛选结果
//	uto iterator3 = third_result.begin();
//	while (iterator3 != third_result.end())
//	{
//		QPair<int, int> pair = iterator3.key();
//		uto step_x = m_nReferenceWidth / 5;
//		uto step_y = m_nReferenceHeight / 5;
//		for (int i = 1; i <= 5; ++i)
//		{
//			Point2f start_point(pair.first, pair.second + step_y * i);
//			Point2f stop_point(pair.first + step_x * i, pair.second);
//			line(output_image, start_point, stop_point, Scalar(0, 0, 255), 3);
//		}
//		for (int i = 1; i < 5; ++i)
//		{
//			Point2f start_point(pair.first + m_nReferenceWidth, pair.second + m_nReferenceHeight - step_y * i);
//			Point2f stop_point(pair.first + m_nReferenceWidth - step_x * i, pair.second + m_nReferenceHeight);
//			line(output_image, start_point, stop_point, Scalar(0, 0, 255), 3);
//		}
//		for (int i = 1; i <= 5; ++i)
//		{
//			Point2f start_point(pair.first, pair.second + m_nReferenceHeight - step_y * i);
//			Point2f stop_point(pair.first + step_x * i, pair.second + m_nReferenceHeight);
//			line(output_image, start_point, stop_point, Scalar(0, 0, 255), 3);
//		}
//		for (int i = 1; i <= 5; ++i)
//		{
//			Point2f start_point(pair.first + m_nReferenceWidth - step_x*i, pair.second);
//			Point2f stop_point(pair.first + m_nReferenceWidth, pair.second + step_y*i);
//			line(output_image, start_point, stop_point, Scalar(0, 0, 255), 3);
//		}
//		iterator3++;
//	}
//	// 绘制第1层筛选结果
//	uto iterator1 = first_result.begin();
//	while (iterator1 != first_result.end())
//	{
//		QPair<int, int> pair = iterator1.key();
//		Rect rect(pair.first, pair.second, m_nReferenceWidth, m_nReferenceHeight);
//		rectangle(output_image, rect, Scalar(255, 0, 0), 5, 8);
//		iterator1++;
//	}
//	QString path = m_strOutputFilePath1.replace("result1", "result");
//	imwrite(path.toLocal8Bit().data(), output_image);
//}

// 把两个图像按比例合成为一个图像，保留不删
//void AddMaskRectangleToImage(cv::Mat& image, cv::Rect roi_rect, cv::Scalar scalar, double alpha)
//{
//	static cv::Mat mask;
//	static cv::Scalar static_scalar(-1, -1, -1);
//	if (mask.cols != roi_rect.width || mask.rows != roi_rect.height || static_scalar != scalar)
//	{
//		mask = cv::Mat::zeros(roi_rect.size(), image.type());
//		static_scalar = scalar;
//		vector<cv::Mat> channels;
//		split(mask, channels);
//		channels[0] += static_scalar[0];
//		channels[1] += static_scalar[1];
//		channels[2] += static_scalar[2];
//		merge(channels, mask);
//	}
//	// 和图像的ROI加权求和
//	cv::Mat roi_image(image(roi_rect));
//	addWeighted(roi_image, 1 - alpha, mask, alpha, 0, roi_image);
//}

QList<QPair<int, int> > GetPositionIndex(int count_x, int count_y, QString csv_path, ExecuteFlowEnum execute_flow , 
	double optical_geo_transform[6], double sar_geo_transform[6],cv::Mat sar_opencv_mat,QList<cv::Rect> &sar_rect_list)
{
	QList<QPair<int, int> > col_row_list;
	if (SiBuUpdate == WhichSystem && UpdateFlow == execute_flow)
	{
		QFile csv_file(csv_path);
		if (!csv_file.open(QFile::ReadOnly))
		{
			QString text = "打开文件失败：" + csv_path + "。";
			Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
			InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
			return col_row_list;
		}
		QTextStream text_stream(&csv_file);
		QStringList field_split = text_stream.readLine().split(",");
		int probability_index = -1, stddev_index = -1, range_index = -1,tlwh_index=-1;
		for (int index = 0; index < field_split.size(); ++index)
		{
			if("PredictProbability"==field_split.at(index))
			{
				probability_index = index;
			}
			else if ("ElevationStdDev" == field_split.at(index))
			{
				stddev_index = index;
			}
			else if ("ElevationRange" == field_split.at(index))
			{
				range_index = index;
			}
			else if(field_split.at(index).contains("TopLeftWidthHeight"))
			{
				tlwh_index = index;

				QString chars = field_split.at(index);
				QStringList ttemp_split = chars.remove("TopLeftWidthHeight(").remove(")").split(";");
			}
		}
		while (!text_stream.atEnd())
		{
			QStringList read_line = text_stream.readLine().split(",");
			QStringList rows_cols = read_line.at(1).split(";");
			int row = rows_cols[0].toInt();
			int col = rows_cols[1].toInt();
			if (row < 0 || row > count_y - 1 || col < 0 || col > count_x - 1) continue;
			// 把经过适配区筛选后，预测结果大于预测置信度的区域选出来
			float predict_value = read_line.at(probability_index).toFloat();
			if (predict_value < PredictConfidence) continue; 
			// 把不在光学影像和雷达影像的地理空间的交集部分的滑窗，过滤掉
			QStringList split = read_line.at(tlwh_index).split(";");
			double geography_x1, geography_y1;
			GISHelper::PixelPointToGeographyPoint(split.at(1).toInt(), split.at(0).toInt(), geography_x1, geography_y1, optical_geo_transform);
			int pixel_x = -1, pixel_y = -1;
			GISHelper::GeographyPointToPixelPoint(geography_x1, geography_y1, pixel_x, pixel_y, sar_geo_transform);
			int left = pixel_x - (BigImageWidth - SmallImageWidth) / 2;
			int top = pixel_y -(BigImageHeight - SmallImageHeight) / 2;
			if (left <0 || top<0 || left + BigImageWidth > sar_opencv_mat.cols || top + BigImageHeight>sar_opencv_mat.rows)
			{
				continue;
			}
			// 按地形高程过滤
			if(stddev_index!=-1 && range_index!=-1)
			{
				float stddev_value = read_line.at(stddev_index).toFloat();
				float range_value = read_line.at(range_index).toFloat();
				if(stddev_value < ElevationStdDevThreshold&& range_value < ElevationRangeThreshold)
				{
					col_row_list.append(QPair<int, int>(col, row));
					sar_rect_list.append(cv::Rect(left, top, BigImageWidth, BigImageHeight));
				}
			}
			else
			{
				col_row_list.append(QPair<int, int>(col, row));
				sar_rect_list.append(cv::Rect(left, top, BigImageWidth, BigImageHeight));
			}
		}
		csv_file.close();
		TotalSlidingCount = col_row_list.size();
		return col_row_list;
	}
	if (SlidingColRowIndex.isEmpty()) // 分析整个图像
	{
		for (int row = 0; row < count_y; ++row)
		{
			for (int col = 0; col < count_x; ++col)
			{
				col_row_list.append(QPair<int, int>(col, row));
			}
		}
	}
	else // 分析配置文件中指定索引位置的滑窗图片
	{
		QStringList split = SlidingColRowIndex.split(".");
		for (int index = 0; index < split.size(); ++index)
		{
			QStringList sub_split = split.at(index).split(",");
			int row = sub_split.at(0).toInt();
			int col = sub_split.at(1).toInt();
			if (row < 0 || row > count_y - 1 || col < 0 || col > count_x - 1) continue;
			col_row_list.append(QPair<int, int>(col, row));
		}
	}
	return col_row_list;
}

// 注意训练和预测的区别
bool ExtractSingleImage(QString optical_image_path, ExecuteFlowEnum execute_flow, QString sar_image_path)
{
	QFile temp_file(optical_image_path);
	if (!temp_file.exists())
	{
		QString text = "未读取到文件：" + optical_image_path + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return false;
	}
	// 初始化OpenCV的随机种子，用于控制在 AddGaussMultipleNoise 函数中调用 randn 的结果，对于每一张大图添加的噪声相同
	cv::setRNGSeed(1024); 
	cv::Mat optical_opencv_mat, sar_opencv_mat;
	GDALDataset* optical_gdal_dataset = NULL;
	GDALDataset* sar_gdal_dataset = NULL;
	GISHelper::ReadImage(optical_image_path, optical_opencv_mat, sar_opencv_mat, optical_gdal_dataset, sar_gdal_dataset, execute_flow, sar_image_path);
	QString csv_path = optical_image_path.left(optical_image_path.size() - 4);
	if (TrainFlow == execute_flow)
	{
		csv_path += "_Train.csv";
	}
	else if (PredictFlow == execute_flow)
	{
		csv_path += "_Predict.csv";
	}
	else if(UpdateFlow == execute_flow)
	{
		csv_path += "_Predict_Filter.csv";
	}
	if (UpdateFlow != execute_flow) QFile::remove(csv_path);
	if (!optical_opencv_mat.empty())
	{
		ProgressText += "。共1块，第1块。";
		bool result = ExtractSingleBlock(optical_image_path, optical_opencv_mat,sar_image_path, sar_opencv_mat, csv_path, execute_flow);
		qDebug() << result;
		if (!result) return false;
	}
	else
	{
		int vertical_block_count = -1;
		long long image_width = optical_gdal_dataset->GetRasterXSize();
		int image_height = optical_gdal_dataset->GetRasterYSize();
		long long total_pixel_count = image_width * image_height;
		long long threshold11 = 100000.0 * 100000.0; // 分3挡
		long long threshold22 = 100000.0 * 20000.0;
		long long threshold33 = 100000.0 * 4000.0;
		if (total_pixel_count > threshold11)
		{
			vertical_block_count = 20;
		}
		else if (total_pixel_count > threshold22)
		{
			vertical_block_count = 10;
		}
		else if (total_pixel_count > threshold33)
		{
			vertical_block_count = 5;
		}
		int count_y = image_height / BigImageHeight;
		int count_y_in_block = count_y / (vertical_block_count - 1); // 整个大影像的每个分块中，y方向上有多少个基准图
		int block_height_pixel = count_y_in_block * BigImageHeight;
		int margin_y = (image_height - BigImageHeight * count_y) / 2; // 按大图的高度的整倍数分块
		for (int block_index = 0; block_index < vertical_block_count; ++block_index)
		{
			int start_y = block_height_pixel * block_index + margin_y;
			if (block_index == vertical_block_count - 1)
			{
				block_height_pixel = (count_y - count_y_in_block * (vertical_block_count - 1)) * BigImageHeight;
			}
			// 此处申请内存时，中括号内不写成两个变量相乘，可申请到较大的内存，测试影像单通道 56515 * 50210 = 28 3761 8150
			uint* optical_image_data = new uint[image_width * block_height_pixel];
			GDALDataType data_type = optical_gdal_dataset->GetRasterBand(1)->GetRasterDataType();
			// GDALDataset 转 OpenCV 的 Mat， 注意：第一个参数是图像高度
			optical_gdal_dataset->GetRasterBand(1)->RasterIO(GF_Read, 0, start_y, image_width, block_height_pixel, optical_image_data, image_width, block_height_pixel, data_type, 0, 0);
			optical_opencv_mat = cv::Mat(block_height_pixel, image_width, CV_8UC1, optical_image_data);
			uint* sar_image_data = new uint[image_width * block_height_pixel];
			sar_gdal_dataset->GetRasterBand(1)->RasterIO(GF_Read, 0, start_y, image_width, block_height_pixel, sar_image_data, image_width, block_height_pixel, data_type, 0, 0);
			sar_opencv_mat = cv::Mat(block_height_pixel, image_width, CV_8UC1, sar_image_data);
			int last_index = ProgressText.lastIndexOf("共");
			if (-1 != last_index)
			{
				ProgressText = ProgressText.left(last_index - 1);
			}
			ProgressText += "，共" + QString::number(vertical_block_count) + "块，第" + QString::number(block_index + 1) + "块。";

			bool result = ExtractSingleBlock(optical_image_path, optical_opencv_mat, sar_image_path, sar_opencv_mat, csv_path, execute_flow,
			                                 start_y);
			if (!result)
			{
				delete[] optical_image_data;
				delete[] sar_image_data;
				break;
			}
			delete[] optical_image_data;
			delete[] sar_image_data;
		}
	}
	GDALClose(optical_gdal_dataset);
	GDALClose(sar_gdal_dataset);
	return true;
}

bool ExtractSingleBlock(QString image_path, cv::Mat optical_opencv_mat, QString sar_image_path, cv::Mat sar_opencv_mat, QString csv_path, ExecuteFlowEnum execute_flow, int start_y)
{
	if (optical_opencv_mat.cols < 400 && optical_opencv_mat.rows < 400)
	{
		QString text = "图像尺寸为：" + QString::number(optical_opencv_mat.cols) + "*" + QString::number(optical_opencv_mat.rows) + "像素。";
		text += "请选择图像宽高均大于400像素的图像后重试。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return false;
	}
	if(SiBuUpdate!= WhichSystem && !sar_opencv_mat.empty())
	{
		if (optical_opencv_mat.cols !=sar_opencv_mat.cols || optical_opencv_mat.rows != sar_opencv_mat.rows)
		{
			QString text = "光学影像和雷达影像的像素宽高，不相等。";
			Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
			InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
			return false;
		}
	}
	QFile output_csv_file(csv_path);
	bool is_exist = output_csv_file.exists();
	if (!output_csv_file.open(QFile::ReadWrite | QFile::Append))
	{
		QString text = "以写入方式打开文件失败：" + csv_path + "。";
		Log4cppPrintf("ERROR", QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__) + "\n", text);
		InfoWarningError::GetInstance()->ShowMessage(QString::number(__LINE__) + "行 @ " + static_cast<QString>(__FILE__), text);
		return false;
	}
	// 区分同源还是异源，重新读取不同的降质参数
	QString str = sar_opencv_mat.empty() ? "ReduceMassParamOfSameSource" : "ReduceMassParamOfDiffSource";
	QDomElement dom_element = AppDomDocument.documentElement().firstChildElement(str);
	StretchDensityThreshold = dom_element.firstChildElement("StretchDensityThreshold").text().toDouble();
	GaussianBlurKernelSize = dom_element.firstChildElement("GaussianBlurKernelSize").text().toDouble();
	GaussianBlurSigma = dom_element.firstChildElement("GaussianBlurSigma").text().toDouble();
	GaussianNoiseStdDev = dom_element.firstChildElement("GaussianNoiseStdDev").text().toDouble();
	RotateAngle = dom_element.firstChildElement("RotateAngle").text().toDouble();
	CompressionRatio = dom_element.firstChildElement("CompressionRatio").text().toDouble();
	double optical_geo_transform[6];
	QString projection_ref="";
	GISHelper::GetCoordinateSystemInfo(image_path, optical_geo_transform, projection_ref);
	double sar_geo_transform[6];
	QString projection_ref11 = "";
	GISHelper::GetCoordinateSystemInfo(sar_image_path, sar_geo_transform, projection_ref11);

	// 宽度一般要大于等于步长，否则可能计算错误。默认按小图宽高的一半滑动，在小图范围内要取到完整的大图
	int delta_x = BigImageWidth - SmallImageWidth;
	int delta_y = BigImageHeight - SmallImageHeight;
	int count_x = (optical_opencv_mat.cols - delta_x - SmallImageWidth) / SmallImageStepX + 1;
	int count_y = (optical_opencv_mat.rows - delta_y - SmallImageHeight) / SmallImageStepY + 1;
	int margin_x = (optical_opencv_mat.cols - SmallImageStepX * (count_x - 1) - SmallImageWidth) / 2;
	int margin_y = (optical_opencv_mat.rows - SmallImageStepY * (count_y - 1) - SmallImageHeight) / 2;
	QString row_col_step = QString::number(count_y) + ";" + QString::number(count_x) + ";" +
		QString::number(SmallImageStepX) + ";" + QString::number(SmallImageStepY);
	QTextStream text_stream(&output_csv_file);
	
	if (!is_exist || UpdateFlow != execute_flow) // 分块读原始文件，滑窗扫描提特征，若文件不存在，则认为是读的某个文件的第一块，插入表头
	{
		text_stream << "ImageName,RowColStep(" + row_col_step + "),TopLeftWidthHeight(" + QString::number(margin_x) + ";" + QString::number(margin_y) + ")," + FeatureKeyAll;
		if (TrainFlow == execute_flow)
		{
			text_stream << ",MatchDistance,MatchProportion\n";
		}
		else if (PredictFlow == execute_flow)
		{
			text_stream << ",MatchDistance,PredictProbability\n";
		}
	}
	QList<cv::Rect> sar_rect_list;
	QList<QPair<int, int> > col_row_list = GetPositionIndex(count_x, count_y, csv_path, execute_flow ,optical_geo_transform, sar_geo_transform,sar_opencv_mat,sar_rect_list);
	if (ParallelExecutionOrNot)
	{
		ExecuteParallelly(optical_opencv_mat, sar_opencv_mat, col_row_list, margin_x, margin_y, start_y, execute_flow, &text_stream, sar_rect_list);
	}
	else
	{
		ExecuteInOrder(optical_opencv_mat, sar_opencv_mat, col_row_list, margin_x, margin_y, start_y, execute_flow, &text_stream, sar_rect_list);
	}
	output_csv_file.close();
	return true;
}

// 注意：和下面的 ExecuteInOrder 代码基本相同，只差一个OpenMP并行指令
void ExecuteParallelly(cv::Mat optical_opencv_mat, cv::Mat sar_opencv_mat, QList<QPair<int, int> > col_row_list, int margin_x, int margin_y, int start_y,
	ExecuteFlowEnum execute_flow, QTextStream* text_stream, QList<cv::Rect> sar_rect_list)
{
	QMutex mutex_write_csv;
	double increasing_number = 0;
	double total_count = col_row_list.size();
#ifdef PlatformIsWindows
	omp_set_num_threads(12);
#pragma omp parallel for
#endif 
	for (int index = 0; index < col_row_list.size(); ++index)
	{
		increasing_number++;
		int col = col_row_list.at(index).first;
		int row = col_row_list.at(index).second;
		QString line_text;
		if (SiBuUpdate == WhichSystem && UpdateFlow == execute_flow)
		{
			line_text = ExtractSingleSlidingWindow(optical_opencv_mat, sar_opencv_mat, col, row, margin_x, margin_y,
				start_y, execute_flow, sar_rect_list.at(index));
		}
		else
		{
			line_text = ExtractSingleSlidingWindow(optical_opencv_mat, sar_opencv_mat, col, row, margin_x, margin_y,
				start_y, execute_flow);
		}
		if (line_text.isEmpty()) continue;
		if (UpdateFlow != execute_flow)
		{
			mutex_write_csv.lock();
			*text_stream << line_text;
			mutex_write_csv.unlock();
		}
		ProgressValue = increasing_number / total_count * 100;
		if (NeedStop) break;
	}
}

// 注意：和上面的 ExecuteParallelly 代码基本相同，只差一个OpenMP并行指令
void ExecuteInOrder(cv::Mat optical_opencv_mat,cv::Mat sar_opencv_mat,QList<QPair<int, int> > col_row_list,int margin_x,int margin_y,int start_y,
	ExecuteFlowEnum execute_flow, QTextStream* text_stream, QList<cv::Rect> sar_rect_list)
{
	QMutex mutex_write_csv;
	double increasing_number = 0;
	double total_count = col_row_list.size();
	for (int index = 0; index < col_row_list.size(); ++index)
	{
		increasing_number++;
		int col = col_row_list.at(index).first;
		int row = col_row_list.at(index).second;
		QString line_text;
		if (SiBuUpdate == WhichSystem && UpdateFlow == execute_flow)
		{
			line_text = ExtractSingleSlidingWindow(optical_opencv_mat, sar_opencv_mat, col, row, margin_x, margin_y,
				start_y, execute_flow, sar_rect_list.at(index));
		}
		else
		{
			line_text = ExtractSingleSlidingWindow(optical_opencv_mat, sar_opencv_mat, col, row, margin_x, margin_y,
				start_y, execute_flow);
		}
		if (line_text.isEmpty()) continue;
		if (UpdateFlow != execute_flow)
		{
			mutex_write_csv.lock();
			*text_stream << line_text;
			mutex_write_csv.unlock();
		}
		ProgressValue = increasing_number / total_count * 100;
		if (NeedStop) break;
	}
}

QString ExtractSingleSlidingWindow(cv::Mat& optical_opencv_mat, cv::Mat& sar_opencv_mat, int col, int row, int margin_x, int margin_y, int start_y, ExecuteFlowEnum execute_flow, cv::Rect sar_rect)
{
	int left_top_x = col * SmallImageStepX + margin_x;
	int left_top_y = row * SmallImageStepY + margin_y;
	cv::Mat small_image_mat(optical_opencv_mat, cv::Rect(left_top_x, left_top_y, SmallImageWidth, SmallImageHeight));
	int small_in_big_x = (BigImageWidth - SmallImageWidth) / 2;
	int small_in_big_y = (BigImageHeight - SmallImageHeight) / 2;
	cv::Rect big_rect = cv::Rect(left_top_x - small_in_big_x, left_top_y - small_in_big_y, BigImageWidth, BigImageHeight);
	cv::Mat big_image_mat = optical_opencv_mat(big_rect);
	if (ImageHaveBlackArea(big_image_mat)) // 黑边区域筛除
	{
		TotalSlidingCount--;
		return "";
	}
	cv::Mat assistant_mat; // 用于辅助分析待分析的 small_image_mat 的图像
	if (!sar_opencv_mat.empty())
	{

		cv::Mat temp_image_mat;
		if (SiBuUpdate == WhichSystem && UpdateFlow == execute_flow)
		{
			temp_image_mat = sar_opencv_mat(sar_rect);
		}
		else
		{
			temp_image_mat= sar_opencv_mat(big_rect);
		}
		if (ImageHaveBlackArea(temp_image_mat)) // 黑边区域筛除
		{
			TotalSlidingCount--;
			return "";
		}
		assistant_mat = temp_image_mat;
	}
	else
	{
		assistant_mat = big_image_mat;
	}
	// 对基准图拉伸对比度
	cv::Mat stretched_image;
	StretchContrast(small_image_mat, stretched_image, StretchDensityThreshold);
	small_image_mat = stretched_image;
	// 降质处理：对实时图降质
	ReduceImageMass(assistant_mat, assistant_mat, GaussianBlurKernelSize, GaussianBlurSigma, GaussianNoiseStdDev);
	QString tlwh = QString::number(left_top_y + start_y) + ";" + QString::number(left_top_x)
		+ ";" + QString::number(SmallImageWidth) + ";" + QString::number(SmallImageHeight);
	QString line_text = ProgressText + "," + QString::number(row) + ";" + QString::number(col) + "," + tlwh;
	cv::Mat temp_image;
	QMap<QString, QString> feature_key_value = CalcImageFeature(small_image_mat, assistant_mat, temp_image, -1, -1);
	QString distance_str = "";
	if (TrainFlow == execute_flow || PredictFlow == execute_flow)
	{
		QString col_row = QString::number(row) + ";" + QString::number(col);
		distance_str = MatchImagePair( small_image_mat,big_image_mat,assistant_mat, col_row, feature_key_value);
	}
	else if (UpdateFlow == execute_flow)
	{
		QString col_row = QString::number(col) + ";" + QString::number(row);
		distance_str = MatchImagePair(small_image_mat, big_image_mat , assistant_mat,  col_row, feature_key_value);
		int match_distance_threshold = AppDomDocument.documentElement().firstChildElement("MatchDistanceThreshold").
		                                              text().toInt();
		static QMutex mutex;
		mutex.lock(); // 对线程加锁，要求每个线程都必须执行到这里
		if (distance_str.toFloat() < static_cast<float>(match_distance_threshold))
		{
			MatchedSlidingCount++;
		}
		else
		{
			QString temp = tlwh + "\n";
			UnmatchedSlideWindow += temp;
		}
		mutex.unlock();
	}
	QStringList feature_name_split = FeatureKeyAll.split(",");
	for (int index = 0; index < feature_name_split.size(); ++index)
	{
		line_text += "," + feature_key_value[feature_name_split.at(index)];
	}
	if (TrainFlow == execute_flow || PredictFlow == execute_flow)
	{
		line_text += "," + distance_str;
	}
	line_text += ",-1111\n";
	return line_text;
}

// small_image 是待分析的图，assistant_mat 用于模拟实时图，直接从雷达影像裁剪或由可见光加噪而来，big_image 仅用于调试时保存到文件夹中查看
QString MatchImagePair(cv::Mat& small_image, cv::Mat& big_image, cv::Mat& assistant_mat, QString col_row, QMap<QString, QString>& feature_key_value)
{
	float max_peak = 0, sub_peak = -1, max_sub_ratio = 0;
	cv::Point matched_point = MatchByAlgorithm(assistant_mat, small_image, max_peak, sub_peak, max_sub_ratio);
	if (IsProductionEnvironment)
	{
		feature_key_value["MainPeakValue"] = QString::number(max_peak, 10, 6);
		feature_key_value["SubPeakValue"] = QString::number(sub_peak, 10, 6);
		feature_key_value["MainSubPeakRatio"] = QString::number(max_sub_ratio, 10, 6);
		feature_key_value["MainSubPeakDifference"] = QString::number(max_peak - sub_peak, 10, 6);
	}
	int actual_point_x = (BigImageWidth - SmallImageWidth) / 2;
	int actual_point_y = (BigImageHeight - SmallImageHeight) / 2;
	int delta_x = matched_point.x - actual_point_x;
	int delta_y = matched_point.y - actual_point_y;
	int match_distance = round(sqrt(delta_x * delta_x + delta_y * delta_y));
	if(SaveIntermediateResultOrNot)
	{
		// 把单通道的灰度图像，合并为三通道彩色图像
		vector<cv::Mat> three_channels;
		for (int channel_index = 0; channel_index < 3; ++channel_index)
		{
			three_channels.push_back(assistant_mat);
		}
		merge(three_channels, assistant_mat);
		QString image_path = QCoreApplication::applicationDirPath() + "/Log/" + col_row + "_" + QString::number(match_distance);
		// image_path += +"_" + feature_key_value["MainPeakValue"] + "_" + feature_key_value["MainSubPeakRatio"];
		// 输出日志显示主次峰比的实际含义，准则训练实际使用的是 (1 - 次峰 / 主峰)
		float ratio = feature_key_value["MainPeakValue"].toFloat() / feature_key_value["SubPeakValue"].toFloat();
		image_path += +"_" + QString::number(feature_key_value["MainPeakValue"].toFloat(),10,2) + "_" + QString::number(ratio, 10, 2);
		imwrite((image_path + "_1.Small待分析的光学.bmp").toLocal8Bit().data(), small_image);
		imwrite((image_path + "_2.Big原图.bmp").toLocal8Bit().data(), big_image);  // 同源匹配降质前的原图，或异源匹配的不加噪声的Sar大图
		cv::Rect actuall_rect(actual_point_y, actual_point_x, SmallImageWidth, SmallImageHeight);
		rectangle(assistant_mat, actuall_rect, cv::Scalar(0, 255, 0), 1, 8);
		cv::Rect matched_rect(matched_point.y, matched_point.x, SmallImageWidth, SmallImageHeight);
		rectangle(assistant_mat, matched_rect, cv::Scalar(0, 0, 255), 1, 8);
		// 文件名示例： 0;2_1_0.692255_1.713646_Big.bmp， 行列号_匹配距离_主峰值_次峰值_Big.bmp
		imwrite((image_path + "_3.Big加噪加框.bmp").toLocal8Bit().data(), assistant_mat); // 用于显示真实位置和匹配位置误差
	}
	return QString::number(match_distance);
}

cv::Point MatchByAlgorithm(cv::Mat& big_image, cv::Mat& rotate_resize_image, float& max_peak, float& sub_peak, float& max_sub_ratio)
{
	cv::Point matched_point;
	if (!IsProductionEnvironment)
	{
		cv::Mat small_hog_mat = MatchAlgorithm::CalculateHogFeature(rotate_resize_image);
		cv::Mat big_hog_mat = MatchAlgorithm::CalculateHogFeature(big_image);
		matched_point = MatchAlgorithm::MatchByHogAndTemplate(big_image, big_hog_mat, rotate_resize_image, small_hog_mat);
	}
	else
	{
		int widthBig = big_image.cols;
		int heightBig = big_image.rows;
		int widthSmall = rotate_resize_image.cols;
		int heightSmall = rotate_resize_image.rows;
		uchar* pImgBig = new uchar[widthBig * heightBig];
		uchar* pImgSmall = new uchar[widthSmall * heightSmall];

		for (int row = 0; row < big_image.rows; ++row)
		{
			for (int col = 0; col < big_image.cols; ++col)
			{
				pImgBig[row * big_image.cols + col] = big_image.at<uchar>(row * big_image.cols + col);
			}
		}
		for (int row = 0; row < rotate_resize_image.rows; ++row)
		{
			for (int col = 0; col < rotate_resize_image.cols; ++col)
			{
				pImgSmall[row * rotate_resize_image.cols + col] = rotate_resize_image.at<uchar>(row * rotate_resize_image.cols + col);
			}
		}
		// 不删，测试用
		//QString str = "";
		//for (int index = 0; index < big_image.total(); ++index)
		//{
		//	if (index % widthBig == 0) 
		//		str += "\r\n";
		//	else 
		//		str += "\t";
		//	str += QString::number(pImgBig[index]);
		//}
		//qDebug() << str;
		float* pCorrelation = new float[widthBig * heightBig];
		float* pBuffer = new float[2000000];
		int matchX = -1, matchY = -1;
		// MatchByGradient(pImgBig, pImgSmall, widthBig, heightBig, widthSmall, heightSmall, matchX, matchY, pCorrelation, pBuffer);
		int cols = widthBig - widthSmall + 1;
		int rows = heightBig - heightSmall + 1;
		cv::Mat correlative_surface = cv::Mat::zeros(rows, cols, CV_32FC1);
		for (int row = 0; row < rows; ++row)
		{
			for (int col = 0; col < cols; ++col)
			{
				correlative_surface.at<float>(row, col) = pCorrelation[row * widthBig + col];
			}
		}
		if (!SlidingColRowIndex.isEmpty()) // 输出可视化相关面
		{
			TestAndValidate::GetInstance()->VisualizeCorrelativeSurface(correlative_surface);
		}
		matched_point = cv::Point(matchX, matchY);
		CalcMaxPeakSubPeak(pCorrelation, widthBig, heightBig, max_peak, sub_peak, max_sub_ratio);
		delete[] pImgBig;
		delete[] pImgSmall;
		delete[] pCorrelation;
		delete[] pBuffer;
	}
	return matched_point;
}

// 不删，这个是接口形式
//void MatchByGradient(uchar* pImgBig, uchar* pImgSmall, int widthBig, int heightBig, int widthSmall, int heightSmall, int matchX, int matchY, float* pCorrelation, float* pBuffer)
//{
//matchX = -9999;
//matchY=-9999;
//}

//// 保留不删，有内存图层
//QgsVectorLayer* ConvertToMemoryLayer(QgsVectorLayer* vector_layer)
//{
//	static QgsVectorLayer* memory_polygon_layer = NULL;
//	if (NULL == memory_polygon_layer)
//	{
//		QString LineLayerPros = QStringLiteral("Polygon?"); //几何类型QString layerProperties = "Point?" 定义了我们创建图层的几何类型，可以是"Point"、"LineString"、"Polygon"、"MultiPoint"、"MultiLineString"、"MultiPolygon"其中之一；
//		memory_polygon_layer = new QgsVectorLayer(LineLayerPros, QString("MemoryPolygonLayer"), QString("memory"));
//	}
//	memory_polygon_layer->startEditing(); // 开始编辑
//	QgsFeatureIds SetFeaIds;
//	QgsFeatureIterator feature_iterator = memory_polygon_layer->getFeatures();
//	QgsFeature f;
//	while (feature_iterator.nextFeature(f))
//	{
//		QVariant fieldID = f.attribute(QStringLiteral("id"));
//		SetFeaIds << f.id();
//	}
//	memory_polygon_layer->dataProvider()->deleteFeatures(SetFeaIds);
//	//先删除，再添加
//	feature_iterator = vector_layer->getFeatures();
//	while (feature_iterator.nextFeature(f))
//	{
//		memory_polygon_layer->dataProvider()->addFeature(f);
//	}
//	memory_polygon_layer->commitChanges(); // 保存
//	return memory_polygon_layer;
//}
