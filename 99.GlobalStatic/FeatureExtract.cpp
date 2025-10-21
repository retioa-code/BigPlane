#include "FeatureExtract.h"

// #include <opencv2/xfeatures2d/nonfree.hpp>

#include "HeaderFiles.h"
#include "TestAndValidate.h"

void CalcMeanStdDev(cv::Mat image, double& mmean, double& std_dev)
{
	if (image.empty()) return;
	cv::Mat mean_mat, std_dev_mat;
	meanStdDev(image, mean_mat, std_dev_mat);
	mmean = mean_mat.at<double>(0, 0);
	std_dev = std_dev_mat.at<double>(0, 0);
}

// 计算单幅图像的最小局部标准差
double CalcMaxMinLocalStdDev(cv::Mat& image, double& max_std_dev, double& min_std_dev)
{
	if (image.empty()) return 0;
	// 默认切分为4*4的网格
	int col_count = 4;
	int row_count = 4;
	int col_step = image.cols / col_count;
	int row_step = image.rows / row_count;
	max_std_dev = -DBL_MAX;
	min_std_dev = DBL_MAX;
	for (int col_index = 0; col_index < col_count; ++col_index)
	{
		for (int row_index = 0; row_index < row_count; ++row_index)
		{
			int left = col_index * col_step;
			int top = row_index * row_step;
			cv::Mat roi_mat = image(cv::Rect(left, top, col_step, row_step));
			cv::Mat mean_mat, std_dev_mat;
			meanStdDev(roi_mat, mean_mat, std_dev_mat);
			max_std_dev = max(max_std_dev, std_dev_mat.at<double>(0));
			min_std_dev = min(min_std_dev, std_dev_mat.at<double>(0));
		}
	}
	return min_std_dev;
}

// 计算单幅图像的Hog熵
double CalcHogEntropy(cv::Mat& image, cv::Size block_stride, cv::Size block_size, cv::Size cell_size, int bins)
{
	// 缩放原图，到 BlockStride 的整数倍
	int new_image_cols = round(image.cols * 1.0 / block_stride.width) * block_stride.width;
	int new_image_rows = round(image.rows * 1.0 / block_stride.height) * block_stride.height;
	cv::Size new_image_size(new_image_cols, new_image_rows);
	cv::Mat new_image;
	resize(image, new_image, new_image_size);
	cv::HOGDescriptor hog_descriptor(new_image_size, block_size, block_stride, cell_size, bins);
	vector<float> hog_feature;
	hog_descriptor.compute(new_image, hog_feature);
	// 在每个 Block 上计算熵值
	int dimension_of_block = (block_size.width / cell_size.width) * (block_size.height / cell_size.height) * bins; // 每个 block 上的特征个数
	int block_count = hog_feature.size() / dimension_of_block;
	cv::Mat hog_mat(block_count, dimension_of_block, CV_32FC1);
	for (int index = 0; index < hog_mat.total(); ++index)
	{
		hog_mat.at<float>(index) = hog_feature[index]; // hog_feature表示每个特征(bin个方向)，在该block上的个数比例
	}
	// 在 Block 上归一化，默认使用的是L2范数，每个 Block 上所有特征的平方和等于1，每个元素的平方表示所占的概率，元素值可能为0
	hog_mat = hog_mat.mul(hog_mat); // 加上这句才是做了归一化，毎36个像素值表示的一个block的和为1
	// 异常值处理，log(0)没有意义。计算对数前，如果元素等于0 ，则替换为1，log(1)等于0，再与原mat中的元素相乘，结果还等于0
	for (int index = 0; index < hog_mat.total(); ++index)
	{
		if (hog_mat.at<float>(index) == 0)
		{
			hog_mat.at<float>(index) = 1;
		}
	}
	cv::Mat log_mat;
	log(hog_mat, log_mat); // OpenCV里的这个log函数是自然指数，是以e为底的，信息熵的定义是以2为底的，二者用换底公式换算就相差了一个常数
	hog_mat = hog_mat.mul(log_mat);
	// 不乘以 feature_number 表示对每个像素的梯度的熵取均值，乘以 dimension_of_block 表示对每个block的熵取均值。和定义不完全一致，是为了好量化
	double hog_entropy = -mean(hog_mat)[0] * dimension_of_block;
	return hog_entropy;
}

// 计算单幅图像的边缘密度，参考自： https://blog.csdn.net/qq_23880193/article/details/49257035
double CalcEdgeDensity(cv::Mat& image,cv::Mat gradient_mat, int contour_length_threshold)
{
	if (image.empty()) return 0;
	cv::Mat otsu_image;
	// 求梯度，使用otsu对梯度图，计算阈值
	gradient_mat.convertTo(gradient_mat, CV_8UC1);
	// 自适应调整 Canny 函数的大小阈值，使用最大类间方差Otsu(大津分割算法)的自适应阈值，参考自网络
	double big_threshold = threshold(gradient_mat, otsu_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	// 为了得到二值图像，对灰度图进行边缘检测
	cv::Mat canny_image; // 阈值设为正负百分之三十：0.7和1.3
	Canny(image, canny_image, big_threshold * 0.7, big_threshold * 1.3, 3);
	// 在得到的二值图像中寻找轮廓
	vector<vector<cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;
	findContours(canny_image, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	// 计算轮廓的长度
	double totoal_length = 0;
	cv::Mat temp_mat = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
	for (int i = 0; i < contours.size(); ++i)
	{
		vector<cv::Point> contour = contours.at(i);
		int single_length = contour.size();
		if (single_length < contour_length_threshold)
		{
			continue;
		}
		totoal_length += single_length;
		for (int i = 0; i < contour.size(); ++i)
		{
			temp_mat.at<uchar>(contour.at(i).y, contour.at(i).x) = 255;
		}
	}
	double edge_density = totoal_length / image.total();
	return edge_density;
}

double CalcEdgeDensity2(cv::Mat& image, int contour_length_threshold)
{
	if (image.empty()) return 0;
	cv::Mat gauss_mat;
	GaussianBlur(image, gauss_mat, cv::Size(5,5), 0, 0, cv::BORDER_REFLECT101);
	cv::Mat kernel = (cv::Mat_<char>(5, 5) << -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 24, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
	cv::Mat convolution_image;
	filter2D(gauss_mat, convolution_image, CV_32FC1, kernel);
	convolution_image.convertTo(convolution_image, CV_8UC1);
	// 自适应调整 Canny 函数的大小阈值，使用最大类间方差Otsu(大津分割算法)的自适应阈值，参考自网络
	cv::Mat otsu_image;
	double big_threshold = cv::threshold(convolution_image, otsu_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	cv::Mat canny_image; // 阈值设为正负百分之三十：0.7和1.3
	Canny(image, canny_image, big_threshold * 0.7, big_threshold * 1.3, 3);
	// 在得到的二值图像中寻找轮廓
	vector<vector<cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;
	findContours(canny_image, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	// 计算轮廓的长度
	double totoal_length = 0;
	cv::Mat temp_mat = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
	for (int i = 0; i < contours.size(); ++i)
	{
		vector<cv::Point> contour = contours.at(i);
		int single_length = contour.size();
		if (single_length < contour_length_threshold)
		{
			continue;
		}
		totoal_length += single_length;
		for (int i = 0; i < contour.size(); ++i)
		{
			temp_mat.at<uchar>(contour.at(i).y, contour.at(i).x) = 255;
		}
	}
	double edge_density = totoal_length / image.total();
	return edge_density;
}

double CalcMinLocalEdgeDensity(cv::Mat& image, cv::Mat& gradient_mat, int contour_length_threshold)
{
	if (image.empty()) return 0;
	// 默认切分为4*4的网格
	int col_count = 4;
	int row_count = 4;
	int col_step = image.cols / col_count;
	int row_step = image.rows / row_count;
	double min_local_edge_density = DBL_MAX;
	for (int col_index = 0; col_index < col_count; ++col_index)
	{
		for (int row_index = 0; row_index < row_count; ++row_index)
		{
			int left = col_index * col_step;
			int top = row_index * row_step;
			cv::Mat roi_mat = image(cv::Rect(left, top, col_step, row_step));
			cv::Mat roi_gradient_mat = gradient_mat(cv::Rect(left, top, col_step, row_step));
			double local_edge_density = CalcEdgeDensity(roi_mat, roi_gradient_mat, contour_length_threshold / 4);
			min_local_edge_density = min(min_local_edge_density, local_edge_density);
		}
	}
	return min_local_edge_density;
}

// 保留不删，获取LOG滤波核，参考自：https://www.bilibili.com/video/av455967910/
cv::Mat GetLOGKernel(int ksize, double sigma)
{
	cv::Mat kernel(ksize, ksize, CV_32FC1);
	int center = ksize / 2;
	double hg = 0;
	for (int i = 0; i < ksize; i++)
	{
		float* p = kernel.ptr<float>(i);
		for (int j = 0; j < ksize; j++)
		{
			int x = j - center;
			int y = i - center;
			double h = std::exp(-(x * x + y * y) / (2.0 * sigma * sigma));
			hg += h;
			p[j] = (x * x + y * y - 2 * sigma * sigma) * h / (sigma * sigma * sigma * sigma);
		}
	}
	kernel /= hg;
	return kernel;
}

// 计算单幅图像及其高斯模糊图像的均方误差
double CalcSingleImageMSE(cv::Mat& image)
{
	cv::Mat gauss_mat, laplacian_mat;
	GaussianBlur(image, gauss_mat, cv::Size(3, 3), 0, 0, cv::BORDER_REFLECT101);

	double result = CalcTwoImageMSE(image, gauss_mat);
	return result;
}

// 计算两幅图像的均方误差，参考自： https://blog.csdn.net/TracelessLe/article/details/110203048
double CalcTwoImageMSE(cv::Mat& source_image, cv::Mat& target_image)
{
	if (source_image.empty() || target_image.empty()
		|| source_image.cols != target_image.cols
		|| source_image.rows != target_image.rows)
	{
		return 0;
	}
	cv::Mat different;
	absdiff(source_image, target_image, different);
	different.convertTo(different, CV_32F);
	different = different.mul(different);
	double result = mean(different)[0];
	return result;
}

// 计算两幅图像的峰值信噪比(Peak Signal Noise Ratio)
// 参考自： https://blog.csdn.net/laoxuan2011/article/details/51519062/
double CalcTwoImagePSNR(cv::Mat& source_image, cv::Mat& target_image)
{
	if (source_image.empty() || target_image.empty()
		|| source_image.cols != target_image.cols
		|| source_image.rows != target_image.rows)
	{
		return 0;
	}
	double mean_square_error = CalcTwoImageMSE(source_image, target_image);
	double result = 10.0 * log10(255 * 255 / mean_square_error);
	return result;
}

// 计算单幅图像的峰值信噪比(Peak Signal Noise Ratio)
// 用高斯―拉普拉斯算子作为滤波器，得到噪声图像，将噪声图像和原图像进行方差比值
double CalcSingleImagePSNR(cv::Mat& image)
{
	if (image.empty()) return 0;
	cv::Mat gauss_mat, laplacian_mat;
	GaussianBlur(image, gauss_mat, cv::Size(3, 3), 0, 0, cv::BORDER_REFLECT101);
	Laplacian(gauss_mat, laplacian_mat, CV_8UC1, 1, 1.0, 0.0, cv::BORDER_REFLECT101);
	//// 暂时不删测试用，结果和上面的接近，虽然只做了一步卷积，但是计算速度慢
	//cv::Mat log_mat;
	//cv::Mat kernel = GetLOGKernel(kernel_size, sigma);
	//filter2D(image, log_mat, CV_8UC1, kernel, Point(-1, -1), 0.0, BORDER_REFLECT101);
	double result = CalcTwoImagePSNR(image, laplacian_mat);
	return result;
}

// 计算单幅图像的绝对值粗糙度
double CalcAbsoluteRough(cv::Mat& image)
{
	if (image.empty()) return 0;
	cv::Mat grad_x_mat, grad_y_mat;
	// 注意是当前像素值，横向上减去右边一个像素值，纵向上减去下边一个像素值
	cv::Mat kernel_x = (cv::Mat_<char>(1, 3) << 0, 1, -1);
	filter2D(image, grad_x_mat, CV_32FC1, kernel_x);
	cv::Mat kernel_y = (cv::Mat_<char>(3, 1) << 0, 1, -1);
	filter2D(image, grad_y_mat, CV_32FC1, kernel_y);
	// 对梯度强度图，取绝对值
	grad_x_mat = abs(grad_x_mat);
	grad_y_mat = abs(grad_y_mat);
	double result = (mean(grad_x_mat)[0] + mean(grad_y_mat)[0]) / 2;
	return result;
}

// 计算单幅图像的拉普拉斯边缘均值
double CalcLaplacianEdgeMean(cv::Mat& image)
{
	if (image.empty()) return 0;
	cv::Mat temp_image;
	cv::GaussianBlur(image, temp_image, cv::Size(5, 5), 0, 0, cv::BORDER_DEFAULT);
	cv::Mat dst_image;
	Laplacian(temp_image, dst_image, CV_32FC1, 3, 1, 0, cv::BORDER_DEFAULT);
	convertScaleAbs(dst_image, dst_image);
	return mean(dst_image)[0];
}

//// 计算单幅图像的灰度信息熵的原始计算方法，保留不删
//double CalcGrayEntropy(cv::Mat& image)
//{
//	if (image.empty()) return 0;
//	uto min_value = 0;
//	uto max_value = 255;
//	cv::MatND source_hist = CalcHistogram(image, min_value, max_value);
//	// 按公式计算信息熵
//	uto result = 0.0;
//	for (int i = 0; i < source_hist.total(); ++i)
//	{
//		double hist_value = source_hist.at<float>(i);
//		if (0.0 == hist_value)
//		{
//			continue;
//		}
//		result -= hist_value * log2(hist_value);
//	}
//	return result;
//}

// 改进版信息熵，有界单调，和 Consistency 特征强相关
double CalcGrayEntropy_2(cv::Mat& image)
{
	if (image.empty()) return 0;
	int min_value = 0;
	int max_value = 255;
	cv::MatND source_hist = CalcHistogram(image, min_value, max_value);
	// 按公式计算信息熵
	double result = 0.0;
	for (int i = 0; i < source_hist.total(); ++i)
	{
		double hist_value = source_hist.at<float>(i);
		if (0.0 == hist_value)
		{
			continue;
		}
		result += hist_value * exp(1 - hist_value);
	}
	return result;
}

// 计算单幅图像的Frieden灰度熵，参考自：一种前视红外视觉导航连续跟踪导航区选取方法
double CalcFriedenEntropy(cv::Mat& image)
{
	if (image.empty()) return 0;
	cv::Mat temp_image;
	image.convertTo(temp_image, CV_32FC1);
	cv::Mat proportion_mat = temp_image / sum(temp_image)[0];
	// 按公式计算
	cv::Mat exp_mat;
	exp(1 - proportion_mat, exp_mat);
	double result = -sum(proportion_mat.mul(exp_mat))[0];
	return result;
	//// 保留不删，对于多数样本，上一步的计算结果都集中在了 -2.7182 附近， 修正一下范围，但会使得Windows和NoKylin平台运行的结果差异比较大
	//result += exp(1);
	//return result * 1000; // 乘以常数1000，仅仅是为了调节结果大小，好量化
}

void CalcSobelGradXY(cv::Mat& image, cv::Mat& sobel_x, cv::Mat& sobel_y, bool need_absolute)
{
	if (image.empty()) return;
	Sobel(image, sobel_x, CV_32FC1, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
	Sobel(image, sobel_y, CV_32FC1, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
	if (!need_absolute) return;
	sobel_x = abs(sobel_x);
	sobel_y = abs(sobel_y);
}

// 计算单幅图像的方向梯度熵，越小说明轮廓方向一致性高，轮廓尺度大，越容易匹配
double CalcDirectionGradEntropy(cv::Mat& image)
{
	// 每个block都是正方形，边长越大，越是在较大图像块上统计直方图，越能反映大轮廓
	int block_edge_length = min(image.rows, image.cols) / 4; 
	if (image.empty()) return 0;
	cv::Mat grad_x, grad_y;
	CalcGradByDifference(image, grad_x, grad_y);
	// 梯度强度图
	cv::Mat intensity_mat;
	magnitude(grad_x, grad_y, intensity_mat);
	cv::Mat theta_mat = CalcGradAngleImage(grad_x, grad_y);
	QList<double> block_entropy_list;
	// 按 block_size 滑窗大小，遍历梯度角度图和梯度强度图
	int block_count_x = theta_mat.cols / block_edge_length;
	int block_count_y = theta_mat.rows / block_edge_length;
	for (int row_index = 0; row_index < block_count_y; ++row_index)
	{
		for (int col_index = 0; col_index < block_count_x; ++col_index)
		{
			int left = col_index * block_edge_length;
			int top = row_index * block_edge_length;
			cv::Mat roi_theta_mat = theta_mat(cv::Rect(left, top, block_edge_length, block_edge_length));
			cv::Mat roi_intensity_mat = intensity_mat(cv::Rect(left, top, block_edge_length, block_edge_length));
			int bins = 9; // 默认划分为 18 个方向区间，每个区间20度，认为方向一致的轮廓，误差在20度以内
			double probability[9]; // 用函数 CalcGradAngleImage 计算的 theta_mat ，元素值是0--360范围
			for (int i = 0; i < bins; ++i)
			{
				probability[i] = 0;
			}
			double total_intensity_value = 0.0;
			int bin_step = 180 / bins; // 在每个图像块内，按bins大小，按比重，统计直方图
			for (int data_index = 0; data_index < roi_theta_mat.total(); ++data_index)
			{
				float theta_value = roi_theta_mat.at<float>(data_index);
				if (theta_value > 180) theta_value -= 180;  // 从0-360映射到0-180
				float intensity_value = roi_intensity_mat.at<float>(data_index);
				int current_bin = theta_value / bin_step;
				current_bin = max(0, min(current_bin, bins - 1)); // 防止数组越界
				probability[current_bin] += intensity_value;
				total_intensity_value += intensity_value;
			}
			// 计算每个图像块的方向梯度熵
			double block_entropy = 0.0;
			for (int i = 0; i < bins; ++i)
			{
				if (0.0 == probability[i]) continue;
				probability[i] /= total_intensity_value;
				block_entropy -= probability[i] * log2(probability[i]);
			}
			// 把所有图像块的方向梯度熵，放进临时列表
			block_entropy_list.append(block_entropy);
		}
	}
	qSort(block_entropy_list.begin(), block_entropy_list.end());
	int total_count = block_count_x * block_count_y;
	double result30 = 0.0;
	double count30 = floor(total_count * 0.3);
	for (int index = total_count-1; index > total_count - count30-1; --index)
	{
		result30 += block_entropy_list.at(index);
	}
	// 对所有图像块，求平均
	result30 /= count30;
	return result30;
}

// 计算单幅图像的独立像元数，其绝对值越小越容易匹配，参考自论文：《SAR图像可匹配性研究》，正文第13页
double CalcIndependentPixels(cv::Mat& image)
{
	if (image.empty()) return 0;
	double threshold = 1 / exp(1);
	// 计算行方向上的相关长度
	int correlation_length_x = image.cols;
	int before_index = 0;
	int after_index = image.cols - 2; // 减2，比减1，可能会减少一次计算
	int current_index = (before_index + after_index) / 2;
	while (true) // 使用二分查找，加速，精度稍有影响
	{
		cv::Mat roi_before_1 = image(cv::Rect(0, 0, image.cols - current_index, image.rows));
		cv::Mat roi_before_2 = image(cv::Rect(current_index, 0, image.cols - current_index, image.rows));
		double corr_coefficient_before = CalcCorrelationCoefficient(roi_before_1, roi_before_2);

		cv::Mat roi_after_1 = image(cv::Rect(0, 0, image.cols - (current_index+2), image.rows));
		cv::Mat roi_after_2 = image(cv::Rect(current_index+2, 0, image.cols - (current_index + 2), image.rows));
		double corr_coefficient_after = CalcCorrelationCoefficient(roi_after_1, roi_after_2);

		if (corr_coefficient_before > threshold && corr_coefficient_after < threshold)
		{
			correlation_length_x = current_index + 1;  // 此处加1，或者加2，或者加0，并无太大区别
			break;
		}
		if (corr_coefficient_before < threshold && corr_coefficient_after > threshold)
		{
			correlation_length_x = current_index;
			break;
		}
		if(after_index - before_index < 5)   // 前后索引的差，减小到一定程度时，认为找到了临界点附近，退出循环
		{
			correlation_length_x = current_index;
			break;
		}
		if (corr_coefficient_before > threshold && corr_coefficient_after > threshold)  // 相关系数大于阈值，索引后移
		{
			before_index = current_index;
			current_index = (current_index + after_index) / 2;
			continue;
		}
		if (corr_coefficient_before < threshold && corr_coefficient_after < threshold)  // 相关系数小于阈值，索引前移
		{
			after_index = current_index;
			current_index = (before_index + current_index) / 2;
		}
	}
	// 计算列方向上的相关长度
	int correlation_length_y = image.rows;
	before_index = 0;
	after_index = image.rows - 2; // 减2，比减1，可能会减少一次计算
	current_index = (before_index + after_index) / 2;
	while (true) // 使用二分查找，加速，精度稍有影响
	{
		cv::Mat roi_before_1 = image(cv::Rect(0, 0, image.cols, image.rows - current_index));
		cv::Mat roi_before_2 = image(cv::Rect(0, current_index, image.cols, image.rows - current_index));
		double corr_coefficient_before = CalcCorrelationCoefficient(roi_before_1, roi_before_2);

		cv::Mat roi_after_1 = image(cv::Rect(0, 0, image.cols, image.rows - (current_index + 2)));
		cv::Mat roi_after_2 = image(cv::Rect(0, current_index + 2, image.cols, image.rows - (current_index + 2)));
		double corr_coefficient_after = CalcCorrelationCoefficient(roi_after_1, roi_after_2);

		if (corr_coefficient_before > threshold && corr_coefficient_after < threshold)
		{
			correlation_length_y = current_index + 1;  // 此处加1，或者加2，或者加0，并无太大区别
			break;
		}
		if (corr_coefficient_before < threshold && corr_coefficient_after > threshold)
		{
			correlation_length_y = current_index;
			break;
		}
		if (after_index - before_index < 5)   // 前后索引的差，减小到一定程度时，认为找到了临界点附近，退出循环
		{
			correlation_length_y = current_index;
			break;
		}
		if (corr_coefficient_before > threshold && corr_coefficient_after > threshold)  // 相关系数大于阈值，索引后移
		{
			before_index = current_index;
			current_index = (current_index + after_index) / 2;
			continue;
		}
		if (corr_coefficient_before < threshold && corr_coefficient_after < threshold)  // 相关系数小于阈值，索引前移
		{
			after_index = current_index;
			current_index = (before_index + current_index) / 2;
		}
	}
	//int zzz = 3;
	//{
	//	// 暂不删除，可用于测上面的代码的正确性，误差一般在5像素以内
	//	// 计算行方向上的相关长度
	//	int correlation_length_xxx = image.cols;
	//	for (int col_index = 1; col_index < image.cols; col_index += 5) // +=5 是为了加快计算速度，可改为二分查找
	//	{
	//		cv::Mat roi_1 = image(cv::Rect(0, 0, image.cols - col_index, image.rows));
	//		cv::Mat roi_2 = image(cv::Rect(col_index, 0, image.cols - col_index, image.rows));
	//		uto corr_coefficient = CalcCorrelationCoefficient(roi_1, roi_2);
	//		if (corr_coefficient < threshold)
	//		{
	//			correlation_length_xxx = col_index;
	//			break;
	//		}
	//	}
	//	// 计算列方向上的相关长度
	//	int correlation_length_yyy = image.rows;
	//	for (int row_index = 1; row_index < image.rows; row_index += 5) // +=5 是为了加快计算速度，可改为二分查找
	//	{
	//		cv::Mat roi_1 = image(cv::Rect(0, 0, image.cols, image.rows - row_index));
	//		cv::Mat roi_2 = image(cv::Rect(0, row_index, image.cols, image.rows - row_index));
	//		uto corr_coefficient = CalcCorrelationCoefficient(roi_1, roi_2);
	//		if (corr_coefficient < threshold)
	//		{
	//			correlation_length_yyy = row_index;
	//			break;
	//		}
	//	}
	//	qDebug() << correlation_length_x;
	//	qDebug() << correlation_length_y;
	//	qDebug() << correlation_length_xxx;
	//	qDebug() << correlation_length_yyy;
	//	int zzz = 3;
	//}

	//uto result = image.total() * 1.0 / (correlation_length_x * correlation_length_y);
	// 独立像元数的含义，是上面这样，实际使用下面的公式，数字超过10000时，GradientBoostingClassifier.fit方法报错
	double result = sqrt(correlation_length_x * correlation_length_x + correlation_length_y * correlation_length_y);
	result /= sqrt(image.cols * image.cols + image.rows * image.rows);
	return result;
}

// 计算单幅图像的零交叉密度，参考自《巡航导弹景象匹配区选取关键技术研究》
// http://www.cocoachina.com/cms/wap.php?action=article&id=79594
double CalcZeroCrossDensity2(cv::Mat& image)
{
	if (image.empty()) return 0;
	cv::Mat gauss_mat, laplacian_mat;
	GaussianBlur(image, gauss_mat, cv::Size(3, 3), 0, 0, cv::BORDER_REFLECT101);
	Laplacian(gauss_mat, laplacian_mat, CV_32F, 1, 1.0, 0.0, cv::BORDER_REFLECT101);

	double zero_cross_count = 0.0;
	for (int row = 0; row < laplacian_mat.rows - 1; ++row)
	{
		for (int column = 0; column < laplacian_mat.cols - 1; ++column)
		{
			float value = laplacian_mat.at<float>(row, column);
			float down = laplacian_mat.at<float>(row, column + 1);
			float right = laplacian_mat.at<float>(row + 1, column);
			// 横向或纵向上，若两个相邻的像素点，具有相反的符号，则认为是一个零点
			if (value < 0 && down > 0 || value > 0 && down < 0 || value < 0 && right > 0 || value > 0 && right < 0)
			{
				zero_cross_count++;
			}
		}
	}
	double result = zero_cross_count / image.total();
	return result;
}

// 计算单幅图像的纹理复杂度，参考自《巡航导弹景象匹配区选取关键技术研究》
// https://blog.csdn.net/weixin_33971130/article/details/91871201
// https://blog.csdn.net/weixin_35089891/article/details/112772196
double CalcTextureComplexity(cv::Mat& image)
{
	if (image.empty()) return 0;
	// 计算一阶偏导
	cv::Mat grad_x, grad_y;
	CalcGradByDifference(image, grad_x, grad_y, true);
	// 用Laplacian算子，计算二阶混合偏导
	cv::Mat laplacian_xy;
	Laplacian(image, laplacian_xy, CV_32F);
	convertScaleAbs(laplacian_xy, laplacian_xy);

	double result = (sum(grad_x)[0] + sum(grad_y)[0]) / sum(laplacian_xy)[0];
	if (std::isnan(result)) return -9999;
	return result;
}

// 计算单幅图像的灰度差分的直方图的相关统计特征，参考自：
// https://www.cnblogs.com/wojianxin/p/11425052.html
// https://blog.csdn.net/qq_40532645/article/details/86319846
// 暂不删除，可借鉴灰度共生矩阵的计算方法，改造为计算4个方向的差分图像，然后计算各个特征的均值，考虑是否有意义
// 使用38QMRQ数据集，测试这几个特征都会使得准确率下降
void CalcGrayDifference(cv::Mat& image, map<string, double>& feature_key_value)
{
	if (image.empty()) return;
	// 计算45方向的Roberts梯度
	cv::Mat kernel_45 = (cv::Mat_<int>(2, 2) << 0, -1, 1, 0);
	cv::Mat difference_45;
	filter2D(image, difference_45, CV_32FC1, kernel_45);
	difference_45 = abs(difference_45);
	// 计算差分图像的归一化直方图，作为差分图像取某一灰度值的概率图
	int min_value = 0;
	int max_value = 255;
	cv::MatND histogram = CalcHistogram(difference_45, min_value, max_value);
	double mean = 0.0; // 灰度差分均值
	double angular_second_moment = 0.0; // 角度方向二阶矩
	double contrast = 0.0; // 灰度差分对比度
	double entropy = 0.0; // 灰度差分熵
	for (int index = 0; index < histogram.rows; ++index)
	{
		float value = histogram.at<float>(index);
		float temp = index * value;
		mean += temp;
		contrast += index * temp;
		angular_second_moment += value * value;
		if (value > 0)
			entropy -= value * log2(value);
	}
	feature_key_value["DiffMean"] = mean / 256;
	feature_key_value["DiffContrast"] = contrast;
	feature_key_value["DiffEntropy"] = entropy;
	feature_key_value["DiffAngularSecondMoment"] = angular_second_moment;
}

// 计算单幅图像的灰度直方图统计矩的相关特征，参考自：http://www.doc88.com/p-7177453334066.html
void CalcGrayHistogramMoment(cv::Mat& image, map<string, double>& feature_key_value)
{
	if (image.empty()) return;
	int min_value = 0;
	int max_value = 255;
	cv::MatND histogram = CalcHistogram(image, min_value, max_value);
	cv::Scalar mean, std_dev;
	meanStdDev(image, mean, std_dev);
	feature_key_value["GrayScaleMean"] = mean[0];
	feature_key_value["GrayScaleStdDev"] = std_dev[0];
	// 保留不删，用灰度直方图的方式计算出的一阶矩等于图像均值，二阶矩等于图像方差，用上面的OpenCV自带的函数代替
	//uto gray_scale_mean = 0.0;
	//for (int index = 0; index < histogram.total(); ++index)
	//{
	//	gray_scale_mean += index * histogram.at<float>(index);
	//}
	//feature_key_value["GrayScaleMean"] = gray_scale_mean;
	//if (feature_key_value.find("GrayScaleStdDev") != feature_key_value.end())
	//{
	//	uto std_dev = 0.0;
	//	for (int index = 0; index < histogram.total(); ++index)
	//	{
	//		std_dev += pow(index - gray_scale_mean,2) * histogram.at<float>(index);
	//	}
	//	feature_key_value["GrayScaleStdDev"] = std_dev;
	//}

	//if (feature_key_value.find("Smoothness") != feature_key_value.end()) // 平滑度描述子，由二阶矩计算得到，暂时舍弃
	//{
	//	feature_key_value["Smoothness"] = 1 - 1 / (1 + pow(std_dev[0], 2));
	//}

	// 一致性
	{
		double consistency = 0.0;
		for (int index = 0; index < histogram.total(); ++index)
		{
			consistency += pow(histogram.at<float>(index), 2);
		}
		feature_key_value["Consistency"] = consistency;
	}
	// 三阶矩，表示图像偏斜度，直方图向左偏斜时，三阶矩为负，直方图向右偏斜时，三阶矩为正
	//	if (feature_key_value.find("ThirdOrderMoment") != feature_key_value.end())
	{
		double third_order_moment = 0.0;
		double fourth_order_moment = 0.0;
		for (int index = 0; index < histogram.total(); ++index)
		{
			third_order_moment += pow(index - mean[0], 3) * histogram.at<float>(index);
			fourth_order_moment += pow(index - mean[0], 4) * histogram.at<float>(index);
		}
		feature_key_value["ThirdOrderMoment"] = third_order_moment / image.total(); // 除以图像像素总数，因计算的数字很大
		feature_key_value["FourthOrderMoment"] = fourth_order_moment / image.total(); // 除以图像像素总数，因计算的数字很大
	}
}

// 计算单幅图像的灰度共生矩阵(Gray Level Co-occurrence Matrix)的相关特征，
// 参考自：《基于颜色和纹理特征的图像复杂度研究_王浩》、《基于纹理特征的图像复杂度研究_陈燕芹》
// 参考自： https://blog.csdn.net/qq_37059483/article/details/78292869
void CalcGLCM(cv::Mat& image, map<string, double>& feature_key_value)
{
	if (image.empty()) return;
	int gray_scale = 16;
	// 8位是256个灰度级，4位是0-15共16个灰度级，8位转4位，是为了减少计算量
	cv::Mat image_4bit = image / 16;
	// 0--255的cv::Mat的各元素值除以16，得到的cv::Mat中的元素有可能大于15
	for (int index = 0; index < image_4bit.total(); ++index)
	{
		if (image_4bit.at<uchar>(index) > 15)
		{
			image_4bit.at<uchar>(index) = 15;
		}
	}
	cv::Mat co_occurence_mat0 = cv::Mat::zeros(gray_scale, gray_scale, CV_32F);
	cv::Mat co_occurence_mat45 = cv::Mat::zeros(gray_scale, gray_scale, CV_32F);
	cv::Mat co_occurence_mat90 = cv::Mat::zeros(gray_scale, gray_scale, CV_32F);
	cv::Mat co_occurence_mat135 = cv::Mat::zeros(gray_scale, gray_scale, CV_32F);
	for (int row_index = 0; row_index < image_4bit.rows - 1; ++row_index)
	{
		for (int col_index = 0; col_index < image_4bit.cols - 1; ++col_index)
		{
			unsigned char value = image_4bit.at<uchar>(row_index, col_index);
			unsigned char value0 = image_4bit.at<uchar>(row_index, col_index + 1);
			unsigned char value45 = image_4bit.at<uchar>(row_index + 1, col_index + 1);
			unsigned char value90 = image_4bit.at<uchar>(row_index + 1, col_index);
			co_occurence_mat0.at<float>(value, value0)++;
			co_occurence_mat45.at<float>(value, value45)++;
			co_occurence_mat90.at<float>(value, value90)++;
			// 使用偏移下标的方式，计算135度方向上的灰度共生矩阵
			int row_1351 = row_index;
			int col_1351 = image_4bit.cols - 1 - col_index;
			int row_1352 = row_1351 + 1;
			int col_1352 = col_1351 - 1;
			unsigned char value_1351 = image_4bit.at<uchar>(row_1351, col_1351);
			unsigned char value_1352 = image_4bit.at<uchar>(row_1352, col_1352);
			co_occurence_mat135.at<float>(value_1351, value_1352)++;
		}
	}
	// 把0度方向的灰度共生矩阵，加上最后一行
	for (int col_index = 0; col_index < image_4bit.cols - 1; ++col_index)
	{
		int row = image_4bit.at<uchar>(image_4bit.rows - 1, col_index);
		int col = image_4bit.at<uchar>(image_4bit.rows - 1, col_index + 1);
		co_occurence_mat0.at<float>(row, col)++;
	}
	// 把90度方向的灰度共生矩阵，加上最后一列
	for (int row_index = 0; row_index < image_4bit.rows - 1; ++row_index)
	{
		int row = image_4bit.at<uchar>(row_index, image_4bit.cols - 1);
		int col = image_4bit.at<uchar>(row_index + 1, image_4bit.cols - 1);
		co_occurence_mat90.at<float>(row, col)++;
	}
	// 把灰度共生矩阵归一化，使得矩阵中所有像素值的和为0
	co_occurence_mat0 /= sum(co_occurence_mat0)[0];
	co_occurence_mat45 /= sum(co_occurence_mat45)[0];
	co_occurence_mat90 /= sum(co_occurence_mat90)[0];
	co_occurence_mat135 /= sum(co_occurence_mat135)[0];
	// 特征变量初始化
	double asm11 = 0.0; // ASM 能量（angular second moment，角度方向二阶矩)
	double entropy = 0.0; // 灰度共生矩阵熵
	double contrast = 0.0; // 灰度共生矩阵对比度，惯性矩
	double homogeneity = 0.0; // 同质性/逆差距（Homogeneity）
	double correlation0 = 0.0, correlation45 = 0.0, correlation90 = 0.0, correlation135 = 0.0; // 相关性
	double ux0 = 0.0, uy0 = 0.0, delta_x0 = 0.0, delta_y0 = 0.0;
	double ux45 = 0.0, uy45 = 0.0, delta_x45 = 0.0, delta_y45 = 0.0;
	double ux90 = 0.0, uy90 = 0.0, delta_x90 = 0.0, delta_y90 = 0.0;
	double ux135 = 0.0, uy135 = 0.0, delta_x135 = 0.0, delta_y135 = 0.0;
	for (int row_index = 0; row_index < gray_scale; ++row_index)
	{
		for (int col_index = 0; col_index < gray_scale; ++col_index)
		{
			float value0 = co_occurence_mat0.at<float>(row_index, col_index);
			float value45 = co_occurence_mat45.at<float>(row_index, col_index);
			float value90 = co_occurence_mat90.at<float>(row_index, col_index);
			float value135 = co_occurence_mat135.at<float>(row_index, col_index);
			// 能量
			asm11 += value0 * value0 + value45 * value45 + value90 * value90 + value135 * value135;
			// 灰度共生矩阵熵
			if (value0 > 0)
				entropy -= value0 * log2(value0);
			if (value45 > 0)
				entropy -= value45 * log2(value45);
			if (value90 > 0)
				entropy -= value90 * log2(value90);
			if (value135 > 0)
				entropy -= value135 * log2(value135);
			// 灰度共生矩阵对比度
			double temp = pow(static_cast<double>(row_index - col_index), 2);
			contrast += temp * (value0 + value45 + value90 + value135);
			// 同质性/逆差距（Homogeneity）
			homogeneity += (value0 + value45 + value90 + value135) / (1 + temp);
			// 均值			
			ux0 += row_index * value0;
			uy0 += col_index * value0;
			ux45 += row_index * value45;
			uy45 += col_index * value45;
			ux90 += row_index * value90;
			uy90 += col_index * value90;
			ux135 += row_index * value135;
			uy135 += col_index * value135;
		}
	}
	// 灰度共生矩阵相关度特征计算，参考自：《基于颜色和纹理特征的图像复杂度研究_王浩》
	// https://www.pianshen.com/article/67152054784/ ，https://www.pianshen.com/article/7007309871/
	for (int row_index = 0; row_index < gray_scale; ++row_index)
	{
		for (int col_index = 0; col_index < gray_scale; ++col_index)
		{
			float value0 = co_occurence_mat0.at<float>(row_index, col_index);
			float value45 = co_occurence_mat45.at<float>(row_index, col_index);
			float value90 = co_occurence_mat90.at<float>(row_index, col_index);
			float value135 = co_occurence_mat135.at<float>(row_index, col_index);
			correlation0 += row_index * col_index * value0;
			correlation45 += row_index * col_index * value45;
			correlation90 += row_index * col_index * value90;
			correlation135 += row_index * col_index * value135;
			// 方差
			delta_x0 += pow(row_index - ux0, 2) * value0;
			delta_y0 += pow(col_index - uy0, 2) * value0;
			delta_x45 += pow(row_index - ux45, 2) * value45;
			delta_y45 += pow(col_index - uy45, 2) * value45;
			delta_x90 += pow(row_index - ux90, 2) * value90;
			delta_y90 += pow(col_index - uy90, 2) * value90;
			delta_x135 += pow(row_index - ux135, 2) * value135;
			delta_y135 += pow(col_index - uy135, 2) * value135;
		}
	}
	correlation0 = (correlation0 - ux0 * uy0) / (delta_x0 * delta_y0);
	correlation45 = (correlation45 - ux45 * uy45) / (delta_x45 * delta_y45);
	correlation90 = (correlation90 - ux90 * uy90) / (delta_x90 * delta_y90);
	correlation135 = (correlation135 - ux135 * uy135) / (delta_x135 * delta_y135);
	feature_key_value["GLCMASM"] = asm11 / 4; // 能量
	feature_key_value["GLCMEntropy"] = entropy / 4; // 熵
	feature_key_value["GLCMContrast"] = contrast / 4; // 对比度  加上这个特征，使得38QMRQ数据集的准确率下降从0.781149到0.777276
	feature_key_value["GLCMHomogeneity"] = homogeneity / 4; // 逆差矩
	feature_key_value["GLCMCorrelation"] = (correlation0 + correlation45 + correlation90 + correlation135) / 4; // 相关度
	if (std::isnan(feature_key_value["GLCMCorrelation"]))
	{
		feature_key_value["GLCMCorrelation"] = 0.0;
	}
}

// 计算图像的Hu矩相关特征
void CalcHuMoment(cv::Mat& image, map<string, double>& feature_key_value)
{
	cv::Moments mts = moments(image);
	double hu[7];
	cv::HuMoments(mts, hu);
	feature_key_value["HuMoment0"] = log(abs(hu[0]));
	feature_key_value["HuMoment1"] = log(abs(hu[1]));
	feature_key_value["HuMoment2"] = log(abs(hu[2]));
	feature_key_value["HuMoment3"] = log(abs(hu[3]));
	feature_key_value["HuMoment4"] = log(abs(hu[4]));
	feature_key_value["HuMoment5"] = log(abs(hu[5]));
	feature_key_value["HuMoment6"] = log(abs(hu[6]));
}

// 计算单幅图像的频域相关的特征
// 参考自：https://blog.csdn.net/xddwz/article/details/110938652
// 参考自：https://www.cnblogs.com/HL-space/p/10546602.html
// 计算频域熵(FDE, Frequency Domain Entropy), 参考自：https://blog.csdn.net/u011178262/article/details/124060454
void CalcFourierVeinsPercent(cv::Mat& image, std::map<std::string, float>& feature_key_value)
{
	if (image.empty()) return;
	cv::Mat fourier = ConvertToFourier(image);
	vector<cv::Mat> channels;
	split(fourier, channels);
	cv::Mat real_part = channels[0]; // 实部
	cv::Mat imaginary_part = channels[1]; // 虚部
	cv::Mat amplitude; // 幅度
	magnitude(real_part, imaginary_part, amplitude);
	// 取对数后，归一化，转换类型
	amplitude += cv::Scalar(1);
	log(amplitude, amplitude);
	normalize(amplitude, amplitude, 0, 255, cv::NORM_MINMAX);
	amplitude.convertTo(amplitude, CV_8UC1);
	// 交换位置，把原点(代表低频能量)放在图像中心位置
	cv::Mat mQuadrant1 = amplitude(cv::Rect(amplitude.cols / 2, 0, amplitude.cols / 2, amplitude.rows / 2));
	cv::Mat mQuadrant2 = amplitude(cv::Rect(0, 0, amplitude.cols / 2, amplitude.rows / 2));
	cv::Mat mQuadrant3 = amplitude(cv::Rect(0, amplitude.rows / 2, amplitude.cols / 2, amplitude.rows / 2));
	cv::Mat mQuadrant4 = amplitude(cv::Rect(amplitude.cols / 2, amplitude.rows / 2, amplitude.cols / 2, amplitude.rows / 2));
	cv::Mat mChange1 = mQuadrant1.clone(); // 交换左下和右上象限
	mQuadrant3.copyTo(mQuadrant1);
	mChange1.copyTo(mQuadrant3);
	cv::Mat mChange2 = mQuadrant2.clone(); // 交换左上和右下象限
	mQuadrant4.copyTo(mQuadrant2);
	mChange2.copyTo(mQuadrant4);
	// 按高频能量的区域，计算纹理能量百分百
	double total_sum = sum(amplitude)[0];
	int left16 = (amplitude.cols - amplitude.cols / 16) / 2; // 中心位置
	int top16 = (amplitude.rows - amplitude.rows / 16) / 2;
	cv::Mat percent16_mat = amplitude(cv::Rect(left16, top16, amplitude.cols / 16, amplitude.rows / 16));
	feature_key_value["FourierBW16"] = sum(percent16_mat)[0] / total_sum;
	int left8 = (amplitude.cols - amplitude.cols / 8) / 2; // 从1/8到1/16位置

	int top8 = (amplitude.rows - amplitude.rows / 8) / 2;
	cv::Mat percent8_mat = amplitude(cv::Rect(left8, top8, amplitude.cols / 8, amplitude.rows / 8));
	feature_key_value["FourierBW8"] = sum(percent8_mat)[0] / total_sum - feature_key_value["FourierBW16"];
	int
	left4 = (amplitude.cols - amplitude.cols / 4) / 2; // 从1/4到1/8位置
	int top4 = (amplitude.rows - amplitude.rows / 4) / 2;
	cv::Mat percent4_mat = amplitude(cv::Rect(left4, top4, amplitude.cols / 4, amplitude.rows / 4));
	feature_key_value["FourierBW4"] = sum(percent4_mat)[0] / total_sum - feature_key_value["FourierBW16"] - feature_key_value["FourierBW8"];
	int left2 = (amplitude.cols - amplitude.cols / 2) / 2; // 从1/2到1/14位置
	int top2 = (amplitude.rows - amplitude.rows / 2) / 2;
	cv::Mat percent2_mat = amplitude(cv::Rect(left2, top2, amplitude.cols / 2, amplitude.rows / 2));
	feature_key_value["FourierBW2"] = sum(percent2_mat)[0] / total_sum
		- feature_key_value["FourierBW16"] - feature_key_value["FourierBW8"] - feature_key_value["FourierBW4"];
	// 按高频能量阈值，计算高频信息比，参考自论文：巡航导弹景象匹配区选取关键技术研究
	double high_frequency_percent80 = 0;
	double high_frequency_percent112 = 0;
	double high_frequency_percent144 = 0;
	double high_frequency_percent176 = 0;
	for (int i = 0; i < amplitude.total(); ++i)
	{
		unsigned char value = amplitude.at<uchar>(i);
		if (value < 80) continue;
		high_frequency_percent80 += value;
		if (value < 112) continue;
		high_frequency_percent112 += value;
		if (value < 144) continue;
		high_frequency_percent144 += value;
		if (value < 176) continue;
		high_frequency_percent176 += value;
	}
	feature_key_value["HighFrequencyPercent80"] = high_frequency_percent80 / total_sum;
	feature_key_value["HighFrequencyPercent112"] = high_frequency_percent112 / total_sum;
	feature_key_value["HighFrequencyPercent144"] = high_frequency_percent144 / total_sum;
	feature_key_value["HighFrequencyPercent176"] = high_frequency_percent176 / total_sum;
	// 参考自：https://github.com/cggos/ccv/blob/a97ae018136388a33cc821ada46cb3a6466dff1c/scripts/cv_py/fft_fde.py
	feature_key_value["FrequencyDomainEntropy"] = CalcGrayEntropy_2(amplitude);
	// 计算最小局部高频信息均值，参考自论文：巡航导弹景象匹配区选取关键技术研究
	cv::Mat result = FrequencyFilter(fourier, image.cols, image.rows);
	double min_mean, min_std_dev;
	CalcMaxMinLocalStdDev(result, min_mean, min_std_dev);
	feature_key_value["MinLocalHighFrequencyMean"] = min_mean;
}

cv::Mat ConvertToFourier(cv::Mat& image)
{
	if (image.empty()) return image;
	// 插值扩展一点图像，是为了提高计算速度
	int row = cv::getOptimalDFTSize(image.rows);
	int col = cv::getOptimalDFTSize(image.cols);
	cv::Mat source_image = image.clone();
	copyMakeBorder(image, source_image, 0, row - image.rows, 0, col - image.cols, cv::BORDER_CONSTANT, cv::Scalar(0));
	// 进行傅里叶变换，计算幅值
	cv::Mat fourier(source_image.rows + row, source_image.cols + col, CV_32FC2, cv::Scalar(0, 0));
	cv::Mat for_fourier[] = {cv::Mat_<float>(source_image), cv::Mat::zeros(source_image.size(), CV_32F)};
	merge(for_fourier, 2, source_image);
	dft(source_image, fourier);
	return fourier;
}

cv::Mat _FourierMask;
// 生成高通滤波的掩模图，中间是黑色0，其余部分是白色1，可保留住高频信号，去掉低频信号
cv::Mat GetFourierMask(int col, int row)
{
	if (!_FourierMask.empty())
	{
		if (_FourierMask.cols != col || _FourierMask.rows != row)
		{
			resize(_FourierMask, _FourierMask, cv::Size(col, row));
		}
		return _FourierMask;
	}
	_FourierMask = cv::Mat::ones(cv::Size(col, row), CV_32FC1);
	// 默认半径是宽高均值的一半的一半
	int radius = (row + col) / 2 / 2 / 3 * 2;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			double d = sqrt(pow((i - row / 2), 2) + pow((j - col / 2), 2));
			if (d <= radius)
			{
				_FourierMask.at<float>(i, j) = 0;
			}
		}
	}
	return _FourierMask;
}

// 频域滤波后，转空域，参考自： https://blog.csdn.net/cyf15238622067/article/details/87933464
cv::Mat FrequencyFilter(cv::Mat& fourier, int cols, int rows)
{
	vector<cv::Mat> channel;
	split(fourier, channel);
	int cx = channel[0].cols / 2;
	int cy = channel[0].rows / 2; //以下的操作是移动图像  (零频移到中心)
	cv::Mat part1_r(channel[0], cv::Rect(0, 0, cx, cy)); //元素坐标表示为(cx,cy)
	cv::Mat part2_r(channel[0], cv::Rect(cx, 0, cx, cy));
	cv::Mat part3_r(channel[0], cv::Rect(0, cy, cx, cy));
	cv::Mat part4_r(channel[0], cv::Rect(cx, cy, cx, cy));

	cv::Mat temp;
	part1_r.copyTo(temp); //左上与右下交换位置(实部)
	part4_r.copyTo(part1_r);
	temp.copyTo(part4_r);

	part2_r.copyTo(temp); //右上与左下交换位置(实部)
	part3_r.copyTo(part2_r);
	temp.copyTo(part3_r);

	cv::Mat part1_i(channel[1], cv::Rect(0, 0, cx, cy)); //元素坐标(cx,cy)
	cv::Mat part2_i(channel[1], cv::Rect(cx, 0, cx, cy));
	cv::Mat part3_i(channel[1], cv::Rect(0, cy, cx, cy));
	cv::Mat part4_i(channel[1], cv::Rect(cx, cy, cx, cy));

	part1_i.copyTo(temp); //左上与右下交换位置(虚部)
	part4_i.copyTo(part1_i);
	temp.copyTo(part4_i);

	part2_i.copyTo(temp); //右上与左下交换位置(虚部)
	part3_i.copyTo(part2_i);
	temp.copyTo(part3_i);

	cv::Mat fourier_mask = GetFourierMask(channel.at(0).cols, channel.at(0).rows);

	// 用高通滤波器，进行滤波
	cv::Mat BLUR;
	cv::Mat real_part = channel.at(0).mul(fourier_mask);
	cv::Mat imaginary_part = channel.at(1).mul(fourier_mask);
	cv::Mat plane1[] = {real_part, imaginary_part};
	merge(plane1, 2, BLUR);

	idft(BLUR, BLUR); // idft的结果也为复数
	split(BLUR, channel);
	magnitude(channel[0], channel[1], channel[0]);
	normalize(channel[0], channel[0], 0, 255, cv::NORM_MINMAX);
	channel[0].convertTo(channel[0], CV_8UC1);
	// 空域转频域前，用OpenCV的getOptimalDFTSize函数扩展了图像，此处再裁剪回来
	channel[0] = channel[0](cv::Rect(0, 0, cols, rows));
	return channel[0];
}

// 计算后一个减去前一个的差分的梯度，is_absolute表示是否对结果取绝对值
void CalcGradByDifference(cv::Mat& image, cv::Mat& grad_x, cv::Mat& grad_y, bool is_absolute)
{
	if (image.empty()) return;
	cv::Mat kernel_x = (cv::Mat_<char>(1, 3) << -1, 0, 1);
	filter2D(image, grad_x, CV_32FC1, kernel_x);

	cv::Mat kernel_y = (cv::Mat_<char>(3, 1) << -1, 0, 1);
	filter2D(image, grad_y, CV_32FC1, kernel_y);
	if (!is_absolute) return;
	grad_x = abs(grad_x);
	grad_y = abs(grad_y);
}

// 计算单幅图像的等效视数(ENL, Equivalent Numbers of Look)
double CalcENL(cv::Mat& image)
{
	if (image.empty()) return 0;
	cv::Mat mean_mat, std_dev_mat;
	meanStdDev(image, mean_mat, std_dev_mat);
	double mean_double = mean_mat.at<double>(0, 0);
	double std_dev_double = std_dev_mat.at<double>(0, 0);
	double result = mean_double * mean_double / (std_dev_double * std_dev_double);
	return result;
}

// 计算单幅图像的图像对比度，参考自：
// https://blog.csdn.net/weixin_45342712/article/details/96591834
// https://blog.csdn.net/lien0906/article/details/40742409
double CalcGrayContrast(cv::Mat& image)
{
	if (image.empty()) return 0;
	double result = 0.0;
	for (int row_index = 0; row_index < image.rows - 1; ++row_index)
	{
		for (int col_index = 0; col_index < image.cols - 1; ++col_index)
		{
			unsigned char value = image.at<uchar>(row_index, col_index);
			// 按4邻域计算，像素差值的平方和
			if (0 != row_index)
			{
				unsigned char value_up = image.at<uchar>(row_index - 1, col_index);
				result += pow(value - value_up, 2);
			}
			if (image.rows - 1 != row_index)
			{
				unsigned char value_down = image.at<uchar>(row_index + 1, col_index);
				result += pow(value - value_down, 2);
			}
			if (0 != col_index)
			{
				unsigned char value_left = image.at<uchar>(row_index, col_index - 1);
				result += pow(value - value_left, 2);
			}
			if (image.cols - 1 != col_index)
			{
				unsigned char value_right = image.at<uchar>(row_index, col_index + 1);
				result += pow(value - value_right, 2);
			}
		}
	}
	// 差值的个数：图像的中间区域、图像的上边界和下边界区域、图像的左边界和右边界区域、图像的4个角点区域
	int count = 4 * (image.rows - 2) * (image.cols - 2) + 2 * (image.rows - 2) * 3 + 2 * (image.cols - 2) * 3 + 4 * 2;
	result /= count;
	return result;
}

// 计算单幅图像的梯度图特征：信息熵，能量，对比度和均匀度
// 参考自： https://blog.csdn.net/qq_48176859/article/details/110050055
void CalcGradientFeature(cv::Mat gradient_image, map<string, double>& feature_key_value)
{
	if (gradient_image.empty()) return;
	cv::Mat temp_image = gradient_image / sum(gradient_image)[0]; // 对整个梯度矩阵归一化
	double entropy = 0.0, energy = 0.0, contrast = 0.0, evenness = 0.0;
	for (int col_index = 0; col_index < temp_image.cols; ++col_index)
	{
		for (int row_index = 0; row_index < temp_image.rows; ++row_index)
		{
			float value = temp_image.at<float>(row_index, col_index);
			if(0.0 == value) continue;
			entropy += -value * log2(value);
			energy += value * value;
			contrast += pow(row_index - col_index, 2) * value;
			evenness += value / (1 + abs(row_index - col_index));
		}
	}
	feature_key_value["GradientWeightEntropy"] = entropy; 
	feature_key_value["GradientEnergy"] = energy*1000000; // 乘以常数1000000，仅仅是为了调节结果大小，好量化;
	feature_key_value["GradientContrast"] = contrast/10000; // 除以常数10000，仅仅是为了调节结果大小，好量化;
	feature_key_value["GradientEvenness"] = evenness;   // 梯度均匀度
	// 转8位后，计算直方图相关特征等
	gradient_image.convertTo(gradient_image, CV_8UC1);
	int min_value = 0;
	int max_value = 255;
	cv::MatND source_hist = CalcHistogram(gradient_image, min_value, max_value);
	// 按公式计算信息熵
	double result = 0.0;
	for (int i = 0; i < source_hist.total(); ++i)
	{
		double hist_value = source_hist.at<float>(i);
		if (0.0 == hist_value)
		{
			continue;
		}
		result -= hist_value * log2(hist_value);
	}
	feature_key_value["GradientProportionEntropy"] = result;
}

// 计算单幅图像的图像清晰度，和梯度强度均值类似，只是算子不同
double CalcDefinition(cv::Mat& image)
{
	if (image.empty()) return 0;
	cv::Mat grad_x_mat, grad_y_mat;
	CalcGradByDifference(image, grad_x_mat, grad_y_mat);
	// 按公式计算清晰度
	cv::Mat grad_mat;
	magnitude(grad_x_mat, grad_y_mat, grad_mat);
	double result = mean(grad_mat)[0] / sqrt(2);
	return result;
}

// 计算单幅图像的信噪比
double CalcSingnalNoiseRatio(cv::Mat& image)
{
	//// 第一种算法：均值除以标准差
	//double dMean = 0, dStdDev = 0;
	//CalcMeanStdDev(image, dMean, dStdDev);
	//uto result = dMean / dStdDev;
	//return result;

	// 第二种算法：最大标准差除以最小标准差
	double max_std_dev = 0, min_std_dev = 0;
	CalcMaxMinLocalStdDev(image, max_std_dev, min_std_dev);

	double signal_noise_rotio = 6.66666; // 信噪比默认等于 6.66666 ，大约是平均值
	if (max_std_dev != 0)
	{
		signal_noise_rotio = 1 -  min_std_dev / max_std_dev;
	}
	return signal_noise_rotio;
}

// 计算单幅图像的Roberts边缘梯度强度均值，原先叫：基于边缘的图像质量评价
double CalcRobertGradientMean(cv::Mat& image)
{
	// 计算135方向的Robrets梯度
	cv::Mat kernel_135 = (cv::Mat_<int>(2, 2) << -1, 0, 0, 1);
	cv::Mat gradient_135;
	filter2D(image, gradient_135, CV_32FC1, kernel_135);
	// 计算45方向的Roberts梯度
	cv::Mat kernel_45 = (cv::Mat_<int>(2, 2) << 0, -1, 1, 0);
	cv::Mat gradient_45;
	filter2D(image, gradient_45, CV_32FC1, kernel_45);
	// 按公式计算
	cv::Mat gradient_mat;
	magnitude(gradient_135, gradient_45, gradient_mat);
	double result = mean(gradient_mat)[0];
	return result;
}

// 实际测试，把准确率降低了，从0.791478 到 0.785668
int CalcOrbKeyPointCount(cv::Mat& big_image)
{
	vector<cv::KeyPoint> keypoints_small, keypoints_big;
	cv::Mat descriptors_small, descriptors_big;
	cv::Ptr<cv::ORB> detector = cv::ORB::create();
	detector->detectAndCompute(big_image, cv::Mat(), keypoints_big, descriptors_big);
	return keypoints_big.size();
}

//// 使用Sift算法计算两幅图像的匹配点数
//double CalcSiftMatchedPointCount(cv::Mat& big_image, cv::Mat& small_image)
//{
//	cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
//	std::vector<cv::KeyPoint>  kps1, kps2;
//	cv::Mat desc1, desc2;
//	sift->detectAndCompute(big_image, cv::Mat(), kps1, desc1);
//	sift->detectAndCompute(small_image, cv::Mat(), kps2, desc2);
//	// 基于 FLANN 匹配
//	cv::Ptr<cv::FlannBasedMatcher> knnmatcher = cv::FlannBasedMatcher::create();
//	std::vector<std::vector<cv::DMatch> > matches;
//	knnmatcher->knnMatch(desc1, desc2, matches, 2);
//	std::vector<cv::DMatch> good_matches;
//	for (size_t i = 0; i < matches.size(); i++)
//	{
//		uto threshold = 0.6; // 如果最近距离除以次近距离小于某个阈值，则判定为一对匹配点
//		if (matches[i][0].distance < threshold * matches[i][1].distance)
//		{
//			good_matches.push_back(matches[i][0]);
//		}
//	}
//	// 不删，测试用
//	//Mat result;
//	//drawMatches(big_image, keypoints_big, small_image, keypoints_small, goodMatches, result);
//	//imshow("匹配点对", result);
//	//waitKey(0);
//	return good_matches.size() * 1000.0 / sqrt(small_image.total()); // 乘以常数1000，仅仅是为了调节结果大小，好量化
//}
//
//// 使用Surf算法计算两幅图像的匹配点数
//double CalcSurfMatchedPointCount(cv::Mat& scene_image, cv::Mat& object_image)
//{
//	//检测特征点
//	const int minHessian = 400;
//	cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);
//	vector<cv::KeyPoint> keypoints_object, keypoints_scene;
//	detector->detect(object_image, keypoints_object);
//	detector->detect(scene_image, keypoints_scene);
//	//计算特征点描述子
//	cv::Mat descriptors_object, descriptors_scene;
//	detector->compute(object_image, keypoints_object, descriptors_object);
//	detector->compute(scene_image, keypoints_scene, descriptors_scene);
//	//使用FLANN进行特征点匹配
//	cv::FlannBasedMatcher matcher;
//	vector<cv::DMatch> matches;
//	matcher.match(descriptors_object, descriptors_scene, matches);
//	//计算匹配点之间的最大和最小距离
//	double min_dist = 100;
//	for (int i = 0; i < descriptors_object.rows; i++)
//	{
//		double dist = matches[i].distance;
//		if (dist < min_dist)
//		{
//			min_dist = dist;
//		}
//	}
//	//绘制好的匹配点
//	vector<cv::DMatch> good_matches;
//	for (int i = 0; i < descriptors_object.rows; i++)
//	{
//		if (matches[i].distance < 2 * min_dist)
//		{
//			good_matches.push_back(matches[i]);
//		}
//	}
//	// 不删，测试用
//	//Mat result;
//	//drawMatches(big_image, keypoints_big, small_image, keypoints_small, goodMatches, result);
//	//imshow("匹配点对", result);
//	//waitKey(0);
//	return good_matches.size() * 1000.0 / sqrt(object_image.total()); // 乘以常数1000，仅仅是为了调节结果大小，好量化
//}
//
//// 使用Orb算法计算两幅图像的匹配点数
//double CalcOrbMatchedPointCount(cv::Mat& big_image, cv::Mat& small_image)
//{
//	vector<cv::KeyPoint> keypoints_small, keypoints_big;
//	cv::Mat descriptors_small, descriptors_big;
//	cv::Ptr<cv::ORB> detector = cv::ORB::create();
//	detector->detectAndCompute(big_image, cv::Mat(), keypoints_big, descriptors_big);
//	detector->detectAndCompute(small_image, cv::Mat(), keypoints_small, descriptors_small);
//	vector<cv::DMatch> matches;
//	uto matcher = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
//	matcher->match(descriptors_small, descriptors_big, matches);
//	// 发现匹配点
//	vector<cv::DMatch> good_matches;
//	float maxdist = 0;
//	for (unsigned int i = 0; i < matches.size(); ++i)
//	{
//		maxdist = max(maxdist, matches[i].distance);
//	}
//	for (unsigned int i = 0; i < matches.size(); ++i)
//	{
//		uto threshold = 0.4; // 如果距离小于最大距离乘以某个阈值，则判定为一对匹配点
//		if (matches[i].distance < maxdist * threshold)
//			good_matches.push_back(matches[i]);
//	}
//	// 不删，测试用
//	//Mat result;
//	//drawMatches(big_image, keypoints_big, small_image, keypoints_small, goodMatches, result);
//	//imshow("匹配点对", result);
//	//waitKey(0);
//	return good_matches.size() * 1000.0 / sqrt(small_image.total()); // 乘以常数1000，仅仅是为了调节结果大小，好量化
//}

// 计算单幅图像的灰度-梯度共生矩阵相关特征
void CalcGGCM(cv::Mat& image, map<string, double>& feature_key_value)
{
	int gray_scale = 16;
	// 8位是256个灰度级，4位是0-15共16个灰度级，8位转4位，是为了减少计算量
	cv::Mat image_4bit = image / 16;
	// 0--255的cv::Mat的各元素值除以16，得到的cv::Mat中的元素有可能大于15
	for (int index = 0; index < image_4bit.total(); ++index)
	{
		if (image_4bit.at<uchar>(index) > 15)
		{
			image_4bit.at<uchar>(index) = 15;
		}
	}
	// image_4bit = image_4bit * 16; // 调试用，看是否正确，不删
	// 计算图像的梯度强度图，并把每个元素值缩放到0-15之间
	cv::Mat gradient_image = CalcGradientIntensity(image);
	gradient_image.convertTo(gradient_image, CV_8UC1);
	double max_value = 0.0, min_value = 0.0;
	minMaxLoc(gradient_image, &min_value, &max_value);
	double max_min_diff = max_value - min_value;
	for (int row = 0; row < gradient_image.rows; ++row)
	{
		for (int col = 0; col < gradient_image.cols; ++col)
		{
			int value = gradient_image.at<uchar>(row, col);
			value = (value - min_value) / max_min_diff * 16;
			value = min(max(value, 0), 15);
			gradient_image.at<uchar>(row, col) = value;
		}
	}
	// gradient_intensity_image *= 16; // 调试用，看是否正确，不删
	// 计算灰度梯度共生矩阵
	cv::Mat ggcm = cv::Mat::zeros(gray_scale, gray_scale, CV_32FC1);
	for (int row = 0; row < image_4bit.rows; ++row)
	{
		for (int col = 0; col < image_4bit.cols; ++col)
		{
			unsigned char gray = image_4bit.at<uchar>(row, col);
			unsigned char gradient = gradient_image.at<uchar>(row, col);
			ggcm.at<float>(gray, gradient)++;
		}
	}
	// 计算灰度梯度共生矩阵的特征参数
	feature_key_value["SmallGradientDominance"] = 0.0; // 1. 小梯度优势
	feature_key_value["BigGradientDominance"] = 0.0; //  2. 大梯度优势
	for (int row = 0; row < ggcm.rows; ++row)
	{
		for (int col = 0; col < ggcm.cols; ++col)
		{
			float value = ggcm.at<float>(row, col);
			feature_key_value["SmallGradientDominance"] += value / pow(col + 1, 2);
			feature_key_value["BigGradientDominance"] += value * pow(col + 1, 2);
		}
	}
	double ggcm_sum = sum(ggcm)[0];
	feature_key_value["SmallGradientDominance"] /= ggcm_sum;  // 除以常数10000，仅仅是为了调节结果大小，好量化;
	feature_key_value["BigGradientDominance"] /= ggcm_sum; // 除以常数10000，仅仅是为了调节结果大小，好量化;
	feature_key_value["GrayAsymmetry"] = 0; // 3. 灰度分布不均匀性
	feature_key_value["GradientAsymmetry"] = 0; // 4. 梯度分布不均匀性
	cv::Mat row_sum_mat(16, 1, CV_32FC1, cv::Scalar(0));
	cv::Mat col_sum_mat(1, 16, CV_32FC1, cv::Scalar(0));
	reduce(ggcm, row_sum_mat, 1, cv::REDUCE_SUM); // 按行求和
	reduce(ggcm, col_sum_mat, 0, cv::REDUCE_SUM); // 按列求和
	for (int i = 0; i < gray_scale; ++i)
	{
		feature_key_value["GrayAsymmetry"] += pow(row_sum_mat.at<float>(i, 0), 2);
		feature_key_value["GradientAsymmetry"] += pow(col_sum_mat.at<float>(0, i), 2);
	}
	feature_key_value["GrayAsymmetry"] /= ggcm_sum;
	feature_key_value["GrayAsymmetry"] /= 100000;
	feature_key_value["GradientAsymmetry"] /= ggcm_sum;
	feature_key_value["GradientAsymmetry"] /= 100000;
	cv::Mat proportion_mat = cv::Mat(gray_scale, gray_scale, CV_32FC1);
	proportion_mat = ggcm / ggcm_sum;
	cv::Mat log_mat = proportion_mat.clone();
	for (int index = 0; index < log_mat.total(); ++index)
	{
		if (log_mat.at<float>(index) == 0)
		{
			log_mat.at<float>(index) = 1;
		}
	}
	log(log_mat, log_mat);
	feature_key_value["Energy"] = 0; // 5. 能量
	feature_key_value["MixedEntropy"] = -sum(proportion_mat.mul(log_mat))[0]; // 6. 混合熵

	feature_key_value["Inertia"] = 0; // 7. 惯性
	feature_key_value["DifferMoment"] = 0; // 8. 逆差矩
	feature_key_value["GrayMean"] = 0; // 9. 灰度均值
	feature_key_value["GradientMean"] = 0; // 10. 梯度均值
	feature_key_value["GrayEntropy"] = 0; // 11. 灰度熵
	feature_key_value["GradientEntropy"] = 0; // 12. 梯度熵
	reduce(proportion_mat, row_sum_mat, 1, cv::REDUCE_SUM); // 按行求和
	reduce(proportion_mat, col_sum_mat, 0, cv::REDUCE_SUM); // 按列求和
	for (int row = 0; row < proportion_mat.rows; ++row)
	{
		for (int col = 0; col < proportion_mat.cols; ++col)
		{
			float value = proportion_mat.at<float>(row, col);
			double square = pow(row - col, 2);
			feature_key_value["Inertia"] += square * value;
			feature_key_value["DifferMoment"] += value / (1 + square);
			feature_key_value["Energy"] += value*value;
		}
		float row_sum_value = row_sum_mat.at<float>(row, 0);
		float col_sum_value = col_sum_mat.at<float>(0, row);

		feature_key_value["GrayMean"] += (row + 1) * row_sum_value;
		feature_key_value["GradientMean"] += (row + 1) * col_sum_value;
		if (row_sum_value != 0)
		{
			feature_key_value["GrayEntropy"] -= row_sum_value * log(row_sum_value);
		}
		if (col_sum_value != 0)
		{
			feature_key_value["GradientEntropy"] -= col_sum_value * log(col_sum_value);
		}
	}
	feature_key_value["GrayStddev"] = 0; // 13. 灰度标准差
	feature_key_value["GradientStddev"] = 0; // 14. 梯度标准差
	for (int index = 0; index < gray_scale; ++index)
	{
		feature_key_value["GrayStddev"] += pow(index - feature_key_value["GrayMean"], 2) * row_sum_mat.at<float>(index, 0);
		feature_key_value["GradientStddev"] += pow(index - feature_key_value["GradientMean"], 2) * row_sum_mat.at<float>(0, index);
	}
	feature_key_value["GrayStddev"] = sqrt(feature_key_value["GrayStddev"]);
	feature_key_value["GradientStddev"] = sqrt(feature_key_value["GradientStddev"]);
	feature_key_value["Correlation"] = 0; // 15. 相关性
	for (int row = 0; row < gray_scale; ++row)
	{
		for (int col = 0; col < gray_scale; ++col)
		{
			feature_key_value["Correlation"] += (row  - feature_key_value["GrayMean"]) * (col  - feature_key_value["GradientMean"]) * proportion_mat.at<float>(row, col);
		}
	}
	double multiple = (feature_key_value["GrayStddev"] * feature_key_value["GradientStddev"]);
	if(multiple!=0) feature_key_value["Correlation"] /= multiple;

	//// 测试用不删
	//for (int index = 0; index < feature_key_value.size(); ++index)
	//{
	//	qDebug() << QString::fromLocal8Bit(feature_key_value.keys().at(index).c_str())+":  "+
	//		QString::number(feature_key_value.values().at(index),10,6);
	//}
}

// 计算两幅图像的偏差指数，参考自：https://blog.csdn.net/nanke_yh/article/details/122002805
double CalcDeviationIndex(cv::Mat& source_image, cv::Mat& target_image)
{
	if (source_image.empty() || target_image.empty() ||
		source_image.cols != target_image.cols ||
		source_image.rows != target_image.rows)
	{
		return 0;
	}
	cv::Mat source_image11 = source_image.clone();
	// 图像差分
	cv::Mat abs_diff_mat;
	absdiff(source_image11, target_image, abs_diff_mat);
	abs_diff_mat.convertTo(abs_diff_mat, CV_32FC1);
	source_image11.convertTo(source_image11, CV_32FC1);
	for (int index = 0; index < source_image11.total(); ++index)
	{
		float value = source_image11.at<float>(index);
		if(value == 0)
		{
			source_image11.at<float>(index) = 1;
			abs_diff_mat.at<float>(index) = 0;
		}
	}
	cv::Mat divide_mat;
	divide(abs_diff_mat, source_image11, divide_mat); // 矩阵相除
	// 计算均值
	double result = mean(divide_mat)[0];
	return result;
}

// 计算两幅图像的扭曲度，参考自：https://blog.csdn.net/nanke_yh/article/details/122002805
double CalcDistortion(cv::Mat& source_image, cv::Mat& target_image)
{
	if (source_image.empty() || target_image.empty() ||
		source_image.cols != target_image.cols ||
		source_image.rows != target_image.rows)
	{
		return 0;
	}
	cv::Mat source_image11 = source_image.clone();
	// 图像差分
	cv::Mat abs_diff_mat;
	absdiff(source_image11, target_image, abs_diff_mat);
	// 计算均值
	double result = mean(abs_diff_mat)[0];
	return result;
}

// 计算两幅图像的协方差，参考自：https://baijiahao.baidu.com/s?id=1713400881254439370&wfr=spider&for=pc
double CalcGradientCovariance(cv::Mat& source_image, cv::Mat& target_image)
{
	if (source_image.empty() || target_image.empty() ||
		source_image.cols != target_image.cols ||
		source_image.rows != target_image.rows)
	{
		return 0;
	}
	cv::Mat gradient_source_image = CalcGradientIntensity(source_image);
	cv::Mat gradient_target_image = CalcGradientIntensity(target_image);
	// 第一幅图像减均值
	double source_mean = mean(gradient_source_image)[0];
	gradient_source_image -= source_mean;
	// 第二幅图像减均值
	double target_mean = mean(gradient_target_image)[0];
	gradient_target_image -= target_mean;
	double covariance = mean(gradient_source_image.mul(gradient_target_image))[0] / 100;  // 除以常数100，仅仅是为了调节结果大小，好量化;
	return covariance;
}

// 计算两幅图像的相关系数
double CalcCorrelationCoefficient(cv::Mat& source_image, cv::Mat& target_image)
{
	if (source_image.empty() || target_image.empty() ||
		source_image.cols != target_image.cols ||
		source_image.rows != target_image.rows)
	{
		return 0;
	}
	//// 保留不删，使用OpenCV的 matchTemplate 函数计算相关系数，速度反而慢，该函数适合批量做
	//cv::Mat result11;
	//matchTemplate(source_image, target_image, result11, cv::TM_CCOEFF_NORMED);
	//uto resulttt = result11.at<float>(0);

	// 第一幅图像减均值
	double source_mean = mean(source_image)[0];
	cv::Mat_<float> source_float;
	source_image.convertTo(source_float, CV_32FC1);
	source_float -= source_mean;
	// 第二幅图像减均值
	double target_mean = mean(target_image)[0];
	cv::Mat_<float> target_float;
	target_image.convertTo(target_float, CV_32FC1);
	target_float -= target_mean;
	// 按公式计算
	double number1 = sum(source_float.mul(target_float))[0];
	double number2 = sum(source_float.mul(source_float))[0];
	double number3 = sum(target_float.mul(target_float))[0];
	if(number2==0|| number3==0)
	{
		return 1;
	}
	double result = number1 / sqrt(number2 * number3);
	return result;
}

// 计算两幅图像的交叉熵
double CalcCrossEntropy(cv::Mat& source_image, cv::Mat& target_image)
{
	if (source_image.empty() || target_image.empty() ||
		source_image.cols != target_image.cols ||
		source_image.rows != target_image.rows)
	{
		return 0;
	}
	int min_value = 0;
	int max_value = 255;
	cv::MatND source_hist = CalcHistogram(source_image, min_value, max_value);
	cv::MatND target_hist = CalcHistogram(target_image, min_value, max_value);
	double cross_entropy = 0.0;
	// 计算两图像的直方图后，按公式计算交叉熵
	for (int i = 0; i < max_value + 1; ++i)
	{
		double source_value = source_hist.at<float>(i);
		double target_value = target_hist.at<float>(i);
		if (0.0 == source_value || 0.0 == target_value)
		{
			continue;
		}
		cross_entropy += source_value * log2(source_value / target_value);
	}
	return cross_entropy;
}

// 计算一幅8位单通道的图像的直方图，注意是单通道，多通道图像会崩溃，暂不支持
// 调用OpenCV自带的函数计算的，返回值是一个矩阵，尺寸是 256 * 1
cv::Mat CalcHistogram(cv::Mat& image, int min_value, int max_value, bool need_normed)
{
	//定义存储直方图的矩阵
	cv::Mat result;
	//计算得到直方图bin的数目，直方图数组的大小
	int hist_size = max_value - min_value + 1;
	//定义直方图每一维的bin的变化范围
	float range[] = {static_cast<float>(min_value), static_cast<float>(max_value + 1)};
	//定义直方图所有bin的变化范围
	const float* ranges = {range};
	//计算直方图，src是要计算直方图的图像，1是要计算直方图的图像数目，0是计算直方图所用的图像的通道序号，从0索引
	//cv::Mat()是要用的掩模，result为输出的直方图，1为输出的直方图的维度，histSize直方图在每一维的变化范围
	//ranges，所有直方图的变化范围（起点和终点）
	calcHist(&image, 1, NULL, cv::Mat(), result, 1, &hist_size, &ranges);
	if (!need_normed) return result;
	return result / image.total(); // 归一化
}

// 计算两幅图像的结构相似度(Structural Similarity)，不分块， 待整理
// 原来的代码和文档是否差一个“2”，文档网上公式也有出入，结构对比 还有没有？
// 网上只看到： https://blog.csdn.net/zm1110918/article/details/98870108
// 结构对比函数为啥没了： https://blog.csdn.net/weixin_39776817/article/details/110595367
double CalcGraySSIM(cv::Mat& source_image, cv::Mat& target_image)
{
	if (source_image.empty() || target_image.empty() ||
		source_image.cols != target_image.cols ||
		source_image.rows != target_image.rows)
	{
		return 0;
	}
	cv::Mat temp_mean, std_dev;
	meanStdDev(source_image, temp_mean, std_dev);
	double source_mean = temp_mean.at<double>(0, 0);
	double source_std_dev = std_dev.at<double>(0, 0);
	meanStdDev(target_image, temp_mean, std_dev);
	double target_mean = temp_mean.at<double>(0, 0);
	double target_std_dev = std_dev.at<double>(0, 0);
	// 计算计算协方差
	cv::Mat source_image11, target_image11;
	source_image.convertTo(source_image11, CV_32FC1);
	target_image.convertTo(target_image11, CV_32FC1);
	source_image11 -= source_mean;
	target_image11 -= target_mean;
	// 对应位置相乘，再求平均
	source_image11 = source_image11.mul(target_image11);
	double covariance = mean(source_image11)[0];
	// 亮度对比
	double lightness = (2 * source_mean * target_mean) / (source_mean * source_mean + target_mean * target_mean);
	// 对比度对比
	double contrast = (2 * source_std_dev * target_std_dev) / (source_std_dev * source_std_dev + target_std_dev *
		target_std_dev);
	// 结构对比
	double structure = covariance / (source_std_dev * target_std_dev);
	double result = lightness * contrast * structure;
	return result;
}

// 计算两幅图像的基于梯度的结构相似度，不分块
double CalcGradientSSIM(cv::Mat& source_image, cv::Mat& target_image)
{
	if (source_image.empty() || target_image.empty() ||
		source_image.cols != target_image.cols ||
		source_image.rows != target_image.rows)
	{
		return 0;
	}
	// 计算第一幅图像的梯度幅值
	cv::Mat source_gradient_x, source_gradient_y;
	CalcSobelGradXY(source_image, source_gradient_x, source_gradient_y);
	cv::Mat source_gradient = source_gradient_x + source_gradient_y;
	// 计算第二幅图像的梯度幅值
	cv::Mat target_gradient_x, target_gradient_y;
	CalcSobelGradXY(target_image, target_gradient_x, target_gradient_y);
	cv::Mat target_gradient = target_gradient_x + target_gradient_y;
	// 相乘相加
	double source_sum = sum(source_gradient.mul(source_gradient)).val[0];
	double target_sum = sum(target_gradient.mul(target_gradient)).val[0];
	double two_sum = sum(source_gradient.mul(target_gradient)).val[0];

	double result = (2 * two_sum) / (source_sum + target_sum);
	return result;
}

// 计算两幅图像的边缘保持指数(Edge Keeping Index)
double CalcEdgeKeepingIndex(cv::Mat& source_image, cv::Mat& target_image)
{
	if (source_image.empty() || target_image.empty() ||
		source_image.cols != target_image.cols ||
		source_image.rows != target_image.rows)
	{
		return 0;
	}
	cv::Mat abs_diff_mat;
	// 公式的左上部分
	cv::Mat target_left = target_image(cv::Rect(0, 0, target_image.cols - 1, target_image.rows));
	cv::Mat target_right = target_image(cv::Rect(1, 0, target_image.cols - 1, target_image.rows));
	absdiff(target_left, target_right, abs_diff_mat);
	double number1 = sum(abs_diff_mat).val[0];
	// 公式的右上部分
	cv::Mat abs_diff_mat2;
	cv::Mat target_top = target_image(cv::Rect(0, 0, target_image.cols, target_image.rows - 1));
	cv::Mat target_bottom = target_image(cv::Rect(0, 1, target_image.cols, target_image.rows - 1));
	absdiff(target_top, target_bottom, abs_diff_mat2);
	double number2 = sum(abs_diff_mat2).val[0];
	// 公式的左下部分
	cv::Mat abs_diff_mat3;
	cv::Mat source_left = source_image(cv::Rect(0, 0, source_image.cols - 1, source_image.rows));
	cv::Mat source_right = source_image(cv::Rect(1, 0, source_image.cols - 1, source_image.rows));
	absdiff(source_left, source_right, abs_diff_mat3);
	double number3 = sum(abs_diff_mat3).val[0];
	// 公式的右下部分
	cv::Mat abs_diff_mat4;
	cv::Mat source_top = source_image(cv::Rect(0, 0, source_image.cols, source_image.rows - 1));
	cv::Mat source_bottom = source_image(cv::Rect(0, 1, source_image.cols, source_image.rows - 1));
	absdiff(source_top, source_bottom, abs_diff_mat4);
	double number4 = sum(abs_diff_mat4).val[0];

	double result = (number1 + number2) / (number3 + number4);
	return result;
}

//联合熵 
double ComEntropy(cv::Mat& matImage1, cv::Mat& matImage2)
{
	if (matImage1.rows != matImage2.rows || matImage1.cols != matImage2.cols)
	{
		return 0;
	}
	double arrPixLevel[256][256] = {0.0};
	for (int m1 = 0, m2 = 0; m1 < matImage1.rows, m2 < matImage2.rows; m1++, m2++)
	{
		const uchar* t1 = matImage1.ptr<uchar>(m1);
		const uchar* t2 = matImage2.ptr<uchar>(m2);
		for (int n1 = 0, n2 = 0; n1 < matImage1.cols, n2 < matImage2.cols; n1++, n2++)
		{
			int nV1 = t1[n1];
			int nV2 = t2[n2];
			arrPixLevel[nV1][nV2] = arrPixLevel[nV1][nV2] + 1;
			arrPixLevel[nV2][nV1] = arrPixLevel[nV1][nV2] + 1;
		}
	}
	int nTotalCnt = 2 * matImage1.rows * matImage1.cols;
	double dComEntropy = 0.0;
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			double fR = arrPixLevel[i][j] / nTotalCnt;
			dComEntropy += (fR * exp(1 - fR));
		}
	}

	//dComEntropy的值不会超过自然底数e,做范围调整
	//dComEntropy = (-dComEntropy + exp(1)) * 100;
	return dComEntropy;
}

double NormMutualInformation(double dEntropy1, double dEntropy2, double dComEntropy)
{
	//互信息
	//因为dEntropy1<=dComEntropy,dEntropy2<=dComEntropy,
	//所以dMI <= dComEntropy, 0 <= dMI / dComEntropy <= 1
	double dMI = dEntropy1 + dEntropy2 - dComEntropy;
	//归一化方法一
	//double dNMI = 2*dMI / (dEntropy1 + dEntropy2);
	//归一化方法二，摘自论文:基于信息熵的景象匹配区选取方法。北理工-张晓晨，付梦印
	double dNMI = dMI / dComEntropy;
	return dNMI;
}

//子图唯一性系数
double SubImageUniqueness(cv::Mat& matSmallImage, cv::Mat& matBigImage, cv::Mat& matBigEntropy)
{
	double dEn1 = CalcGrayEntropy_2(matSmallImage);
	double dTotalNmi = 0.0;
	int nCnt = 0;
	for (int nY = 0; nY <= matBigImage.rows - matSmallImage.rows; nY++)
	{
		for (int nX = 0; nX <= matBigImage.cols - matSmallImage.cols; nX++)
		{
			cv::Mat matTmp(matBigImage, cv::Rect(nX, nY, matSmallImage.cols, matSmallImage.rows));
			double dNmi = NormMutualInformation(dEn1, matBigEntropy.at<float>(nY, nX),
			                                    ComEntropy(matSmallImage, matTmp));
			if (dNmi < 0.5)
			{
				continue;
			}
			dTotalNmi += dNmi;
			nCnt++;
		}
	}
	double dUniqueness = dTotalNmi / nCnt;
	return dUniqueness;
}

// 计算单幅图像的梯度强度图
cv::Mat CalcGradientIntensity(cv::Mat& image)
{
	if(image.empty())
	{
		return image;
	}
	// 计算合成梯度
	cv::Mat matGradientXY;
	cv::Mat sobel_x, sobel_y;
	CalcSobelGradXY(image, sobel_x, sobel_y);
	magnitude(sobel_x, sobel_y, matGradientXY);
	return matGradientXY;
}

// 计算单幅图像的梯度角度图
cv::Mat CalcGradientDirection(cv::Mat& matImage)
{
	// 计算合成梯度
	cv::Mat sobel_x, sobel_y;
	CalcSobelGradXY(matImage, sobel_x, sobel_y, false);
	cv::Mat theta_mat = CalcGradAngleImage(sobel_x, sobel_y);
	return theta_mat;
}

cv::Mat CalcGradAngleImage(cv::Mat grad_x, cv::Mat grad_y)
{
	// 梯度角度图
	cv::Mat theta_mat = cv::Mat::zeros(grad_x.size(), CV_32FC1);
	for (int row = 0; row < grad_x.rows; row++)
	{
		for (int col = 0; col < grad_x.cols; col++)
		{
			float x_value = grad_x.at<float>(row, col);
			float y_value = grad_y.at<float>(row, col);
			// atan2 是4象限反正切，值域是 - pi~pi，具体返回值范围由(x, y) 落入哪个象限决定
			theta_mat.at<float>(row, col) = atan2f(y_value, x_value);
		}
	}
	// 把 theta_mat 的值域，改变到 0-360 度
	theta_mat = (theta_mat + CV_PI) * 180 / CV_PI;
	return theta_mat;
}

// 绘制二维的直方图，测试用，暂时保留，参考自：https://blog.csdn.net/qq_37764129/article/details/81871745
// 输出图像：横坐标是0-255共256个灰度级，纵坐标是把各灰度级的像素个数占比量化到0-255
cv::Mat CalcHistograph(cv::Mat& grayImage)
{
	//定义求直方图的通道数目，从0开始索引
	int channels[] = {0};
	//定义直方图的在每一维上的大小，例如灰度图直方图的横坐标是图像的灰度值，就一维，bin的个数
	//如果直方图图像横坐标bin个数为x，纵坐标bin个数为y，则channels[]={1,2}其直方图应该为三维的，Z轴是每个bin上统计的数目
	const int histSize[] = {256};
	//每一维bin的变化范围
	float range[] = {0, 256};

	//所有bin的变化范围，个数跟channels应该跟channels一致
	const float* ranges[] = {range};

	//定义直方图，这里求的是直方图数据
	cv::Mat hist;
	//opencv中计算直方图的函数，hist大小为256*1，每行存储的统计的该行对应的灰度值的个数
	calcHist(&grayImage, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false); //cv中是cvCalcHist
	//找出直方图统计的个数的最大值，用来作为直方图纵坐标的高
	double maxValue = 0;
	//找矩阵中最大最小值及对应索引的函数
	minMaxLoc(hist, NULL, &maxValue, NULL, NULL);
	//最大值取整
	int rows = cvRound(maxValue);
	//定义直方图图像，直方图纵坐标的高作为行数，列数为256(灰度值的个数)
	//因为是直方图的图像，所以以黑白两色为区分，白色为直方图的图像
	cv::Mat histImage = cv::Mat::zeros(rows, 256, CV_8UC1);

	//直方图图像表示
	for (int i = 0; i < 256; i++)
	{
		//取每个bin的数目
		int temp = static_cast<int>(hist.at<float>(i, 0));
		//如果bin数目为0，则说明图像上没有该灰度值，则整列为黑色
		//如果图像上有该灰度值，则将该列对应个数的像素设为白色
		if (temp)
		{
			//由于图像坐标是以左上角为原点，所以要进行变换，使直方图图像以左下角为坐标原点
			histImage.col(i).rowRange(cv::Range(rows - temp, rows)) = 255;
		}
	}
	//由于直方图图像列高可能很高，因此进行图像对列要进行对应的缩减，使直方图图像更直观
	cv::Mat resizeImage;
	resize(histImage, resizeImage, cv::Size(256, 256));
	return resizeImage;
}

double CalcZeroCrossDensity1(cv::Mat& matImage)
{
	cv::Mat laplacian_mat;
	Laplacian(matImage, laplacian_mat, CV_32FC1); // 拉普拉斯算子提取边缘
	float zero_count = (matImage.total() - countNonZero(laplacian_mat)) * 1.0f;
	return zero_count / matImage.total();
}

// 计算关键点相关特征：关键点响应强度均值、关键点直径均值，  
void CalcKeyPoint(cv::Mat& matImage, std::map<std::string, float>& feature_key_value)
{
	int nFeatures = 15;
	cv::Ptr<cv::SIFT> detector = cv::SIFT::create(nFeatures);
	cv::Ptr<cv::SIFT> detector11 = cv::SIFT::create(222);
	vector<cv::KeyPoint> vctkeypoints;
	detector11->detect(matImage, vctkeypoints, cv::Mat());
	if (vctkeypoints.empty()) return;
	double intensity_sum = 0.0;
	double size_sum = 0.0;
	for (int n = 0; n < vctkeypoints.size(); n++)
	{
		intensity_sum += vctkeypoints[n].response;
		size_sum += vctkeypoints[n].size;
	}
	double intensity_mean = intensity_sum / vctkeypoints.size();
	double size_mean = size_sum / vctkeypoints.size();
	feature_key_value["KeyPointIntensityMean"] = intensity_mean;
	feature_key_value["KeyPointSizeMean"] = size_mean;
}

double CalCornerPointsDensity(cv::Mat& matImage)
{
	cv::Mat matCornerStrength;
	cornerHarris(matImage, matCornerStrength, 2, 3, 0.04);
	double harr_min = 0.0, harr_max = 0.0;
	minMaxLoc(matCornerStrength, &harr_min, &harr_max);
	double dThreshold = max(0.00001, harr_max * 0.5);
	cv::Mat matCorner;
	threshold(matCornerStrength, matCorner, dThreshold, 255, cv::THRESH_BINARY);
	int a = countNonZero(matCorner);
	return a * 100.0 / matCorner.total();
}

void CalcGradCorrelativeSurface(cv::Mat& big_image_grad, cv::Mat& small_image_grad, int real_row, int real_col, std::map<std::string, float>& feature_key_value)
{
	cv::Mat correlative_surface_image;
	matchTemplate(big_image_grad, small_image_grad, correlative_surface_image, cv::TM_CCOEFF_NORMED);
	int roi_width_half = big_image_grad.cols * 0.05; // 相关面曲面上的两个极大值点的最小像素距离横坐标
	int roi_height_half = big_image_grad.rows * 0.05;
	cv::Rect truth_value_region_rect = cv::Rect(real_col - roi_width_half, real_row - roi_height_half,
	                                            roi_width_half * 2, roi_height_half * 2);
	float main_peak_value = 0.5, sub_peak_value = 0.000001, main_sub_ratio = 0.000001; // 把次峰值初始化为一个极小的数字
	int main_peak_col =-1, main_peak_row=-1;
	CalcMainPeakSubPeak(correlative_surface_image, truth_value_region_rect,
		main_peak_value, main_peak_col, main_peak_row, sub_peak_value, main_sub_ratio);
	feature_key_value["MainPeakValue"] = main_peak_value;
	feature_key_value["SubPeakValue"] = sub_peak_value;
	feature_key_value["MainSubPeakRatio"] = main_sub_ratio;
	feature_key_value["MainSubPeakDifference"] = main_peak_value-sub_peak_value;
	feature_key_value["MainPeakSharpness"] = CalcMainPeakSharpness(correlative_surface_image, main_peak_col,  main_peak_row);
	float max_peak = 0, second_peak = 0, max_second_ratio = 0;
	CalcMaxPeakSecondPeak(correlative_surface_image, max_peak, second_peak, max_second_ratio);
	feature_key_value["MaxPeakValue"] = max_peak;
	feature_key_value["SecondPeakValue"] = second_peak;
	feature_key_value["MaxSecondPeakRatio"] = max_second_ratio;

	std::map<std::pair<int, int>, float> peak_list;
	cv::Mat flag_mat = cv::Mat::zeros(correlative_surface_image.rows, correlative_surface_image.cols, CV_8UC1);
	CalcRepeatMode(correlative_surface_image, main_peak_value * 0.8, peak_list, flag_mat, roi_width_half, roi_height_half);
	feature_key_value["RepeatMode"] = peak_list.size();
}

// 计算单幅影像的自相关匹配的相关面特征， 在 image 中裁剪出4个子图并加噪声，然后把分别计算出的特征值求平均
void CalcGradCorrelativeSurface(cv::Mat& image, std::map<std::string, float>& feature_key_value)
{
	double margin_ratio = 0.1; // 要去除的图像边缘占图像长宽的比例，仅仅是为了使匹配位置不在图像边缘上
	int margin_x = image.cols * margin_ratio;
	int margin_y = image.rows * margin_ratio;
	int sub_image_cols = image.cols / 2 - margin_x;
	int sub_image_rows = image.rows / 2 - margin_y;
	// 加噪后，把图像去掉四边缘上10%长宽的像素后，裁剪出4块子图
	cv::Mat noise_image;
	ReduceImageMass(image, noise_image, 3, 0, 0.1);
	cv::Mat original_image_grad = CalcGradientIntensity(image);
	noise_image = CalcGradientIntensity(noise_image);
	cv::Rect center_rect = cv::Rect(original_image_grad.rows / 4, original_image_grad.cols / 4,
	                                original_image_grad.rows / 2, original_image_grad.cols / 2); // 子图的长宽是原始图像长宽的0.5倍
	cv::Rect left_top_rect = cv::Rect(margin_x, margin_y, sub_image_cols, sub_image_rows);
	cv::Rect right_top_rect = cv::Rect(margin_x + sub_image_cols, margin_y, sub_image_cols, sub_image_rows);
	cv::Rect left_bottom_rect = cv::Rect(margin_x, margin_y + sub_image_rows, sub_image_cols, sub_image_rows);
	cv::Rect right_bottom_rect = cv::Rect(margin_x + sub_image_cols, margin_y + sub_image_rows, sub_image_cols,
	                                      sub_image_rows);
	cv::Mat center_image = noise_image(center_rect);
	cv::Mat left_top_image = noise_image(left_top_rect);
	cv::Mat right_top_image = noise_image(right_top_rect);
	cv::Mat left_bottom_image = noise_image(left_bottom_rect);
	cv::Mat right_bottom_image = noise_image(right_bottom_rect);
	// 计算加噪声并裁剪出的5个子图和原图的相关面
	cv::Mat center_result, left_top_result, right_top_result, left_bottom_result, right_bottom_result;
	matchTemplate(original_image_grad, center_image, center_result, cv::TM_CCOEFF_NORMED);
	matchTemplate(original_image_grad, left_top_image, left_top_result, cv::TM_CCOEFF_NORMED);
	matchTemplate(original_image_grad, right_top_image, right_top_result, cv::TM_CCOEFF_NORMED);
	matchTemplate(original_image_grad, left_bottom_image, left_bottom_result, cv::TM_CCOEFF_NORMED);
	matchTemplate(original_image_grad, right_bottom_image, right_bottom_result, cv::TM_CCOEFF_NORMED);
	int roi_width_half =  original_image_grad.cols * 0.05; // 相关面曲面上的两个极大值点的最小像素距离横坐标
	int roi_height_half = original_image_grad.rows * 0.05;
	CorrelativeSurfaceStruct correlative_surface_struct = {};
	// 中心位置
	float main_peak_value00 = 0.5, sub_peak_value00 = 0.000001, main_sub_ratio00 = 0.000001; // 把次峰值初始化为一个极小的数字
	int main_peak_col00 = -1, main_peak_row00 = -1;
	cv::Rect truth_value_region_rect00 = cv::Rect(original_image_grad.rows / 4 - roi_width_half,
	                                              original_image_grad.cols / 4 - roi_height_half, roi_width_half * 2,
	                                              roi_height_half * 2);
	CalcMainPeakSubPeak(center_result, truth_value_region_rect00, main_peak_value00, main_peak_col00, main_peak_row00, sub_peak_value00, main_sub_ratio00);
	correlative_surface_struct.MainOrMaxPeak += main_peak_value00;
	correlative_surface_struct.SubPeakValue += sub_peak_value00;
	correlative_surface_struct.MainSubPeakRatio += main_sub_ratio00;
	correlative_surface_struct.MainSubPeakDifference += main_peak_value00 - sub_peak_value00;
	double sharpness00 = CalcMainPeakSharpness(center_result, main_peak_col00, main_peak_row00);
	correlative_surface_struct.MainPeakSharpness += sharpness00;
	std::map<std::pair<int, int>, float> peak_list00;
	cv::Mat flag_mat00 = cv::Mat::zeros(center_result.rows, center_result.cols,CV_8UC1);
	CalcRepeatMode(center_result, main_peak_value00 * 0.5, peak_list00, flag_mat00, roi_width_half, roi_height_half);
	correlative_surface_struct.RepeatMode += peak_list00.size();
	// 左上位置
	float main_peak_value11 = 0.5, sub_peak_value11 = 0.000001, main_sub_ratio11 = 0.000001; // 把次峰值初始化为一个极小的数字
	int main_peak_col11 = -1, main_peak_row11 = -1;
	cv::Rect truth_value_region_rect11 = cv::Rect(left_top_rect.x - roi_width_half, left_top_rect.y - roi_height_half,
	                                              roi_width_half * 2, roi_height_half * 2);
	CalcMainPeakSubPeak(left_top_result, truth_value_region_rect11, main_peak_value11, main_peak_col11, main_peak_row11, sub_peak_value11, main_sub_ratio11);
	correlative_surface_struct.MainOrMaxPeak += main_peak_value11;
	correlative_surface_struct.SubPeakValue += sub_peak_value11;
	correlative_surface_struct.MainSubPeakRatio += main_sub_ratio11;
	correlative_surface_struct.MainSubPeakDifference += main_peak_value11 - sub_peak_value11;
	double sharpness11 = CalcMainPeakSharpness(left_top_result, main_peak_col11, main_peak_row11);
	correlative_surface_struct.MainPeakSharpness += sharpness11;
	std::map<std::pair<int, int>, float> peak_list11;
	cv::Mat flag_mat11 = cv::Mat::zeros(left_top_result.rows, left_top_result.cols,CV_8UC1);
	CalcRepeatMode(left_top_result, main_peak_value11 * 0.5, peak_list11, flag_mat11, roi_width_half, roi_height_half);
	correlative_surface_struct.RepeatMode += peak_list11.size();
	// 右上位置
	float main_peak_value12 = 0.5, sub_peak_value12 = 0.000001, main_sub_ratio12 = 0.000001; // 把次峰值初始化为一个极小的数字
	int main_peak_col12 = -1, main_peak_row12 = -1;
	cv::Rect truth_value_region_rect12 = cv::Rect(right_top_rect.x - roi_width_half, right_top_rect.y - roi_height_half,
	                                              roi_width_half * 2, roi_height_half * 2);
	CalcMainPeakSubPeak(right_top_result, truth_value_region_rect12, main_peak_value12, main_peak_col12, main_peak_row12, sub_peak_value12, main_sub_ratio12);
	correlative_surface_struct.MainOrMaxPeak += main_peak_value12;
	correlative_surface_struct.SubPeakValue += sub_peak_value12;
	correlative_surface_struct.MainSubPeakRatio += main_sub_ratio12;
	correlative_surface_struct.MainSubPeakDifference += main_peak_value12 - sub_peak_value12;
	double sharpness12 = CalcMainPeakSharpness(right_top_result, main_peak_col12, main_peak_row12);
	correlative_surface_struct.MainPeakSharpness += sharpness12;
	std::map<std::pair<int, int>, float> peak_list12;
	cv::Mat flag_mat12 = cv::Mat::zeros(right_top_result.rows, right_top_result.cols, CV_8UC1);
	CalcRepeatMode(right_top_result, main_peak_value12 * 0.5, peak_list12, flag_mat12, roi_width_half, roi_height_half);
	correlative_surface_struct.RepeatMode += peak_list12.size();
	// 左下位置
	float main_peak_value21 = 0.5, sub_peak_value21 = 0.000001, main_sub_ratio21 = 0.000001; // 把次峰值初始化为一个极小的数字
	int main_peak_col21 = -1, main_peak_row21 = -1;
	cv::Rect truth_value_region_rect21 = cv::Rect(left_bottom_rect.x - roi_width_half,
	                                              left_bottom_rect.y - roi_height_half, roi_width_half * 2,
	                                              roi_height_half * 2);
	CalcMainPeakSubPeak(left_bottom_result, truth_value_region_rect21, main_peak_value21, main_peak_col21, main_peak_row21, sub_peak_value21, main_sub_ratio21);
	correlative_surface_struct.MainOrMaxPeak += main_peak_value21;
	correlative_surface_struct.SubPeakValue += sub_peak_value21;
	correlative_surface_struct.MainSubPeakRatio += main_sub_ratio21;
	correlative_surface_struct.MainSubPeakDifference += main_peak_value21 - sub_peak_value21;
	double sharpness21 = CalcMainPeakSharpness(left_bottom_result, main_peak_col21, main_peak_row21);
	correlative_surface_struct.MainPeakSharpness += sharpness21;
	std::map<std::pair<int, int>, float> peak_list21;
	cv::Mat flag_mat21 = cv::Mat::zeros(left_bottom_result.rows, left_bottom_result.cols, CV_8UC1);
	CalcRepeatMode(left_bottom_result, main_peak_value21 * 0.5, peak_list21, flag_mat21, roi_width_half, roi_height_half);
	correlative_surface_struct.RepeatMode += peak_list21.size();
	// 右下位置
	float main_peak_value22 = 0.5, sub_peak_value22 = 0.000001, main_sub_ratio22 = 0.000001; // 把次峰值初始化为一个极小的数字
	int main_peak_col22 = -1, main_peak_row22 = -1;
	cv::Rect truth_value_region_rect22 = cv::Rect(right_bottom_rect.x - roi_width_half,
	                                              right_bottom_rect.y - roi_height_half, roi_width_half * 2,
	                                              roi_height_half * 2);
	CalcMainPeakSubPeak(right_bottom_result, truth_value_region_rect22, main_peak_value22, main_peak_col22, main_peak_row22, sub_peak_value22, main_sub_ratio22);
	correlative_surface_struct.MainOrMaxPeak += main_peak_value22;
	correlative_surface_struct.SubPeakValue += sub_peak_value22;
	correlative_surface_struct.MainSubPeakRatio += main_sub_ratio22;
	correlative_surface_struct.MainSubPeakDifference += main_peak_value22 - sub_peak_value22;
	double sharpness22 = CalcMainPeakSharpness(right_bottom_result, main_peak_col22, main_peak_row22);
	correlative_surface_struct.MainPeakSharpness += sharpness22;
	std::map<std::pair<int, int>, float> peak_list22;
	cv::Mat flag_mat22 = cv::Mat::zeros(right_bottom_result.rows, right_bottom_result.cols, CV_8UC1);
	CalcRepeatMode(right_bottom_result, main_peak_value22 * 0.5, peak_list22, flag_mat22, roi_width_half, roi_height_half);
	correlative_surface_struct.RepeatMode += peak_list22.size();
	// 取均值
	feature_key_value["MainPeakValue"] = correlative_surface_struct.MainOrMaxPeak / 5;
	feature_key_value["SubPeakValue"] = correlative_surface_struct.SubPeakValue / 5;
	feature_key_value["MainSubPeakRatio"] = correlative_surface_struct.MainSubPeakRatio / 5;
	feature_key_value["MainSubPeakDifference"] = correlative_surface_struct.MainSubPeakDifference / 5;
	feature_key_value["MainPeakSharpness"] = correlative_surface_struct.MainPeakSharpness / 5;
	feature_key_value["RepeatMode"] = correlative_surface_struct.RepeatMode / 5;
	if (!SlidingColRowIndex.isEmpty() && !IsProductionEnvironment)
	{
		TestAndValidate::GetInstance()->VisualizeCorrelativeSurface(center_result); // 输出可视化相关面
	}
}

// 计算主峰值和次峰值，主峰值是相关面上真值的位置的邻域内的最大值，次峰值是相关面上除了这个邻域之外的最大的极大值，是为了反映准则的可靠性，可加到特征指标集中
void CalcMainPeakSubPeak(cv::Mat& correlative_surface_image, cv::Rect truth_value_region_rect,
                         float& main_peak_value, int& main_peak_col, int& main_peak_row, float& sub_peak_value, float& main_sub_ratio)
{
	// 越界处理
	truth_value_region_rect.x= max(0,truth_value_region_rect.x);
	truth_value_region_rect.y = max(0, truth_value_region_rect.y);
	truth_value_region_rect.width = min(correlative_surface_image.cols-truth_value_region_rect.x, truth_value_region_rect.width);
	truth_value_region_rect.height = min(correlative_surface_image.rows-truth_value_region_rect.y, truth_value_region_rect.height);
	cv::Mat truth_value_region_image = correlative_surface_image(cv::Rect(truth_value_region_rect));
	double max_value, min_value;
	cv::Point max_point, min_point;
	minMaxLoc(truth_value_region_image, &min_value, &max_value, &min_point, &max_point);
	main_peak_value = max_value;
	main_peak_col = truth_value_region_rect.x + max_point.x;
	main_peak_row = truth_value_region_rect.y + max_point.y;
	// 以上是计算主峰值，以下是计算次峰值，
	std::map<float, std::pair<int, int> > peak_positio_map = FindPeaks(correlative_surface_image);
	QList<float> list;
	//for (int i = 0; i < peak_positio_map.size(); ++i)
	//{
	//	std::pair<int, int> peak_value = peak_positio_map.at(i);
	//	// 此处需要对float类型的列表倒序排序，未找到办法，前面加一个负号，借助升序排序的办法实现
	//	list.insert(0, peak_value.first); // peak_positio_map 中的Key默认是从小到大排序的，所以会使得list是从大到小排序的
	//}
	// 上面的是C++11的写法，下面的是兼容C++98的写法
	std::map<float, std::pair<int, int> >::iterator iter;
	for (iter = peak_positio_map.begin(); iter != peak_positio_map.end(); ++iter)
	{
		// 此处需要对float类型的列表倒序排序，未找到办法，前面加一个负号，借助升序排序的办法实现
		list.insert(0, iter->first); // peak_positio_map 中的Key默认是从小到大排序的，所以会使得list是从大到小排序的
	}
	//// 保留不删，把《极大值, 极大值位置》的Map的key值，按极大值从大到小排列
	//qSort(list.begin(), list.end(), greater<float>());
	for (int i = 0; i < list.size(); ++i)
	{
		float peak_value = list.at(i);
		std::pair<int, int> position;
		std::map<float, std::pair<int, int>>::iterator temp = peak_positio_map.find(peak_value);
		if(temp!=peak_positio_map.end())
		{
			position = temp->second;
		}
		// 过滤掉，主峰值所在位置的邻域范围
		if (position.first < truth_value_region_rect.x || position.first > truth_value_region_rect.x + truth_value_region_rect.width
			|| position.second < truth_value_region_rect.y || position.second > truth_value_region_rect.y + truth_value_region_rect.height)
		{
			sub_peak_value = peak_value;
			break;
		}
	}
	// 保留不删
	//main_sub_ratio = 1 - sub_peak_value / main_peak_value;
	if(sub_peak_value==0)
	{
		main_sub_ratio = 9999;
	}
	main_sub_ratio = main_peak_value / sub_peak_value;
}

// 在全局位置内找最大值作为相关峰，在相关峰的正负3像素外找最大值作为找次峰，是为了反映整张图的匹配性能，可作为判断是否可匹配的依据之一
void CalcMaxPeakSecondPeak(float* correlation_surface, int cols, int rows, float& max_peak, float& second_peak, float& max_second_ratio)
{
	int max_row = 0, max_col = 0;
	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			float current_value = abs(correlation_surface[row * cols + col]); // 注意相关系数，有可能是负数
			if (current_value > max_peak)
			{
				max_peak = current_value;
				max_row = row;
				max_col = col;
			}
		}
	}
	int near_distance = 3;
	int left = max(0, max_col - near_distance);
	int top = max(0, max_row - near_distance);
	int right = min(cols, max_col + near_distance);
	int bottom = min(rows, max_row + near_distance);
	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			if (col >= left && col <= right && row >= top && row <= bottom) continue; // 去除最大值的N邻域范围
			second_peak = max(second_peak, abs(correlation_surface[row * cols + col])); // 注意相关系数，有可能是负数
		}
	}
	max_second_ratio = 1 - second_peak / max_peak;
}

// 在全局位置内找最大值作为相关峰，在相关峰的正负3像素外找最大值作为找次峰，是为了反映整张图的匹配性能，可作为判断是否可匹配的依据之一
void CalcMaxPeakSecondPeak(cv::Mat correlation_surface, float& max_peak, float& second_peak, float& max_second_ratio)
{
	double max_value = 0, min_value =0;
	cv::Point max_point, min_point;
	minMaxLoc(correlation_surface, &min_value, &max_value, &min_point, &max_point);
	max_peak = max_value;

	int near_distance = 5;
	int left = max(0, max_point.x - near_distance);
	int top = max(0, max_point.y - near_distance);
	int right = min(max_point.x, max_point.x + near_distance);
	int bottom = min(max_point.y, max_point.y + near_distance);
	for (int row = 0; row < correlation_surface.rows; row++)
	{
		for (int col = 0; col < correlation_surface.cols; col++)
		{
			if (col >= left && col <= right && row >= top && row <= bottom) continue; // 去除最大值的N邻域范围
			second_peak = max(second_peak, abs(correlation_surface.at<float>(row,col))); // 注意相关系数，有可能是负数
		}
	}
	// 保留不删
	//max_second_ratio = 1 - second_peak / max_peak;

	if (second_peak == 0)
	{
		max_second_ratio = 9999;
	}
	max_second_ratio = max_peak / second_peak;
}

// 暴力寻找三维曲面的极大值，参考自： https://zhuanlan.zhihu.com/p/474453982
std::map<float, std::pair<int, int> > FindPeaks(cv::Mat image)
{
	std::map<float, std::pair<int, int> > result;
	if (image.cols < 10 && image.rows < 10) // 条件限制：对小图不处理
	{
		return result;
	}
	for (int col = 2; col < image.cols - 2; ++col) // 舍弃四边缘的两个像素
	{
		for (int row = 2; row < image.rows - 2; ++row) // 舍弃四边缘的两个像素
		{
			float center_value = image.at<float>(row, col);
			if (center_value < 0) continue;
			float left_top_value = image.at<float>(row - 1, col - 1);
			float top_value = image.at<float>(row - 1, col);
			float right_top_value = image.at<float>(row - 1, col + 1);
			float right_value = image.at<float>(row, col + 1);
			float right_bottom_value = image.at<float>(row + 1, col + 1);
			float bottom_value = image.at<float>(row + 1, col);
			float left_bottom_value = image.at<float>(row + 1, col - 1);
			float left_value = image.at<float>(row, col - 1);
			// 某个像素值数字均大于他的8邻域的像素值，则认为是极大值
			if (center_value > left_top_value && center_value > top_value && center_value > right_top_value && center_value > right_value
				&& center_value > right_bottom_value && center_value > bottom_value && center_value > left_bottom_value && center_value > left_value)
			{
				result[center_value] = std::pair<int, int>(col, row); // result 的 Key 可以重复，计算主次峰比时，只关心大小，不关心位置
			}
		}
	}
	return result;
}

// 计算相关面的重复模式，递归查找相关面中，大于某一阈值的极大值点，计算方式可改为：调用 FindPeaks 函数，统计峰值的方式，比单纯的比较大小更合理
void CalcRepeatMode(cv::Mat& correlative_surface, float peak_value_threshold, std::map<std::pair<int, int>, float>& peak_list, cv::Mat& flag_mat, int roi_width_half, int roi_height_half)
{
	int peak_row = -1, peak_col = -1;
	float peak_value = -1;
	// 遍历相关面
	for (int row = 0; row < correlative_surface.rows; row++)
	{
		for (int col = 0; col < correlative_surface.cols; col++)
		{
			// 跳过范围： 已查找到的极大值位置的邻域范围
			if (1 == flag_mat.at<uchar>(row, col))
			{
				continue;
			}
			float current_value = correlative_surface.at<float>(row, col);
			if (current_value > peak_value)
			{
				peak_value = current_value;
				peak_row = row;
				peak_col = col;
			}
		}
	}
	if (peak_value > peak_value_threshold)
	{
		// 峰值位置外扩小图的长宽大小， 下次查询跳过该范围
		int left = max(0, peak_col - roi_width_half);
		int right = min(correlative_surface.cols, peak_col + roi_width_half);
		int top = max(0, peak_row - roi_height_half);
		int bottom = min(correlative_surface.rows, peak_row + roi_height_half);
		for (int row = top; row < bottom; ++row)
		{
			for (int col = left; col < right; ++col)
			{
				flag_mat.at<uchar>(row, col) = 1;
			}
		}
		//X坐标、Y坐标、峰值
		std::pair<int, int> temp_pair(peak_row, peak_col);
		peak_list[temp_pair] = peak_value;
		CalcRepeatMode(correlative_surface, peak_value_threshold, peak_list, flag_mat, roi_width_half, roi_height_half);
	}
}

// 计算相关面的主峰尖锐度：主峰位置的第1、2、3像素距离的像素均值，除以第4、5、6像素距离的像素均值，实际计算是用的矩形，而不是圆形范围
double CalcMainPeakSharpness(cv::Mat& correlative_surface, int col_index, int row_index)
{
	// 当主峰值位于边缘(距离边缘小于6个像素)时，强行计算会使得，该特征值表示的含义不准确
	if (col_index < 6 || row_index < 6 || col_index + 6 > correlative_surface.cols - 1 || row_index + 6 > correlative_surface.rows - 1)
	{
		return 0;
	}
	cv::Mat roi11 = correlative_surface(cv::Rect(col_index - 3, row_index - 3, 7, 7));
	cv::Mat roi22 = correlative_surface(cv::Rect(col_index - 6, row_index - 6, 13, 13));
	double mean11 = mean(cv::abs(roi11))[0];// 注意计算出的相关面矩阵中，有负数，取绝对值
	double mean22 = mean(cv::abs(roi22))[0];// 注意计算出的相关面矩阵中，有负数，取绝对值
	if (mean22 == 0) return -9999;
	return mean11 / mean22;
	// 按主峰尖锐度的定义，应按以下方法计算，为方便量化，使用以上方法：主峰的小邻域内的像素均值 除以 主峰的大邻域内的像素均值
	//uto mean_circle = mean(cv::abs(roi11)); 
	//uto mean_loop = (sum(cv::abs(roi22)) - mean_circle * 7 * 7) / (13 * 13 - 7 * 7);
	//// 除数是0时，返回
	//if (mean_circle[0] == 0) return -9999;
	//return 1 - mean_loop[0] / mean_circle[0];
}

void ReduceImageMass(cv::Mat big_image_mat, cv::Mat& noise_image, double blur_size, double sigma, double noise_density)
{
	// 对比度拉伸
	cv::Mat stretched_image;
	StretchContrast(big_image_mat, stretched_image, StretchDensityThreshold);
	// 高斯模糊
	cv::Mat blur_image;
	GaussianBlur(stretched_image, blur_image, cv::Size(blur_size, blur_size), sigma, sigma); // 高斯核宽高尺寸一般为奇数，否则可能会崩溃
	// 高斯乘性随机噪声
	noise_image = AddGaussMultipleNoise(blur_image, noise_density);
	// 缩放和旋转噪声
	cv::Point2f center(noise_image.cols / 2, noise_image.rows / 2);
	cv::Mat rotate_mat = getRotationMatrix2D(center, RotateAngle, CompressionRatio);
	cv::Mat rotate_resize_image;
	warpAffine(noise_image, noise_image, rotate_mat, noise_image.size());
}

// 拉伸对比度，解决图像偏暗或偏白的问题，对 input_image 按直方图 source_hist 收缩 stretch_density_threshold 比例
void StretchContrast(cv::Mat& input_image, cv::Mat& output_image, double stretch_density_threshold)
{
	if (0 == stretch_density_threshold) // 若拉伸强度为0，则不拉伸
	{
		output_image = input_image;
		return;
	}
	int min_value = 0;
	int max_value = 255;
	cv::MatND source_hist = CalcHistogram(input_image, min_value, max_value);
	double commulative_value11 = 0.0;
	double commulative_value22 = 0.0;
	int low_pixel_value = -1, high_pixel_value = -1;
	// 找到最大最小像素值阈值， source_hist 是1行256列的矩阵，每个元素存放该像素值所占的比例，有的元素值可能为0
	for (uchar i = 0; i < 256; ++i)
	{
		if (-1 != low_pixel_value && -1 != high_pixel_value)
		{
			break;
		}
		if (commulative_value11 < stretch_density_threshold)
		{
			commulative_value11 += source_hist.at<float>(i);
			if (commulative_value11 > stretch_density_threshold)
			{
				low_pixel_value = i;
			}
		}
		if (commulative_value22 < stretch_density_threshold)
		{
			commulative_value22 += source_hist.at<float>(255 - i);
			if (commulative_value22 > stretch_density_threshold)
			{
				high_pixel_value = 255 - i;
			}
		}
	}
	if (high_pixel_value == low_pixel_value)
	{
		output_image = input_image;
		return;
	}
	output_image = cv::Mat::zeros(input_image.rows, input_image.cols, CV_8UC1);
	for (int i = 0; i < input_image.rows; i++)
	{
		for (int j = 0; j < input_image.cols; j++)
		{
			uchar pixel_value = input_image.at<uchar>(i, j);
			if (pixel_value < low_pixel_value)
			{
				// 大于阈值的，赋值为0
				pixel_value = 0;
			}
			else if (pixel_value > high_pixel_value)
			{
				// 大于阈值的，赋值为255
				pixel_value = 255;
			}
			else
			{
				// 其余的拉伸到0-255之间
				pixel_value = (pixel_value - low_pixel_value) * 255 / (high_pixel_value - low_pixel_value);
			}
			output_image.at<uchar>(i, j) = pixel_value;
		}
	}
}
 
// 添加高斯乘性噪声，模拟斑点噪声，相干斑噪声，
cv::Mat AddGaussMultipleNoise(cv::Mat& image, double noise_stddev)
{
	if (image.empty() || noise_stddev == 0) return image;
	cv::Mat noise_image = image.clone();
	noise_image.convertTo(noise_image, CV_32FC1);
	cv::Mat matGaussNoise(image.rows, image.cols, CV_32FC1);
	randn(matGaussNoise, cv::Scalar::all(0), cv::Scalar::all(noise_stddev));
	cv::Mat matTmp = matGaussNoise.mul(noise_image);
	noise_image += matTmp;
	noise_image.convertTo(noise_image, CV_8UC1);
	return noise_image;
}

// 添加傅里叶频域噪声，暂不删除
cv::Mat AddFourierNoise(cv::Mat& image)
{
	if (image.empty()) return image;
	cv::Mat fourier = ConvertToFourier(image);
	// 对傅里叶变换结果 fourier 加乘性噪声
	cv::Mat random_mat(fourier.rows, fourier.cols, CV_32FC2);
	// TODO 添加到配置文件
	randn(random_mat, cv::Scalar::all(0), cv::Scalar::all(0.2));
	fourier += random_mat.mul(fourier);
	// 从频域反变换到空域
	cv::Mat noise_image;
	idft(fourier, noise_image, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
	noise_image.convertTo(noise_image, CV_8UC1);
	noise_image = noise_image(cv::Rect(0, 0, image.cols, image.rows));
	return noise_image;
}

// 在全局位置内找最大值作为相关峰，在相关峰的正负3像素外找最大值作为找次峰，是为了反映整张图的匹配性能，可作为判断是否可匹配的依据之一
void CalcMaxPeakSubPeak(float* correlation_surface, int cols, int rows, float& max_peak, float& sub_peak, float& max_sub_ratio)
{
	int max_row = 0, max_col = 0;
	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			float current_value = abs(correlation_surface[row * cols + col]); // 注意相关系数，有可能是负数
			if (current_value > max_peak)
			{
				max_peak = current_value;
				max_row = row;
				max_col = col;
			}
		}
	}
	int near_distance = 3;
	int left = max(0, max_col - near_distance);
	int top = max(0, max_row - near_distance);
	int right = min(cols, max_col + near_distance);
	int bottom = min(rows, max_row + near_distance);
	for (int row = 0; row < rows; row++)
	{
		for (int col = 0; col < cols; col++)
		{
			if (col >= left && col <= right && row >= top && row <= bottom) continue; // 去除最大值的N邻域范围
			sub_peak = max(sub_peak, abs(correlation_surface[row * cols + col])); // 注意相关系数，有可能是负数
		}
	}
	// 保留不删
	//max_sub_ratio = 1 - sub_peak / max_peak;
	if (sub_peak == 0)
	{
		max_sub_ratio = 9999;
	}
	max_sub_ratio = max_peak / sub_peak;

}

// 计算LBP相关特征：LBP图像均值、梯度均值、边缘均值，
// 参考自： https://blog.csdn.net/wsp_1138886114/article/details/119667798
// 参考自： https://blog.csdn.net/weixin_44651073/article/details/128022306
void CalcLbp(cv::Mat& source_image, map<string, double>& key_value)
{
	int radius = 3;
	int neighbors = 8;
	// 把原图长宽均缩小2倍
	cv::Mat image;
	cv::resize(source_image, image, cv::Size(source_image.cols / 2, source_image.rows / 2));
	GaussianBlur(image, image, cv::Size(3, 3), 0, 0, cv::BORDER_REFLECT101);
	int height = image.rows;
	int width = image.cols;
	int offset = radius * 2;
	cv::Mat elbp_image = cv::Mat::zeros(height - offset, width - offset, CV_8UC1);
	for (size_t n = 0; n < neighbors; n++) {
		float tmp = 2.0 * CV_PI * n / static_cast<float>(neighbors);
		float x = static_cast<float>(-radius) * sin(tmp);
		float y = static_cast<float>(radius) * cos(tmp);

		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));

		float tx = x - fx;
		float ty = y - fy;

		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx * (1 - ty);
		float w3 = (1 - tx) * ty;
		float w4 = tx * ty;

		for (int i = radius; i < height - radius; i++) {
			for (int j = radius; j < width - radius; j++) {
				float t = w1 * image.at<uchar>(i + fy, j + fx)
					+ w2 * image.at<uchar>(i + fy, j + cx)
					+ w3 * image.at<uchar>(i + cy, j + fx)
					+ w4 * image.at<uchar>(i + cy, j + cx);
				elbp_image.at<uchar>(i - radius, j - radius) += (
					(t > image.at<uchar>(i, j)) &&
					(std::abs(t - image.at<uchar>(i, j)) >
						std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
	key_value["LbpImageMean"] = cv::mean(elbp_image)[0];
	cv::Mat gradient = CalcGradientIntensity(elbp_image);
	key_value["LbpGradientMean"]=cv::mean(gradient)[0];
	cv::Mat sobelX;
	cv::Mat sobelY;
	cv::Sobel(elbp_image, sobelX, CV_16S, 1, 0);
	cv::Sobel(elbp_image, sobelY, CV_16S, 0, 1);
	cv::Mat edge = abs(sobelX) + abs(sobelY); // L1范数
	key_value["LbpEdgeMean"] =  cv::mean(edge)[0];
}
