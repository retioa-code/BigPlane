# pragma execution_character_set("utf-8")
#include "MatchAlgorithm.h"
#include "FeatureExtract.h"
#include "OpenCVHelper.h"
using namespace cv;
//using namespace cv::xfeatures2d;


cv::Point MatchAlgorithm::CrossCoGradXY(cv::Mat& big_image, cv::Mat& small_image)
{
	Mat sobelXMat;
	Mat sobelYMat;
	Sobel(big_image, sobelXMat, CV_32F, 1, 0, 3, 1, 0, BORDER_REFLECT);
	sobelXMat = abs(sobelXMat);
	Sobel(big_image, sobelYMat, CV_32F, 0, 1, 3, 1, 0, BORDER_REFLECT);
	sobelYMat = abs(sobelYMat);

	Mat tempSobelXMat;
	Mat tempSobelYMat;
	Sobel(small_image, tempSobelXMat, CV_32F, 1, 0, 3, 1, 0, BORDER_REFLECT);
	tempSobelXMat = abs(tempSobelXMat);
	Sobel(small_image, tempSobelYMat, CV_32F, 0, 1, 3, 1, 0, BORDER_REFLECT);
	tempSobelYMat = abs(tempSobelYMat);

	Mat crossXMat;
	Mat crossYMat;
	matchTemplate(sobelXMat, tempSobelXMat, crossXMat, cv::TM_CCOEFF_NORMED);
	matchTemplate(sobelYMat, tempSobelYMat, crossYMat, cv::TM_CCOEFF_NORMED);
	Mat resultMat = (crossXMat + crossYMat) / 2;

	double max_value, min_value;
	cv::Point max_point, min_point;
	minMaxLoc(resultMat, &min_value, &max_value, &min_point, &max_point);
	return max_point;
}

cv::Point MatchAlgorithm::MatchByOpenCVTemplate(cv::Mat& big_image, cv::Mat& small_image)
{
	cv::Mat result;
	matchTemplate(big_image, small_image, result, cv::TM_CCOEFF_NORMED);
	double max_value, min_value;
	cv::Point max_point, min_point;
	minMaxLoc(result, &min_value, &max_value, &min_point, &max_point);
	cv::Point point(max_point.x, max_point.y);
	return point;
}

cv::Point MatchAlgorithm::MatchByGradIntensity(cv::Mat& big_grad_intensity, cv::Mat& small_image)
{
	cv::Point point(-1, -1);
	if (big_grad_intensity.empty() || small_image.empty()) return point;
	cv::Mat small_grad_intensity = CalcGradientIntensity(small_image);
	point = MatchByOpenCVTemplate(big_grad_intensity, small_grad_intensity);
	return point;
}

cv::Point MatchAlgorithm::MatchByGradDirection(cv::Mat& big_grad_direction, cv::Mat& small_image)
{
	cv::Point point(-1, -1);
	if (big_grad_direction.empty() || small_image.empty()) return point;
	cv::Mat small_grad_direction = CalcGradientDirection(small_image);
	point = MatchByOpenCVTemplate(big_grad_direction, small_grad_direction);
	return point;
}

double MatchAlgorithm::CalcCorelativeCoefficient(QList<double> list1,QList<double> list2)
{
	// 计算均值

	double mean1 = 0.0, mean2 = 0.0;
	for (int index = 0; index < list1.size(); ++index)
	{
		mean1 += list1.at(index);
	}
	mean1 /= list1.size();
	for (int index = 0; index < 7; ++index)
	{
		mean2 += list2.at(index);
	}
	mean2 /= list2.size();
	// 按公式计算相关系数
	double fenmu1 = 0.0, fenmu2 = 0.0, fenzi = 0.0;
	for (int index = 0; index < list2.size(); ++index)
	{
		fenmu1 += (list1.at(index) - mean1) * (list1.at(index) - mean1);
		fenmu2 += (list2.at(index) - mean2) * (list2.at(index) - mean2);
		fenzi += (list1.at(index) - mean1) * (list2.at(index) - mean2);
	}
	double coefficient = fenzi / sqrt(fenmu1 * fenmu2);
	return coefficient;
}

// Hog 特征向量是把原始图像看做列主序计算而来，hog_cols_by_block 表示把Hog特征向量，转换为cv::Mat后，横向上有多少个 block
cv::Mat MatchAlgorithm::CalculateHogFeature(cv::Mat& image)
{
	// 缩放原图，到 BlockStride 的整数倍
	int new_image_cols = round(image.cols * 1.0 / BlockStride.width) * BlockStride.width;
	int new_image_rows = round(image.rows * 1.0 / BlockStride.height) * BlockStride.height;
	cv::Size new_image_size(new_image_cols, new_image_rows);
	cv::Mat new_image;
	resize(image, new_image, new_image_size);
	cv::HOGDescriptor hog_descriptor(new_image_size, BlockSize, BlockStride, CellSize, Bins);
	vector<float> hog_feature;
	hog_descriptor.compute(new_image, hog_feature);
	int hog_cols_by_block = (new_image_cols - BlockSize.width) / BlockStride.width + 1;
	int hog_rows_by_block = (new_image_rows - BlockSize.height) / BlockStride.height + 1;
	// 把 hog_feature 转换为 cv::Mat 的同时，转换为行主序
	int hog_rows = hog_rows_by_block * FeatureDimensionOfBlock;
	cv::Mat hog_feature_mat(hog_cols_by_block, hog_rows, CV_32FC1);
	int temp_number = 0;
	for (int row_index = 0; row_index < hog_feature_mat.total(); ++row_index)
	{
		hog_feature_mat.at<float>(temp_number) = hog_feature[temp_number++];
	}
	return hog_feature_mat;
}

cv::Point MatchAlgorithm::MatchByHogAndTemplate(cv::Mat& big_image, cv::Mat& big_hog_mat, cv::Mat& small_image, cv::Mat small_hog_mat)
{
	// 计算小图的特征向量在大图的特征向量上滑动的次数
	int slide_cols = (big_hog_mat.cols - small_hog_mat.cols) / FeatureDimensionOfBlock + 1;
	int slide_rows = big_hog_mat.rows - small_hog_mat.rows + 1;
	cv::Point match_position(-1, -1);
	double hog_feature_distance = DBL_MAX;
	//两层循环，遍历大图，裁剪子图，和小图比对，取两个特征向量的欧式距离最小的位置，为Hog算法输出的匹配位置
	for (int row = 0; row < slide_rows; ++row)
	{
		for (int col = 0; col < slide_cols; ++col)
		{
			int left = col * FeatureDimensionOfBlock;
			int top = row;
			int sub_width = small_hog_mat.cols;
			int sub_height = small_hog_mat.rows;
			Mat sub_hog_mat = big_hog_mat(cv::Rect(left, top, sub_width, sub_height));
			double temp_distance = norm(sub_hog_mat - small_hog_mat, cv::NORM_L2);
			if (hog_feature_distance < temp_distance) continue;
			hog_feature_distance = temp_distance;
			match_position = cv::Size(row, col);
		}
	}
	match_position.x *= BlockStride.width;
	match_position.y *= BlockStride.height;
	// 以上是按CellSize大小滑动的Hog匹配，下面是把Hog匹配结果位置，外扩CellSize大小，裁剪子图，逐像素滑动的模板匹配。
	int hog_result_left = max(0, match_position.x - CellSize.width);
	int hog_result_top = max(0, match_position.y - CellSize.height);
	int temp_width = CellSize.width * 2 + small_image.cols;
	int temp_height = CellSize.height * 2 + small_image.rows;
	temp_width = min(temp_width, big_image.cols - hog_result_left);
	temp_height = min(temp_height, big_image.rows - hog_result_top);
	Mat hog_result_image = big_image(cv::Rect(hog_result_left, hog_result_top, temp_width, temp_height));

	// 暂时不删，针对异源图像匹配，尽量不使用灰度，参考“_巡航导弹景象匹配区选取关键技术研究”正文第8页
	// 而是对梯度方向图，使用模板匹配。具体时间指标和提高了多少精度，缺少测试验证
	//QDateTime begin = QDateTime::currentDateTime();
	//cv::Mat small_grad_x, small_grad_y, small_grad_direction;
	//Sobel(small_image, small_grad_x, CV_32FC1, 1, 0); //求梯度强度
	//Sobel(small_image, small_grad_y, CV_32FC1, 0, 1);
	//phase(small_grad_x, small_grad_y, small_grad_direction, true); //求梯度方向的角度值
	//cv::Mat big_grad_x, big_grad_y, big_grad_direction;
	//Sobel(hog_result_image, big_grad_x, CV_32FC1, 1, 0); //求梯度强度
	//Sobel(hog_result_image, big_grad_y, CV_32FC1, 0, 1);
	//phase(big_grad_x, big_grad_y, big_grad_direction, true); //求梯度方向的角度值
	//big_grad_direction.convertTo(big_grad_direction, CV_8UC1);
	//small_grad_direction.convertTo(small_grad_direction, CV_8UC1);
	//QDateTime middle = QDateTime::currentDateTime();
	//uto xxx = middle.msecsTo(begin);
	//cv::Mat result;
	//matchTemplate(big_grad_direction, small_grad_direction, result, TM_CCOEFF_NORMED);

	cv::Mat result;
	cv::Mat matBigGradient = CalcGradientIntensity(hog_result_image);
	cv::Mat matSmallGradient = CalcGradientIntensity(small_image);
	matchTemplate(matBigGradient, matSmallGradient, result, cv::TM_CCOEFF_NORMED);
	double max_value, min_value;
	cv::Point max_point, min_point;
	minMaxLoc(result, &min_value, &max_value, &min_point, &max_point);
	cv::Point point(hog_result_left + max_point.x, hog_result_top + max_point.y);
	return point;
}

#ifdef PlatformIsWindows
// 基于Hu矩相似度的图像匹配，匹配精度一般，耗时太长，是HOG的100多倍
cv::Point MatchAlgorithm::MatchByHuMoment(cv::Mat& big_image, cv::Mat& small_image)
{
	int col_count = big_image.cols - small_image.cols + 1;
	int row_count = big_image.rows - small_image.cols + 1;
	// 计算小图的Hu矩
	map<string, double> small_feature_key_value;
	CalcHuMoment(small_image, small_feature_key_value);
	cv::Point match_point;
	double relative_coefficient = 0.0;
	for (int row = 0; row < row_count; ++row)
	{
		for (int col = 0; col < col_count; ++col)
		{
			// 计算大图的子图的Hu矩
			map<string, double> big_feature_key_value;
			cv::Mat big_temp = big_image(cv::Rect(col, row, small_image.cols, small_image.rows));
			CalcHuMoment(big_temp, big_feature_key_value);
			QList<double> big_list;
			QList<double> small_list;
			// 计算Hu矩的均值
			for (int index = 0; index < 7; ++index)
			{
				big_list.append(big_feature_key_value["HuMonent0"]);
				big_list.append(big_feature_key_value["HuMonent1"]);
				big_list.append(big_feature_key_value["HuMonent2"]);
				big_list.append(big_feature_key_value["HuMonent3"]);
				big_list.append(big_feature_key_value["HuMonent4"]);
				big_list.append(big_feature_key_value["HuMonent5"]);
				big_list.append(big_feature_key_value["HuMonent6"]);
				small_list.append(small_feature_key_value["HuMonent0"]);
				small_list.append(small_feature_key_value["HuMonent1"]);
				small_list.append(small_feature_key_value["HuMonent2"]);
				small_list.append(small_feature_key_value["HuMonent3"]);
				small_list.append(small_feature_key_value["HuMonent4"]);
				small_list.append(small_feature_key_value["HuMonent5"]);
				small_list.append(small_feature_key_value["HuMonent6"]);
			}
			double coefficient = CalcCorelativeCoefficient(big_list, small_list);
			// 计算基于Hu矩的相关系数，值越大则越认为是真实的匹配点，参考自论文：基于多波束测深系统的海底地形匹配导航技术研究_岳增阳
			if (coefficient > relative_coefficient)
			{
				relative_coefficient = coefficient;
				match_point = cv::Point(col, row);
			}
		}
	}
	return match_point;
}

cv::Point2f MatchAlgorithm::MatchBySift(cv::Mat image_scene, cv::Mat image_object)
{
	// 2) 检测特征并计算描述子
	Ptr<SIFT> sift = SIFT::create();
	std::vector<KeyPoint>  kps1, kps2;
	Mat desc1, desc2;
	sift->detectAndCompute(image_object, Mat(), kps1, desc1);
	sift->detectAndCompute(image_scene, Mat(), kps2, desc2);
	// 3) FLANN 匹配
	Ptr<FlannBasedMatcher> knnmatcher = FlannBasedMatcher::create();
	std::vector<std::vector<DMatch> > matches;
	knnmatcher->knnMatch(desc1, desc2, matches, 2);
	// 4) filter matches using Lowe's distance ratio test
	std::vector<DMatch> good_matches;
	const float kRatioThresh = 0.7f;
	for (size_t i = 0; i < matches.size(); i++)
	{
		if (matches[i][0].distance < kRatioThresh * matches[i][1].distance)
		{
			good_matches.push_back(matches[i][0]);
		}
	}
	// 5) draw and show matches
	Mat img_matches;
	drawMatches(image_object, kps1, image_scene, kps2, good_matches, img_matches);
	//imshow("Good Matches", img_matches);
	//waitKey(0);
	// Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		// Get the keypoints from the good matches
		obj.push_back(kps1[good_matches[i].queryIdx].pt);
		scene.push_back(kps2[good_matches[i].trainIdx].pt);
	}
	// estimate H
	// findHomography需要4个或更多个点：obj和scene的点的个数大于等于4
	if (obj.size() < 4 || scene.size() < 4)
	{
		return cv::Point2f(0, 0);
	}
	Mat H = findHomography(obj, scene, RANSAC);

	// get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point2f(0, 0);
	obj_corners[1] = Point2f((float)image_object.cols, 0);
	obj_corners[2] = Point2f((float)image_object.cols, (float)image_object.rows);
	obj_corners[3] = Point2f(0, (float)image_object.rows);

	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H);

	//// draw lines between the corners (the mapped object in the scene - image_2 )
	//line(img_matches, scene_corners[0] + Point2f((float)image_object.cols, 0), scene_corners[1] + Point2f((float)image_object.cols, 0), Scalar(0, 255, 0));
	//line(img_matches, scene_corners[1] + Point2f((float)image_object.cols, 0), scene_corners[2] + Point2f((float)image_object.cols, 0), Scalar(0, 255, 0));
	//line(img_matches, scene_corners[2] + Point2f((float)image_object.cols, 0), scene_corners[3] + Point2f((float)image_object.cols, 0), Scalar(0, 255, 0));
	//line(img_matches, scene_corners[3] + Point2f((float)image_object.cols, 0), scene_corners[0] + Point2f((float)image_object.cols, 0), Scalar(0, 255, 0));

	//// show detected matches
	//imshow("Object detection", img_matches);
	//waitKey(0);
	return scene_corners[0];
}

cv::Point2f MatchAlgorithm::MatchBySurf(cv::Mat image_scene, cv::Mat image_object)
{
	vector<Point2f> scene_corners(4);
	return scene_corners[0];
	////检测特征点
	//const int minHessian = 400;
	//Ptr<SURF> detector = SURF::create(minHessian);
	//vector<KeyPoint> keypoints_object, keypoints_scene;
	//detector->detect(image_object, keypoints_object);
	//detector->detect(image_scene, keypoints_scene);
	////计算特征点描述子
	////SurfDescriptorExtractor extractor;
	//Mat descriptors_object, descriptors_scene;
	//detector->compute(image_object, keypoints_object, descriptors_object);
	//detector->compute(image_scene, keypoints_scene, descriptors_scene);
	////使用FLANN进行特征点匹配
	//FlannBasedMatcher matcher;
	//vector<DMatch> matches;
	//matcher.match(descriptors_object, descriptors_scene, matches);
	////计算匹配点之间的最大和最小距离
	//double max_dist = 0;
	//double min_dist = 100;
	//for (int i = 0; i < descriptors_object.rows; i++)
	//{
	//	double dist = matches[i].distance;
	//	if (dist < min_dist)
	//	{
	//		min_dist = dist;
	//	}
	//	else if (dist > max_dist)
	//	{
	//		max_dist = dist;
	//	}
	//}
	////绘制好的匹配点
	//vector<DMatch> good_matches;
	//for (int i = 0; i < descriptors_object.rows; i++)
	//{
	//	if (matches[i].distance < 2 * min_dist)
	//	{
	//		good_matches.push_back(matches[i]);
	//	}
	//}
	//Mat image_matches;
	//drawMatches(image_object, keypoints_object, image_scene, keypoints_scene, good_matches, image_matches,
	//	Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	////定位好的匹配点
	//vector<Point2f> obj;
	//vector<Point2f> scene;
	//for (int i = 0; i < good_matches.size(); i++)
	//{
	//	//DMathch类型中queryIdx是指match中第一个数组的索引,keyPoint类型中pt指的是当前点坐标
	//	obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
	//	scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	//}
	//// findHomography需要4个或更多个点：obj和scene的点的个数大于等于4
	//if (obj.size() < 4 || scene.size() < 4)
	//{
	//	return cv::Point2f(0, 0);
	//}
	//Mat H = findHomography(cv::Mat(obj), cv::Mat(scene), RANSAC);
	//vector<Point2f> obj_corners(4), scene_corners(4);
	//obj_corners[0] = Point(0, 0);
	//obj_corners[1] = Point(image_object.cols, 0);
	//obj_corners[2] = Point(image_object.cols, image_object.rows);
	//obj_corners[3] = Point(0, image_object.rows);
	//perspectiveTransform(obj_corners, scene_corners, H);
	//////绘制角点之间的直线，保留不删
	////line(image_matches, scene_corners[0] + Point2f(image_object.cols, 0),
	////	scene_corners[1] + Point2f(image_object.cols, 0), Scalar(0, 0, 255), 2);
	////line(image_matches, scene_corners[1] + Point2f(image_object.cols, 0),
	////	scene_corners[2] + Point2f(image_object.cols, 0), Scalar(0, 0, 255), 2);
	////line(image_matches, scene_corners[2] + Point2f(image_object.cols, 0),
	////	scene_corners[3] + Point2f(image_object.cols, 0), Scalar(0, 0, 255), 2);
	////line(image_matches, scene_corners[3] + Point2f(image_object.cols, 0),
	////	scene_corners[0] + Point2f(image_object.cols, 0), Scalar(0, 0, 255), 2);
	////namedWindow("匹配图像", WINDOW_AUTOSIZE);
	////imshow("匹配图像", image_matches);
	////waitKey(0);
	//return scene_corners[0];

}

// 参考自：http://www.manongjc.com/detail/50-ymambgrgwvtubcu.html
cv::Point2f MatchAlgorithm::MatchByOrb(cv::Mat image_scene, cv::Mat image_object)
{
	vector<KeyPoint> keypoints_obj, keypoints_sence;

	Mat descriptors_box, descriptors_sence;

	Ptr<ORB> detector = ORB::create();

	detector->detectAndCompute(image_scene, Mat(), keypoints_sence, descriptors_sence);

	detector->detectAndCompute(image_object, Mat(), keypoints_obj, descriptors_box);


	vector<DMatch> matches;

	// 初始化flann匹配

	// Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create(); // default is bad, using local sensitive hash(LSH)

	Ptr<DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>(makePtr<flann::LshIndexParams>(12, 20, 2));

	matcher->match(descriptors_box, descriptors_sence, matches);


	// 发现匹配

	vector<DMatch> goodMatches;

	printf("total match points : %dn", matches.size());

	float maxdist = 0;

	for (unsigned int i = 0; i < matches.size(); ++i) {

		printf("dist : %.2f n", matches[i].distance);

		maxdist = max(maxdist, matches[i].distance);

	}
	double ratio = 0.4;
	for (unsigned int i = 0; i < matches.size(); ++i) {

		if (matches[i].distance < maxdist * ratio)

			goodMatches.push_back(matches[i]);

	}


	Mat dst;

	drawMatches(image_object, keypoints_obj, image_scene, keypoints_sence, goodMatches, dst);

	//imshow("output", dst);
	//cv::waitKey(0);
	return cv::Point2f(3, 3);
}

// 参考自： https://blog.csdn.net/wsp_1138886114/article/details/119772358
cv::Point2f MatchAlgorithm::MatchByAkaza(cv::Mat image_scene, cv::Mat image_object)
{
	// AKAZE feature detect
	Ptr<AKAZE> akaze_detector = AKAZE::create();
	vector<KeyPoint> keypoints_obj, keypoints_scens;
	Mat descriptor_obj, descriptor_scens;

	double t00 = getTickCount();
	akaze_detector->detectAndCompute(image_object, Mat(), keypoints_obj, descriptor_obj);
	akaze_detector->detectAndCompute(image_scene, Mat(), keypoints_scens, descriptor_scens);
	double t11 = getTickCount();
	cout << "AKAZE cost time:" << ((t11 - t00) / getTickFrequency()) << endl;

	// BFmatch
	vector<DMatch> matches;
	BFMatcher matcher;
	Mat AKAZE_Img;
	matcher.match(descriptor_obj, descriptor_scens, matches, Mat());

	vector<DMatch> good_matches;
	double maxDist = 0, minDist = 1000;
	for (size_t i = 0; i < descriptor_obj.rows; i++) {
		double dist = matches[i].distance;
		if (dist < minDist) { minDist = dist; }
		if (dist > minDist) { maxDist = dist; }
	}

	for (size_t i = 0; i < descriptor_obj.rows; i++) {
		if (matches[i].distance < max(2.0 * minDist, 0.02)) {
			good_matches.push_back(matches[i]);
		}
	}

	drawMatches(
		image_object, keypoints_obj,
		image_scene, keypoints_scens,
		good_matches, AKAZE_Img,
		Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
	);
	//imshow("AKAZE_Img", AKAZE_Img);

	// flann_match
	FlannBasedMatcher flann_matcher(new flann::LshIndexParams(20, 10, 2));
	vector<DMatch> flann_matches;
	Mat flann_AKAZE_Img;
	flann_matcher.match(descriptor_obj, descriptor_scens, flann_matches, Mat());
	drawMatches(
		image_object, keypoints_obj,
		image_scene, keypoints_scens,
		flann_matches, flann_AKAZE_Img,
		Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
	);
	//imshow("flann_AKAZE_Img", flann_AKAZE_Img);
	//cv::waitKey(0);

	return cv::Point2f(2, 2);
}

// 参考自：https://blog.csdn.net/Marchal_G/article/details/51066901
// 参考自：https://www.cnblogs.com/shuimuqingyang/p/14428270.html
cv::Point2f MatchAlgorithm::MatchByBrisk(cv::Mat image_scene, cv::Mat image_object)
{
	//构建BRISK对象，参数都与BRISK论文保持一致，论文中的第一个参数是70
	Ptr<BRISK> brisk = BRISK::create(70, 4, 1.0f);
	std::vector<KeyPoint> kp1, kp2;
	Mat des1, des2;
	brisk->detectAndCompute(image_scene, Mat(), kp1, des1, false);
	brisk->detectAndCompute(image_object, Mat(), kp2, des2, false);
	BFMatcher matcher(NORM_HAMMING, true);
	std::vector<std::vector<DMatch> > matches;
	//此处参数很重要,90.0f为判定为相似特征点的汉明距离阈值
	matcher.radiusMatch(des1, des2, matches, 90.0f, Mat(), true);
	int matchNum = 0; //总的匹配点数
	std::vector<DMatch> good_matches;
	const float kRatioThresh = 0.7f;  // 暂不删除
	for (int i = 0; i < matches.size() && matches[i].size() > 0; i++)
	{
		++matchNum;
		//if (matches[i][0].distance < kRatioThresh * matches[i][1].distance)
		{
			good_matches.push_back(matches[i][0]);
		}
		if (matches[i].size() < 1)
			break;
	}
	//// 保留不删，测试用
	//std::cout << "Brisk匹配点数: " << matchNum << std::endl;
	//Mat img_matches;
	//drawMatches(image_scene, kp1, image_object, kp2, good_matches, img_matches);
	//imshow("Good Matches", img_matches);
	//Mat img_match;
	//drawMatches(image_scene, kp1, image_object, kp2, matches, img_match, Scalar::all(-1), Scalar::all(-1), std::vector<std::vector<char> >(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//imshow("Brisk", img_match);
	//waitKey(0);
	// 图像定位
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		// 读取 good_matches 的关键点
		obj.push_back(kp1[good_matches[i].queryIdx].pt);
		scene.push_back(kp2[good_matches[i].trainIdx].pt);
	}
	// findHomography需要4个或更多个点：obj和scene的点的个数大于等于4
	if (obj.size() < 4 || scene.size() < 4)
	{
		return cv::Point2f(0, 0);
	}
	Mat H = findHomography(obj, scene, RANSAC);
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point2f(0, 0);
	obj_corners[1] = Point2f((float)image_object.cols, 0);
	obj_corners[2] = Point2f((float)image_object.cols, (float)image_object.rows);
	obj_corners[3] = Point2f(0, (float)image_object.rows);
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H);
	return cv::Point2f(abs(scene_corners[0].x),abs(scene_corners[0].y));
}
#endif
