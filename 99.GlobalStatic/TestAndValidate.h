// 调试时，显示一个三维相关面
#ifndef TESTANDVALIDATE_H
#define TESTANDVALIDATE_H

#include "HeaderFiles.h"

class TestAndValidate : public QObject
{
Q_OBJECT
public:
	TestAndValidate();
	static TestAndValidate* TestAndValidate_;

	static TestAndValidate* GetInstance();
	cv::Mat correlative_sufrace_;
signals:
void ShowCorrelativeSurfaceSignal();

public slots:
void ShowCorrelativeSurfaceSlot();
void VisualizeCorrelativeSurface(cv::Mat correlative_sufrace);
};

#endif //
