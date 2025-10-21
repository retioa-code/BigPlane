# pragma execution_character_set("utf-8")
#include "TestAndValidate.h"

TestAndValidate* TestAndValidate::TestAndValidate_ = new TestAndValidate;

TestAndValidate::TestAndValidate()
{
}

TestAndValidate* TestAndValidate::GetInstance()
{
	if (TestAndValidate_ == NULL)
	{
		TestAndValidate_ = new TestAndValidate();
	}
	return TestAndValidate_;
}

void TestAndValidate::ShowCorrelativeSurfaceSlot()
{
#ifdef PlatformIsWindows
	static int count = 0;
	if (count > 1) return;
	count++;
	Q3DScatter* scatter = new Q3DScatter;
	scatter->setFlags(scatter->flags() ^ Qt::FramelessWindowHint);
	QScatter3DSeries *series = new QScatter3DSeries;
	QScatterDataArray data;
	for (int row = 0; row < correlative_sufrace_.rows; ++row)
	{
		for (int col = 0; col < correlative_sufrace_.cols; ++col)
		{
			// 把相关面，拉伸了50倍
			data << QVector3D((float)row, (float)col, correlative_sufrace_.at<float>(col, row) * 50);
		}
	}
	series->dataProxy()->addItems(data);
	scatter->addSeries(series);
	scatter->showMaximized();
#endif
}

void TestAndValidate::VisualizeCorrelativeSurface(cv::Mat correlative_sufrace)
{
	correlative_sufrace_ = correlative_sufrace;
	// 以下方式是弹窗，必须有用户交互
	disconnect(TestAndValidate_, SIGNAL(ShowCorrelativeSurfaceSignal()), this, SLOT(ShowCorrelativeSurfaceSlot()));
	connect(TestAndValidate_, SIGNAL(ShowCorrelativeSurfaceSignal()), this, SLOT(ShowCorrelativeSurfaceSlot()));
	emit ShowCorrelativeSurfaceSignal();
}
