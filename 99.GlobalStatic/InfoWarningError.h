// 辅助实现右下角的消息提示
#ifndef INFOWARNINGERROR_H
#define INFOWARNINGERROR_H

#include "HeaderFiles.h"

class InfoWarningError : public QObject
{
Q_OBJECT
public:
	InfoWarningError();
	static InfoWarningError* InfoWarningError_;

	static InfoWarningError* GetInstance();

signals:
	void Notify(QString title, QString text);

public slots:
	void NotifySlot(QString title, QString text);
	void ShowWindow(QString title, QString text);
void ShowMessage(QString title, QString text);
};

#endif // INFOWARNINGERROR_H
