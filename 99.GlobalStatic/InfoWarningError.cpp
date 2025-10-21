# pragma execution_character_set("utf-8")
#include "InfoWarningError.h"
#include "../0.ApplicationFrame/NotifyWidget/NotifyWidget.h"

InfoWarningError* InfoWarningError::InfoWarningError_ = new InfoWarningError;

InfoWarningError::InfoWarningError()
{
}

InfoWarningError* InfoWarningError::GetInstance()
{
	if (InfoWarningError_ == NULL)
	{
		InfoWarningError_ = new InfoWarningError();
	}
	return InfoWarningError_;
}

void InfoWarningError::NotifySlot(QString title, QString text)
{
	QMessageBox message_box(QMessageBox::Warning, title,text, NULL);
	message_box.addButton("确 定", QMessageBox::AcceptRole);
	message_box.exec();
}

void InfoWarningError::ShowWindow(QString title, QString text)
{
	disconnect(InfoWarningError_, SIGNAL(Notify(QString, QString)), this, SLOT(NotifySlot(QString, QString)));
	connect(InfoWarningError_, SIGNAL(Notify(QString, QString)), this, SLOT(NotifySlot(QString, QString)));
	emit Notify(title, text);
}

void InfoWarningError::ShowMessage(QString title, QString text)
{
	NotifyWidget::PromptLineShow(text);
}