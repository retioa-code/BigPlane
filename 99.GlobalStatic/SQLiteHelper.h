// 实现对SQLite数据库存取
#pragma once
#include "HeaderFiles.h"

QSqlQuery SelectRow(QString table_name, QVariantMap* where_clause_equal = NULL, QString order_column = "",
                    QString where_clause_unequal = NULL, QString columns = "*");

int SelectCount(QString table, QVariantMap* where_clause_equal = NULL, QVariantMap* where_clause_unequal = NULL);

void InsertRow(QString table_name, QVariantMap* field_value_map, bool need_commit = true);

QString ConvertChineseToPinYin(QString input);

void DeleteRow(QString table_name, QVariantMap* where_clause_equal, QVariantMap* where_clause_unequal = NULL);
