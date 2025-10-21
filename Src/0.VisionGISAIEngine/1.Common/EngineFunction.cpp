#include "EngineFunction.h"
#include "MacroEnumStruct.h"

using namespace std;

vector<string> SplitString(const string &str, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(str);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

double CalculateResolution(double fov_angle_y, int image_rows, double ground_height) {
    auto half_angle = fov_angle_y / 2 * M_PI / 180;
    auto distance = ground_height * tan(half_angle) * 2;
    auto resolution_x = distance / image_rows;
    return resolution_x;
}

// 参考自: https://www.cnblogs.com/feipeng8848/p/17046957.html
string ReplaceAll(string &str, string oldStr, string newStr) {
    string::size_type pos = str.find(oldStr);
    while (pos != string::npos) {
        str.replace(pos, oldStr.size(), newStr);
        pos = str.find(oldStr);
    }
    return str;
}

// 对整数的前缀补零，支持2-6位整数
string FillZeroAsPrefix(int number, int bit_count) {
    auto prefix = to_string(number);
    if (bit_count == 2) {
        if (prefix.length() == 1) {
            prefix = "0" + prefix;
        }
    } else if (bit_count == 3) {
        if (prefix.length() == 1) {
            prefix = "00" + prefix;
        } else if (prefix.length() == 2) {
            prefix = "0" + prefix;
        }
    } else if (bit_count == 4) {
        if (prefix.length() == 1) {
            prefix = "000" + prefix;
        } else if (prefix.length() == 2) {
            prefix = "00" + prefix;
        } else if (prefix.length() == 3) {
            prefix = "0" + prefix;
        }
    } else if (bit_count == 5) {
        if (prefix.length() == 1) {
            prefix = "0000" + prefix;
        } else if (prefix.length() == 2) {
            prefix = "000" + prefix;
        } else if (prefix.length() == 3) {
            prefix = "00" + prefix;
        } else if (prefix.length() == 4) {
            prefix = "0" + prefix;
        }
    } else if (bit_count == 6) {
        if (prefix.length() == 1) {
            prefix = "00000" + prefix;
        } else if (prefix.length() == 2) {
            prefix = "0000" + prefix;
        } else if (prefix.length() == 3) {
            prefix = "000" + prefix;
        } else if (prefix.length() == 4) {
            prefix = "00" + prefix;
        } else if (prefix.length() == 5) {
            prefix = "0" + prefix;
        }
    }
    return prefix;
}

void SaveTextLog(string text) {
    if (!NeedSaveDebugInfo()) return;
    static std::shared_ptr<spdlog::logger> SpdLogger = nullptr;
    if (nullptr == SpdLogger) {
        auto path = QCoreApplication::applicationDirPath() + "/Running.log";
        QFile file(path);
        if (file.exists()) {
            file.remove();
        }
        SpdLogger = spdlog::basic_logger_mt("basic_logger", path.toLocal8Bit().data());
    }
    SpdLogger->info(text);
    SpdLogger->flush();
}

bool NeedSaveDebugInfo() {
    static std::atomic<bool> cachedValue = []() {
        QSettings settings("MyCompany", "MySoftware");
        return settings.value("DebugInfo/Enable", false).toBool();
    }();
    return cachedValue;
}

void SetNeedSaveDebugInfo(bool enable) {
    QSettings settings("MyCompany", "MySoftware");
    settings.setValue("DebugInfo/Enable", enable);
}

void ParseFileName(string file_name, double &longitude, double &latitude, double &relative_height, double &roll, double &pitch, double &yaw) {
    auto file_name_split = SplitString(file_name, '_');
    for (int jjj = 0; jjj < file_name_split.size(); ++jjj) {
        string element = file_name_split[jjj];
        if (element.find("Lon") != string::npos) {
            auto temp = ReplaceAll(element, "Longitude", "");
            temp = ReplaceAll(temp, "RtkLon", "");
            temp = ReplaceAll(temp, "GpsLon", "");
            longitude = atof(temp.c_str());
        }
        if (element.find("Lat") != string::npos) {
            auto temp = ReplaceAll(element, "Latitude", "");
            temp = ReplaceAll(temp, "RtkLat", "");
            temp = ReplaceAll(temp, "GpsLat", "");
            latitude = atof(temp.c_str());
        }
        if (element.find("Alt") != string::npos) {
            auto temp = ReplaceAll(element, "Altitude", "");
            temp = ReplaceAll(temp, "RtkAlt", "");
            temp = ReplaceAll(temp, "GpsAlt", "");
            temp = ReplaceAll(temp, "RelAlt", "");
            relative_height = atof(temp.c_str());
        }
        if (element.find("Roll") != string::npos) {
            auto temp = ReplaceAll(element, "Roll", "");
            roll = atof(temp.c_str());
        }
        if (element.find("Pitch") != string::npos) {
            auto temp = ReplaceAll(element, "Pitch", "");
            pitch = atof(temp.c_str());
        }
        if (element.find("Yaw") != string::npos) {
            auto temp = ReplaceAll(element, "Yaw", "");
            yaw = atof(temp.c_str());
        }
    }
}
