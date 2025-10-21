#pragma once

#include "HeaderFiles.h"
#include "MacroEnumStruct.h"

VGA_API void SaveTextLog(string text);

VGA_API std::vector<std::string> SplitString(const std::string &str, char delimiter);

VGA_API double CalculateResolution(double fov_angle_y, int image_rows, double ground_height);

VGA_API std::string ReplaceAll(std::string &str, std::string oldStr, std::string newStr);

VGA_API std::string FillZeroAsPrefix(int number, int bit_count);

VGA_API bool NeedSaveDebugInfo();

VGA_API void SetNeedSaveDebugInfo(bool enable);

VGA_API void ParseFileName(string file_name, double &longitude, double &latitude, double &relative_height, double &roll, double &pitch, double &yaw);
