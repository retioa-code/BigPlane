#pragma once

// https://blog.csdn.net/anlian523/article/details/102768496
#ifdef WIN32
#define __FILENAME__ (strrchr("\\" __FILE__, '\\') + 1)
#else
#define __FILENAME__ (strrchr("/" __FILE__, '/') + 1)
#endif

#ifdef WIN32
#define VGA_API __declspec(dllexport)
#else
#define VGA_API __attribute__((visibility("default")))
#endif

#ifdef WIN32
#define UVP_API __declspec(dllexport)
#else
#define UVP_API __attribute__((visibility("default")))
#endif

enum RunningStatusEnum {
    Debugging = 0,  // 室内调试
    FlyingTest = 1 // 室外飞测
};

struct StdVector2 {
    StdVector2() {
    }

    StdVector2(double x, double y) {
        X = x;
        Y = y;
    }

    double X = 0;
    double Y = 0;
};

// 按习惯顺序表示：经纬高，横滚角俯仰角偏航角
struct VGA_API StdVector3 {
    StdVector3() {
    }

    StdVector3(double x, double y, double z) {
        X = x;
        Y = y;
        Z = z;
    }

    double X = 0;
    double Y = 0;
    double Z = 0;
};
