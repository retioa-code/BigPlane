cmake_minimum_required(VERSION 3.16)
project(UavVisionPlatform)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 17)
set(BUILD_SHARED_LIBS ON)

if (WIN32)
    set(PLATFORM_WINDOWS ON)
    add_definitions(-DPLATFORM_WINDOWS)
elseif (UNIX AND NOT APPLE)
    if (EXISTS "/proc/cpuinfo")
        file(READ "/proc/cpuinfo" CPUINFO)
        if (CPUINFO MATCHES "RK3588")
            set(PLATFORM_RK3588 ON)
            add_definitions(-DPLATFORM_RK3588)
        elseif (CPUINFO MATCHES "bm1684x")
            set(PLATFORM_BM1684X ON)
            add_definitions(-DPLATFORM_BM1684X)
        elseif (EXISTS "/sys/module/tegra/initstate" OR EXISTS "/etc/nv_tegra_release")

            set(PLATFORM_JETSON ON)
            add_definitions(-DPLATFORM_JETSON)
        endif ()
    endif ()
endif ()

if (PLATFORM_WINDOWS)
    # Todo ���Ľ�
    set(CMAKE_BUILD_TYPE Debug)
    message("1.UavVisionPlatform===============Detected platform: Windows")
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/nvcc.exe")
    # OnnxRuntime
    target_include_directories(UavVisionPlatform PRIVATE F:/ThirdPartyLibrary/onnxruntime-win-x64-gpu-1.18.0/include)
    target_link_directories(UavVisionPlatform PRIVATE F:/ThirdPartyLibrary/onnxruntime-win-x64-gpu-1.18.0/lib)
    # CUDA
    target_include_directories(UavVisionPlatform PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include")
    target_link_directories(UavVisionPlatform PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64")
    # TensorRT
    target_include_directories(UavVisionPlatform PRIVATE F:/ThirdPartyLibrary/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/include)
    target_link_directories(UavVisionPlatform PRIVATE F:/ThirdPartyLibrary/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/lib)
    # OpenCV
    target_include_directories(UavVisionPlatform PRIVATE F:/ThirdPartyLibrary/OpenCV4.8.0_VS2019_X64/include)
    target_link_directories(UavVisionPlatform PRIVATE F:/ThirdPartyLibrary/OpenCV4.8.0_VS2019_X64/x64/vc16/lib)
    # GDAL
    target_include_directories(UavVisionPlatform PRIVATE F:/ThirdPartyLibrary/GDAL2.2.1_VS2015_X64/Include)
    target_link_directories(UavVisionPlatform PRIVATE F:/ThirdPartyLibrary/GDAL2.2.1_VS2015_X64/Lib)
    # ShapeLib
    target_include_directories(UavVisionPlatform PRIVATE F:/ThirdPartyLibrary/ShapeLib/Include)
    target_link_directories(UavVisionPlatform PRIVATE F:/ThirdPartyLibrary/ShapeLib/LibDebug)
    # SpdLog
    target_include_directories(UavVisionPlatform PRIVATE F:/ThirdPartyLibrary/SpdLog1.x/Include)
    target_link_directories(UavVisionPlatform PRIVATE F:/ThirdPartyLibrary/SpdLog1.x/LibRelease)
    # Qt
    if (MSVC)  # ��� MSVC �������ر����ã�������⣺ C1189: #error:
        #  "Qt requires a C++17 compiler, and a suitable value for __cplusplus. On MSVC, you must pass the /Zc:__cplusplus option to the compiler."
        add_compile_options(/Zc:__cplusplus)  # ȷ�� __cplusplus ����ȷ����汾
        add_compile_options(/permissive-)     # �����ϸ��׼����ģʽ
    endif ()
    target_include_directories(UavVisionPlatform PRIVATE D:/Qt/6.5.2/msvc2019_64/include)
    target_link_directories(UavVisionPlatform PRIVATE D:/Qt/6.5.2/msvc2019_64/lib)
    # 0.VisionGISAIEngine
    target_include_directories(UavVisionPlatform PRIVATE ../0.VisionGISAIEngine)
    # Compile and Link
    add_library(UavVisionPlatform
            0.Platform/PlatformFunctions.cpp 0.Platform/PlatformFunctions.h
            0.Platform/PlatformVariable.cpp 0.Platform/PlatformVariable.h
            6.Locator/Locator.cpp 6.Locator/Locator.h
            6.Locator/LocationThread.cpp 6.Locator/LocationThread.h
            9.Mapper/Mapper.cpp
            9.Mapper/Mapper.h)
    target_link_libraries(UavVisionPlatform Qt6Cored Qt6Guid Qt6Widgetsd)
    target_link_libraries(UavVisionPlatform opencv_world480d.lib gdal_i)
    add_custom_command(TARGET UavVisionPlatform POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_BINARY_DIR}/Src/1.UavVisionPlatform/${CMAKE_BUILD_TYPE}/UavVisionPlatform.dll"
            "${CMAKE_BINARY_DIR}/Src/5.VisionLocation/${CMAKE_BUILD_TYPE}/UavVisionPlatform.dll"
            COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_BINARY_DIR}/Src/1.UavVisionPlatform/${CMAKE_BUILD_TYPE}/UavVisionPlatform.dll"
            "${CMAKE_BINARY_DIR}/Src/9.DomProduct/${CMAKE_BUILD_TYPE}/UavVisionPlatform.dll"
            VERBATIM
    )
elseif (PLATFORM_RK3588)
    message("1.UavVisionPlatform===============Detected platform: RK3588")
    #Qt
    target_include_directories(UavVisionPlatform PRIVATE /usr/include/aarch64-linux-gnu/qt6)
    #    target_link_directories(UavVisionPlatform PRIVATE /opt/Qt/6.8.3/gcc_arm64/lib)
    # OpenCV
    find_package(OpenCV REQUIRED)
    target_include_directories(UavVisionPlatform PRIVATE ${OpenCV_INCLUDE_DIRS})
    # GDAL
    target_include_directories(UavVisionPlatform PRIVATE /usr/include/gdal)
    target_link_directories(UavVisionPlatform PRIVATE /usr/lib)
    # ShapeLib SpdLog
    target_include_directories(UavVisionPlatform PRIVATE /usr/include)
    target_link_directories(UavVisionPlatform PRIVATE /usr/lib/aarch64-linux-gnu)
    find_package(fmt REQUIRED)  # SpdLog
    # 0.VisionGISAIEngine
    target_include_directories(UavVisionPlatform PRIVATE ../0.VisionGISAIEngine)
    # Compile and Link
    add_library(UavVisionPlatform
            0.Platform/PlatformFunctions.cpp 0.Platform/PlatformFunctions.h
            0.Platform/PlatformVariable.cpp 0.Platform/PlatformVariable.h
            6.Locator/Locator.cpp 6.Locator/Locator.h
            6.Locator/LocationThread.cpp 6.Locator/LocationThread.h
            ../9.DomProduct/main.cpp
            9.Mapper/Mapper.cpp
            9.Mapper/Mapper.h)
    target_link_libraries(UavVisionPlatform VisionGISAIEngine Qt6Core Qt6Gui Qt6Widgets)
    target_link_libraries(UavVisionPlatform ${OpenCV_LIBS} gdal fmt::fmt shp)
elseif (PLATFORM_JETSON)
    message("1.UavVisionPlatform===============Detected platform: JETSON")
    #Qt
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)
    find_package(Qt6 COMPONENTS Core Gui Widgets REQUIRED)
    target_include_directories(UavVisionPlatform PRIVATE /usr/include/aarch64-linux-gnu/qt6)
    target_link_directories(UavVisionPlatform PRIVATE /usr/lib/aarch64-linux-gnu)
    # OpenCV
    target_include_directories(UavVisionPlatform PRIVATE /home/ThirdPartyLibrary/OpenCV4.11.0/include/opencv4)
    target_link_directories(UavVisionPlatform PRIVATE /home/ThirdPartyLibrary/OpenCV4.11.0/lib)
    # GDAL
    target_include_directories(UavVisionPlatform PRIVATE /home/ThirdPartyLibrary/GDA33.3.7.1/include)
    target_link_directories(UavVisionPlatform PRIVATE /home/ThirdPartyLibrary/GDA33.3.7.1/lib)
    # ShapeLib   apt install libspdlog-dev
    target_include_directories(UavVisionPlatform PRIVATE /home/ThirdPartyLibrary/ShapeLib/Include)
    target_link_directories(UavVisionPlatform PRIVATE /home/ThirdPartyLibrary/ShapeLib/Lib)
    # SpdLog
    target_include_directories(UavVisionPlatform PRIVATE /home/ThirdPartyLibrary/SpdLog/Include)
    link_libraries(/home/ThirdPartyLibrary/SpdLog/Lib/libspdlog.so)
    # 0.VisionGISAIEngine
    target_include_directories(UavVisionPlatform PRIVATE ../0.VisionGISAIEngine)
    get_filename_component(CURRENT_BUILD_DIR_NAME ${CMAKE_CURRENT_BINARY_DIR} NAME)
    target_link_directories(UavVisionPlatform PRIVATE ../0.VisionGISAIEngine/${CURRENT_BUILD_DIR_NAME})

    # Compile and Link
    add_library(UavVisionPlatform
            0.Platform/PlatformFunctions.cpp 0.Platform/PlatformFunctions.h
            0.Platform/PlatformVariable.cpp 0.Platform/PlatformVariable.h
#            6.Locator/Locator.cpp 6.Locator/Locator.h
#            6.Locator/LocationThread.cpp 6.Locator/LocationThread.h
            7.Tracker/ITracker.cpp 7.Tracker/ITracker.h
#            9.Mapper/Mapper.cpp 9.Mapper/Mapper.h
    )
    target_link_libraries(UavVisionPlatform Qt6Core Qt6Gui Qt6Widgets)
    target_link_libraries(UavVisionPlatform opencv_world gdal proj shp)
    target_link_libraries(UavVisionPlatform VisionGISAIEngine)
elseif (PLATFORM_BM1684X)
    message("0.UavVisionPlatform===============Detected platform: BM1684X")
    # Compile and Link
    add_library(UavVisionPlatform
            0.Platform/PlatformFunctions.cpp 0.Platform/PlatformFunctions.h
            0.Platform/PlatformVariable.cpp 0.Platform/PlatformVariable.h
            #            6.Locator/Locator.cpp 6.Locator/Locator.h
            #            6.Locator/LocationThread.cpp 6.Locator/LocationThread.h
            #            7.Tracker/ITracker.cpp 7.Tracker/ITracker.h
            #            9.Mapper/Mapper.cpp 9.Mapper/Mapper.h
            8.Detector/Detector.cpp 8.Detector/Detector.h)
    # Qt
    target_include_directories(UavVisionPlatform PRIVATE /data/sn/ThirdPartyLibrary/Qt5.14/include)
    target_link_directories(UavVisionPlatform PRIVATE /data/sn/ThirdPartyLibrary/Qt5.14/lib)
    # OpenCV
    target_include_directories(UavVisionPlatform PRIVATE /data/sn/ThirdPartyLibrary/OpenCV4.11.0/include/opencv4)
    target_link_directories(UavVisionPlatform PRIVATE /data/sn/ThirdPartyLibrary/OpenCV4.11.0/lib)
    # GDAL
    target_include_directories(UavVisionPlatform PRIVATE /data/sn/ThirdPartyLibrary/GDA33.3.7.1/include)
    target_link_directories(UavVisionPlatform PRIVATE /data/sn/ThirdPartyLibrary/GDA33.3.7.1/lib)
    # ShapeLib   apt install libspdlog-dev
    target_include_directories(UavVisionPlatform PRIVATE /data/sn/ThirdPartyLibrary/ShapeLib/Include)
    target_link_directories(UavVisionPlatform PRIVATE /data/sn/ThirdPartyLibrary/ShapeLib/Lib)
    # SpdLog
    target_include_directories(UavVisionPlatform PRIVATE /data/sn/ThirdPartyLibrary/SpdLog/Include)
    target_link_directories(UavVisionPlatform PRIVATE /data/sn/ThirdPartyLibrary/SpdLog/Lib)
    # 0.VisionGISAIEngine
    target_include_directories(UavVisionPlatform PRIVATE /tmp/tmp.2TbcLdBz9E/EnginePlatformProject/Src/0.VisionGISAIEngine)
    target_include_directories(UavVisionPlatform PRIVATE ../0.VisionGISAIEngine)
    target_include_directories(UavVisionPlatform PRIVATE ../0.VisionGISAIEngine/1.Common)
    get_filename_component(CURRENT_BUILD_DIR_NAME ${CMAKE_CURRENT_BINARY_DIR} NAME)
    target_link_directories(UavVisionPlatform PRIVATE ../0.VisionGISAIEngine/${CURRENT_BUILD_DIR_NAME})

    target_link_libraries(UavVisionPlatform Qt5Core Qt5Gui Qt5Widgets)
    target_link_libraries(UavVisionPlatform opencv_world gdal proj shp)
    target_link_libraries(UavVisionPlatform VisionGISAIEngine)
else ()
    message(FATAL_ERROR "===============Detected platform: Unknown")
endif ()
