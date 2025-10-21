cmake_minimum_required(VERSION 3.16)
#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
#project(VisionGISAIEngine LANGUAGES CXX CUDA)
project(VisionGISAIEngine)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CUDA_STANDARD 17)
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
    # Todo
    set(CMAKE_BUILD_TYPE Debug)
    message("0.VisionGISAIEngine===============Detected platform: Windows")
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/nvcc.exe")
    # OnnxRuntime
    target_include_directories(VisionGISAIEngine PRIVATE F:/ThirdPartyLibrary/onnxruntime-win-x64-gpu-1.18.0/include)
    target_link_directories(VisionGISAIEngine PRIVATE F:/ThirdPartyLibrary/onnxruntime-win-x64-gpu-1.18.0/lib)
    # CUDA
    target_include_directories(VisionGISAIEngine PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include")
    target_link_directories(VisionGISAIEngine PRIVATE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64")
    # TensorRT
    target_include_directories(VisionGISAIEngine PRIVATE F:/ThirdPartyLibrary/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/include)
    target_link_directories(VisionGISAIEngine PRIVATE F:/ThirdPartyLibrary/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/lib)
    # OpenCV
    target_include_directories(VisionGISAIEngine PRIVATE F:/ThirdPartyLibrary/OpenCV4.8.0_VS2019_X64/include)
    target_link_directories(VisionGISAIEngine PRIVATE F:/ThirdPartyLibrary/OpenCV4.8.0_VS2019_X64/x64/vc16/lib)
    # GDAL
    target_include_directories(VisionGISAIEngine PRIVATE F:/ThirdPartyLibrary/GDAL2.2.1_VS2015_X64/Include)
    target_link_directories(VisionGISAIEngine PRIVATE F:/ThirdPartyLibrary/GDAL2.2.1_VS2015_X64/Lib)
    # ShapeLib
    target_include_directories(VisionGISAIEngine PRIVATE F:/ThirdPartyLibrary/ShapeLib/Include)
    target_link_directories(VisionGISAIEngine PRIVATE F:/ThirdPartyLibrary/ShapeLib/LibDebug)
    # SpdLog
    target_include_directories(VisionGISAIEngine PRIVATE F:/ThirdPartyLibrary/SpdLog1.x/Include)
    target_link_directories(VisionGISAIEngine PRIVATE F:/ThirdPartyLibrary/SpdLog1.x/LibRelease)
    # Qt
    if (MSVC)  # ��� MSVC �������ر����ã�������⣺ C1189: #error:
        #  "Qt requires a C++17 compiler, and a suitable value for __cplusplus. On MSVC, you must pass the /Zc:__cplusplus option to the compiler."
        add_compile_options(/Zc:__cplusplus)  # ȷ�� __cplusplus ����ȷ����汾
        add_compile_options(/permissive-)     # �����ϸ��׼����ģʽ
    endif ()
    target_include_directories(VisionGISAIEngine PRIVATE D:/Qt/6.5.2/msvc2019_64/include)
    target_link_directories(VisionGISAIEngine PRIVATE D:/Qt/6.5.2/msvc2019_64/lib)
    # Compile and Link
    add_library(VisionGISAIEngine
            1.Common/HeaderFiles.h 1.Common/MacroEnumStruct.h
            1.Common/EngineFunction.cpp 1.Common/EngineFunction.h
            3.Vision/ImageMatcher.cpp 3.Vision/ImageMatcher.h
            3.Vision/OpenCVHelper.cpp 3.Vision/OpenCVHelper.h
            6.GIS/GISHelper.cpp 6.GIS/GISHelper.h
    )
    target_link_libraries(VisionGISAIEngine Qt6Cored Qt6Guid Qt6Widgetsd)
    target_link_libraries(VisionGISAIEngine onnxruntime onnxruntime_providers_shared onnxruntime_providers_cuda)
    target_link_libraries(VisionGISAIEngine shp spdlog nvinfer nvinfer_plugin cudart)
    target_link_libraries(VisionGISAIEngine opencv_world480d gdal_i)

    add_custom_command(TARGET VisionGISAIEngine POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_BINARY_DIR}/Src/0.VisionGISAIEngine/${CMAKE_BUILD_TYPE}/VisionGISAIEngine.dll"
            "${CMAKE_BINARY_DIR}/Src/5.VisionLocation/${CMAKE_BUILD_TYPE}/VisionGISAIEngine.dll"
            COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_BINARY_DIR}/Src/0.VisionGISAIEngine/${CMAKE_BUILD_TYPE}/VisionGISAIEngine.dll"
            "${CMAKE_BINARY_DIR}/Src/9.DomProduct/${CMAKE_BUILD_TYPE}/VisionGISAIEngine.dll"
            VERBATIM
    )
elseif (PLATFORM_RK3588)
    message("0.VisionGISAIEngine===============Detected platform: RK3588")
    #    #Qt, lib�ļ�λ�ã� /usr/lib/aarch64-linux-gnu
    target_include_directories(VisionGISAIEngine PRIVATE /usr/include/aarch64-linux-gnu/qt6)
    # OpenCV
    find_package(OpenCV REQUIRED)
    target_include_directories(VisionGISAIEngine PRIVATE ${OpenCV_INCLUDE_DIRS})
    # GDAL
    target_include_directories(VisionGISAIEngine PRIVATE /usr/include/gdal)
    target_link_directories(VisionGISAIEngine PRIVATE /usr/lib)
    # ShapeLib SpdLog
    target_include_directories(VisionGISAIEngine PRIVATE /usr/include)
    target_link_directories(VisionGISAIEngine PRIVATE /usr/lib/aarch64-linux-gnu)
    find_package(fmt REQUIRED)  # SpdLog needs fmt
    # Compile and Link
    add_library(VisionGISAIEngine
            1.Common/HeaderFiles.h 1.Common/MacroEnumStruct.h
            1.Common/EngineFunction.cpp 1.Common/EngineFunction.h
            3.Vision/ImageMatcher.cpp 3.Vision/ImageMatcher.h
            3.Vision/OpenCVHelper.cpp 3.Vision/OpenCVHelper.h
            6.GIS/GISHelper.cpp 6.GIS/GISHelper.h)
    target_link_libraries(VisionGISAIEngine Qt6Core Qt6Gui Qt6Widgets)
    target_link_libraries(VisionGISAIEngine ${OpenCV_LIBS} gdal shp)
elseif (PLATFORM_JETSON)
    message("0.VisionGISAIEngine===============Detected platform: Jetson")
    set(CMAKE_CUDA_ARCHITECTURES 87)
    # CUDA
    target_include_directories(VisionGISAIEngine PRIVATE /usr/local/cuda/targets/aarch64-linux/include)
    # TensorRT
    target_include_directories(VisionGISAIEngine PRIVATE /usr/src/tensorrt/samples/common)
    #Qt
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)
    find_package(Qt6 COMPONENTS Core Gui Widgets REQUIRED)
    target_include_directories(VisionGISAIEngine PRIVATE /usr/include/aarch64-linux-gnu/qt6)
    target_link_directories(VisionGISAIEngine PRIVATE /usr/lib/aarch64-linux-gnu)
    # OpenCV
    target_include_directories(VisionGISAIEngine PRIVATE /home/ThirdPartyLibrary/OpenCV4.11.0/include/opencv4)
    target_link_directories(VisionGISAIEngine PRIVATE /home/ThirdPartyLibrary/OpenCV4.11.0/lib)
    # GDAL
    target_include_directories(VisionGISAIEngine PRIVATE /home/ThirdPartyLibrary/GDA33.3.7.1/include)
    target_link_directories(VisionGISAIEngine PRIVATE /home/ThirdPartyLibrary/GDA33.3.7.1/lib)
    # ShapeLib   apt install libspdlog-dev
    target_include_directories(VisionGISAIEngine PRIVATE /home/ThirdPartyLibrary/ShapeLib/Include)
    target_link_directories(VisionGISAIEngine PRIVATE /home/ThirdPartyLibrary/ShapeLib/Lib)
    # SpdLog
    target_include_directories(VisionGISAIEngine PRIVATE /home/ThirdPartyLibrary/SpdLog/Include)
    target_link_directories(/home/ThirdPartyLibrary/SpdLog/Lib)
    #    # MyPlugins
    #    target_include_directories(VisionGISAIEngine PRIVATE /home/ThirdPartyLibrary/MyPlugins/Include)
    #    target_link_directories(VisionGISAIEngine PRIVATE /home/ThirdPartyLibrary/MyPlugins/Lib)
    # Compile and Link
    add_library(VisionGISAIEngine
            1.Common/HeaderFiles.h 1.Common/MacroEnumStruct.h
            1.Common/EngineFunction.cpp 1.Common/EngineFunction.h
            3.Vision/ImageMatcher.cpp 3.Vision/ImageMatcher.h
            3.Vision/OpenCVHelper.cpp 3.Vision/OpenCVHelper.h
            6.GIS/GISHelper.cpp 6.GIS/GISHelper.h)
    target_link_libraries(VisionGISAIEngine Qt6Core Qt6Gui Qt6Widgets spdlog)
    target_link_libraries(VisionGISAIEngine opencv_world gdal proj shp nvinfer nvinfer_plugin)
    #    target_link_libraries(VisionGISAIEngine opencv_world gdal proj shp myplugins nvinfer nvinfer_plugin)
elseif (PLATFORM_BM1684X)
    message("0.VisionGISAIEngine===============Detected platform: BM1684X")

    add_library(VisionGISAIEngine
            1.Common/HeaderFiles.h 1.Common/MacroEnumStruct.h
            1.Common/EngineFunction.cpp 1.Common/EngineFunction.h
            3.Vision/ImageMatcher.cpp 3.Vision/ImageMatcher.h
            3.Vision/OpenCVHelper.cpp 3.Vision/OpenCVHelper.h
            6.GIS/GISHelper.cpp 6.GIS/GISHelper.h)
    # Qt
    target_include_directories(VisionGISAIEngine PRIVATE /data/sn/ThirdPartyLibrary/Qt5.14/include)
    target_link_directories(VisionGISAIEngine PRIVATE /data/sn/ThirdPartyLibrary/Qt5.14/lib)
    # OpenCV
    target_include_directories(VisionGISAIEngine PRIVATE /data/sn/ThirdPartyLibrary/OpenCV4.11.0/include/opencv4)
    target_link_directories(VisionGISAIEngine PRIVATE /data/sn/ThirdPartyLibrary/OpenCV4.11.0/lib)
    # GDAL
    target_include_directories(VisionGISAIEngine PRIVATE /data/sn/ThirdPartyLibrary/GDA33.3.7.1/include)
    target_link_directories(VisionGISAIEngine PRIVATE /data/sn/ThirdPartyLibrary/GDA33.3.7.1/lib)
    # ShapeLib   apt install libspdlog-dev
    target_include_directories(VisionGISAIEngine PRIVATE /data/sn/ThirdPartyLibrary/ShapeLib/Include)
    target_link_directories(VisionGISAIEngine PRIVATE /data/sn/ThirdPartyLibrary/ShapeLib/Lib)
    # SpdLog
    target_include_directories(VisionGISAIEngine PRIVATE /data/sn/ThirdPartyLibrary/SpdLog/Include)
    target_link_directories(VisionGISAIEngine PRIVATE /data/sn/ThirdPartyLibrary/SpdLog/Lib)

    target_link_libraries(VisionGISAIEngine Qt5Core Qt5Gui Qt5Widgets)
    target_link_libraries(VisionGISAIEngine opencv_world gdal proj shp spdlog)
else ()
    message(FATAL_ERROR "===============Detected platform: Unknown")
endif ()
