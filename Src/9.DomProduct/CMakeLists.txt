cmake_minimum_required(VERSION 3.16)
project(DomProduct)

set(CMAKE_CXX_STANDARD 17)

if (WIN32)
    set(PLATFORM_WINDOWS ON)
    add_definitions(-DPLATFORM_WINDOWS)
elseif (UNIX AND NOT APPLE)
    if (EXISTS "/proc/cpuinfo")
        file(READ "/proc/cpuinfo" CPUINFO)
        if (CPUINFO MATCHES "RK3588")
            set(PLATFORM_RK3588 ON)
            add_definitions(-DPLATFORM_RK3588)
        elseif (EXISTS "/sys/module/tegra/initstate" OR EXISTS "/etc/nv_tegra_release")
            set(PLATFORM_JETSON ON)
            add_definitions(-DPLATFORM_JETSON)
        endif ()
    endif ()
endif ()


if (PLATFORM_WINDOWS)
    message("9.DomProduct===============Detected platform: Windows")
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/nvcc.exe")
    # OnnxRuntime
    include_directories(F:/ThirdPartyLibrary/onnxruntime-win-x64-gpu-1.18.0/include)
    link_directories(F:/ThirdPartyLibrary/onnxruntime-win-x64-gpu-1.18.0/lib)
    # CUDA
    include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include")
    link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64")
    # TensorRT
    include_directories(F:/ThirdPartyLibrary/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/include)
    link_directories(F:/ThirdPartyLibrary/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/lib)
    # OpenCV
    include_directories(F:/ThirdPartyLibrary/OpenCV4.8.0_VS2019_X64/include)
    link_directories(F:/ThirdPartyLibrary/OpenCV4.8.0_VS2019_X64/x64/vc16/lib)
    # GDAL
    include_directories(F:/ThirdPartyLibrary/GDAL2.2.1_VS2015_X64/Include)
    link_directories(F:/ThirdPartyLibrary/GDAL2.2.1_VS2015_X64/Lib)
    # ShapeLib
    include_directories(F:/ThirdPartyLibrary/ShapeLib/Include)
    link_directories(F:/ThirdPartyLibrary/ShapeLib/LibDebug)
    # SpdLog
    include_directories(F:/ThirdPartyLibrary/SpdLog1.x/Include)
    link_directories(F:/ThirdPartyLibrary/SpdLog1.x/LibRelease)
    # Qt
    if (MSVC)  # ��� MSVC �������ر����ã�������⣺ C1189: #error:
        #  "Qt requires a C++17 compiler, and a suitable value for __cplusplus. On MSVC, you must pass the /Zc:__cplusplus option to the compiler."
        add_compile_options(/Zc:__cplusplus)  # ȷ�� __cplusplus ����ȷ����汾
        add_compile_options(/permissive-)     # �����ϸ��׼����ģʽ
    endif ()
    include_directories(D:/Qt/6.5.2/msvc2019_64/include)
    link_directories(D:/Qt/6.5.2/msvc2019_64/lib)
    # 0.VisionGISAIEngine UavVisionPlatform
    include_directories(../0.VisionGISAIEngine)
    include_directories(../1.UavVisionPlatform)
    # Compile and Link
    add_executable(DomProduct main.cpp)
    target_link_libraries(DomProduct Qt6Cored Qt6Guid Qt6Widgetsd)
    #    target_link_libraries(DomProduct opencv_world480d.lib)
elseif (PLATFORM_RK3588)
    message("9.DomProduct===============Detected platform: RK3588")
    #Qt
    include_directories(/usr/include/aarch64-linux-gnu/qt6)
    link_directories(/opt/Qt/6.8.3/gcc_arm64/lib)
    # OpenCV
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    # GDAL
    include_directories(/usr/include/gdal)
    link_directories(/usr/lib)
    # ShapeLib SpdLog
    include_directories(/usr/include)
    link_directories(/usr/lib/aarch64-linux-gnu)
    find_package(fmt REQUIRED)  # SpdLog
    # VisionGISAIEngine UavVisionPlatform
    include_directories(../0.VisionGISAIEngine)
    include_directories(../1.UavVisionPlatform)
    # Compile and Link
    add_executable(DomProduct main.cpp
            ../7.TargetTrack/main.cpp)
    target_link_libraries(DomProduct VisionGISAIEngine Qt6Core Qt6Gui Qt6Widgets)
    target_link_libraries(DomProduct ${OpenCV_LIBS} fmt::fmt shp)
elseif (PLATFORM_JETSON)
    message("9.DomProduct===============Detected platform: JETSON")
else ()
    message(FATAL_ERROR "===============Detected platform: Unknown")
endif ()



