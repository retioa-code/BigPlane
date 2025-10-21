add_executable(TargetTrack main.cpp)
message("7.TargetTrack===============================================")
# Qt
target_include_directories(TargetTrack PRIVATE /data/sn/ThirdPartyLibrary/Qt5.14/include)
target_link_directories(TargetTrack PRIVATE /data/sn/ThirdPartyLibrary/Qt5.14/lib)
# OpenCV
target_include_directories(TargetTrack PRIVATE /data/sn/ThirdPartyLibrary/OpenCV4.11.0/include/opencv4)
target_link_directories(TargetTrack PRIVATE /data/sn/ThirdPartyLibrary/OpenCV4.11.0/lib)
# GDAL
target_include_directories(TargetTrack PRIVATE /data/sn/ThirdPartyLibrary/GDA33.3.7.1/include)
target_link_directories(TargetTrack PRIVATE /data/sn/ThirdPartyLibrary/GDA33.3.7.1/lib)
# ShapeLib   apt install libspdlog-dev
target_include_directories(TargetTrack PRIVATE /data/sn/ThirdPartyLibrary/ShapeLib/Include)
target_link_directories(TargetTrack PRIVATE /data/sn/ThirdPartyLibrary/ShapeLib/Lib)
# SpdLog
target_include_directories(TargetTrack PRIVATE /data/sn/ThirdPartyLibrary/SpdLog/Include)
target_link_directories(TargetTrack PRIVATE /data/sn/ThirdPartyLibrary/SpdLog/Lib)

target_link_libraries(TargetTrack PRIVATE Qt5Core Qt5Gui Qt5Widgets)
target_link_libraries(TargetTrack PRIVATE opencv_world gdal proj shp spdlog)
target_link_libraries(TargetTrack PRIVATE VisionGISAIEngine UavVisionPlatform)