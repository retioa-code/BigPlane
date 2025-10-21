cmake_minimum_required(VERSION 3.10)
project(EnginePlatformProject)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(Src/0.VisionGISAIEngine)
add_subdirectory(Src/1.UavVisionPlatform)
#add_subdirectory(Src/7.TargetTrack)
add_subdirectory(Src/8.ChangeDetect)
