cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(sdf2d LANGUAGES CXX )
add_executable(main main.cpp)


#add_library(pba2D SHARED pba/pba2DHost.cu)

#target_link_libraries(main pba2D)
