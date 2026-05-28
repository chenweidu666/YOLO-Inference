cmake_minimum_required(VERSION 3.10)
project(yolo-c C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

# OpenVINO (pip / conda yolo_env)
if(NOT DEFINED OPENVINO_DIR)
    set(OPENVINO_DIR "/home/chenwei/miniconda3/envs/yolo_env/lib/python3.12/site-packages/openvino")
endif()

include_directories(
    ${OPENVINO_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
)
link_directories(${OPENVINO_DIR}/libs)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolo_detect
    src/main.c
    src/preprocess.c
    src/postprocess.c
    src/opencv_bridge.cpp
)

target_link_libraries(yolo_detect
    openvino_c
    openvino
    ${OpenCV_LIBS}
    m
)

set_target_properties(yolo_detect PROPERTIES
    BUILD_RPATH "${OPENVINO_DIR}/libs"
    INSTALL_RPATH "${OPENVINO_DIR}/libs"
)
