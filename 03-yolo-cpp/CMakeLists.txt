cmake_minimum_required(VERSION 3.10)
project(yolo_detect)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenVINO paths (from pip install openvino)
set(OPENVINO_DIR /home/chenwei/miniconda3/envs/yolo_env/lib/python3.12/site-packages/openvino)

find_package(OpenCV REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${OPENVINO_DIR}/include
)

link_directories(${OPENVINO_DIR}/libs)

add_executable(yolo_detect
    src/main_ov.cpp
    src/preprocess.cpp
    src/postprocess.cpp
    src/ov_inference.cpp
)

target_link_libraries(yolo_detect
    ${OpenCV_LIBS}
    openvino
)
