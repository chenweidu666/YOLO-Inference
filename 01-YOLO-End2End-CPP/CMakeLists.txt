cmake_minimum_required(VERSION 3.10)
project(yolo_detect)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime 路径
set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/onnxruntime)

find_package(OpenCV REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
)

link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(yolo_detect
    src/main_onnx.cpp
    src/preprocess.cpp
    src/postprocess.cpp
    src/onnx_inference.cpp
)

target_link_libraries(yolo_detect
    ${OpenCV_LIBS}
    onnxruntime
)
