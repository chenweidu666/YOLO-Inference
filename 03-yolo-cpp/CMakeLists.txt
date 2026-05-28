cmake_minimum_required(VERSION 3.10)
project(yolo_detect)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime paths
set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/onnxruntime)

# OpenVINO paths (from Python package)
set(OPENVINO_DIR /home/chenwei/miniconda3/envs/yolo_env/lib/python3.12/site-packages/openvino)

find_package(OpenCV REQUIRED)

# ONNX Runtime executable
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
)

link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(yolo_detect_onnx
    src/main_onnx.cpp
    src/preprocess.cpp
    src/postprocess.cpp
    src/onnx_inference.cpp
)

target_link_libraries(yolo_detect_onnx
    ${OpenCV_LIBS}
    onnxruntime
)

# OpenVINO executable
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${OPENVINO_DIR}/include
)

link_directories(${OPENVINO_DIR}/libs)

add_executable(yolo_detect_ov
    src/main_ov.cpp
    src/preprocess.cpp
    src/postprocess.cpp
    src/ov_inference.cpp
)

target_link_libraries(yolo_detect_ov
    ${OpenCV_LIBS}
    openvino
)
