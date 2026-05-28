# 05-yolo-c — YOLOv5su 纯 C + OpenVINO C API 推理

使用 **OpenVINO C API**（`openvino/c/openvino.h`，链接 `libopenvino_c.so`）对 YOLOv5su 做端到端推理，支持 FP32 / INT8 ONNX。

## 依赖

| 组件 | 路径 / 版本 |
|------|-------------|
| OpenVINO | `yolo_env` conda：`.../site-packages/openvino`（2026.2.0） |
| OpenCV | 系统 4.6（仅 `opencv_bridge.cpp` 用于读图/画图） |
| 模型 | `../01-models/yolov5su_fp32.onnx` / `yolov5su_int8.onnx` |
| 测试图 | `../00-data/zidane.jpg` |

## 编译

```bash
conda activate yolo_env   # 或确保 OPENVINO_DIR 指向 openvino 安装目录
cd 05-yolo-c
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

自定义 OpenVINO 路径：

```bash
cmake .. -DOPENVINO_DIR=/path/to/openvino
```

## 运行

```bash
export LD_LIBRARY_PATH=/home/chenwei/miniconda3/envs/yolo_env/lib/python3.12/site-packages/openvino/libs:$LD_LIBRARY_PATH
./yolo_detect <image.jpg> [model.onnx] [device]
```

示例：

```bash
# FP32
./yolo_detect ../../00-data/zidane.jpg ../../01-models/yolov5su_fp32.onnx

# INT8（OpenVINO 原生支持，无需 ORT 1.26）
./yolo_detect ../../00-data/zidane.jpg ../../01-models/yolov5su_int8.onnx
```

## 目录结构

```text
05-yolo-c/
├── include/
│   ├── preprocess.h
│   └── postprocess.h
├── src/
│   ├── main.c           # OpenVINO C API 推理主流程
│   ├── preprocess.c     # LetterBox + 归一化 + HWC→CHW（纯 C）
│   ├── postprocess.c    # 解码 + NMS（纯 C）
│   └── opencv_bridge.cpp # 读图/画图（C++ 薄封装）
├── CMakeLists.txt
└── README.md
```

## 与 03-yolo-cpp 的区别

| 项目 | 语言 | API |
|------|------|-----|
| 03-yolo-cpp | C++ | OpenVINO C++ (`openvino.hpp`) |
| **05-yolo-c** | **C** | OpenVINO **C API** (`openvino/c/openvino.h`) |

预处理/后处理逻辑与 03 一致，便于板端嵌入无 C++ 运行时的场景。
