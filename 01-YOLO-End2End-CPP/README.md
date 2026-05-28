# YOLOv5su C++ ONNX 端到端推理示例

YOLOv5su 目标检测端到端 C++ 推理示例，基于 ONNX Runtime。

## Project Structure

```
01-YOLO-End2End-CPP/
├── CMakeLists.txt          # CMake 构建配置
├── build.sh                 # 一键编译脚本
├── README.md               # 本文档
├── src/                    # 源代码
│   ├── main_onnx.cpp       # 主程序入口
│   ├── preprocess.cpp     # 图像预处理 (letterbox + 归一化 + CHW)
│   ├── postprocess.cpp    # 后处理 (NMS + 坐标映射)
│   └── onnx_inference.cpp # ONNX Runtime 推理
├── include/                # 头文件
│   ├── preprocess.h
│   ├── postprocess.h
│   └── onnx_inference.h
├── build/                  # 编译输出
│   └── yolo_detect        # 可执行文件
├── data/                  # 输入数据 (图片/模型)
└── output/                # 输出结果 (检测图片)
```

## Build

```bash
# 方式1: 使用 build.sh (自动清理并重新编译)
bash build.sh

# 方式2: 手动编译
cd build
cmake ..
make -j$(nproc)
```

## Run

```bash
./build/yolo_detect <image_path> [model_path] [device]

# 示例
./build/yolo_detect data/test.jpg /path/to/yolov5s.onnx cpu
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| image_path | (必填) | 输入图像路径 |
| model_path | yolov5s.onnx | ONNX 模型路径 |
| device | cpu | 推理设备 (cpu/cuda) |

## Dependencies

- OpenCV 4.2+
- ONNX Runtime
- CMake 3.10+

## Model

使用 YOLOv5su ONNX 模型（已包含在 `data/` 目录）：

```bash
# 运行推理
./build/yolo_detect data/zidane.jpg data/yolov5su.onnx cpu
```

文件说明：
- `data/yolov5su.onnx` - YOLOv5su ONNX 模型 (36MB)
- `data/zidane.jpg` - 测试图片
