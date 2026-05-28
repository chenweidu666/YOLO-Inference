# YOLOv5su C++ ONNX 端到端推理

基于 ONNX Runtime 的 YOLOv5su 目标检测 C++ 推理。

## Project Structure

```
├── src/            # 源代码
│   ├── main_onnx.cpp    # 主程序入口
│   ├── preprocess.cpp   # 图像预处理
│   ├── postprocess.cpp  # 后处理 (NMS + 坐标映射)
│   └── onnx_inference.cpp # ONNX Runtime 推理
├── include/        # 头文件
├── build/          # 编译输出
├── data/           # 测试图片
└── outputs/        # 检测结果
```

## Build

```bash
bash build.sh
```

## Run

```bash
# FP32 model (default)
./build/yolo_detect data/zidane.jpg

# Specify model path
./build/yolo_detect data/zidane.jpg ../00-Models/yolov5su_fp32.onnx cpu
```

## Models

| Model Type | Shared Location | Status |
|------------|----------------|--------|
| FP32 | ../00-Models/yolov5su_fp32.onnx | ✅ Verified (3 detections, matches Python) |
| INT8 | ../00-Models/yolov5su_int8.onnx | ⚠️ Not supported on CPU (requires special execution provider) |

Models are shared via `../00-Models/` directory at the project root level.
