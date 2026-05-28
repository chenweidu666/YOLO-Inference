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

## Build

```bash
bash build.sh
```

## Run

### ONNX Runtime backend
```bash
# FP32 model
./build/yolo_detect_onnx data/zidane.jpg

# Specify model path
./build/yolo_detect_onnx data/zidane.jpg ../01-models/yolov5su_fp32.onnx cpu
```

### OpenVINO backend (supports INT8)
```bash
# FP32 model
./build/yolo_detect_ov data/zidane.jpg

# INT8 model (dynamic quantization)
./build/yolo_detect_ov data/zidane.jpg ../01-models/yolov5su_int8.onnx CPU
```

## Models

| Model Type | Shared Location | Status |
|------------|----------------|--------|
| FP32 | ../01-models/yolov5su_fp32.onnx | ✅ Verified (3 detections, matches Python) |
| INT8 (dynamic) | ../01-models/yolov5su_int8.onnx | ✅ Verified (OpenVINO backend, 3 detections) |
| INT8 (static QDQ) | ../01-models/yolov5su_int8_cpu.onnx | ❌ All class scores zeroed by quantization |

Models are shared via `../01-models/` directory at the project root level.

## Test Results (zidane.jpg)

| Backend | Model | Detections | Key Objects |
|---------|-------|-----------|-------------|
| ONNX Runtime (C++) | FP32 | 3 | sports ball 0.893, person 0.865, person 0.829 |
| OpenVINO (C++) | FP32 | 3 | sports ball 0.893, person 0.864, person 0.832 |
| OpenVINO (C++) | INT8 | 3 | sports ball 0.911, person 0.857, person 0.839 |
