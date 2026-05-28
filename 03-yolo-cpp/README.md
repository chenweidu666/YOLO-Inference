# YOLOv5su C++ OpenVINO 推理

基于 OpenVINO 的 YOLOv5su 目标检测 C++ 推理，支持 FP32 / INT8。

## Build

```bash
bash build.sh
```

## Run

```bash
# FP32
./build/yolo_detect ../00-data/zidane.jpg ../01-models/yolov5su_fp32.onnx CPU

# INT8 (dynamic quantization)
./build/yolo_detect ../00-data/zidane.jpg ../01-models/yolov5su_int8.onnx CPU
```

## Models (shared via ../01-models/)

| Model | Status |
|-------|--------|
| FP32 | ✅ Verified (3 detections, matches Python) |
| INT8 (dynamic) | ✅ Verified (OpenVINO, 3 detections) |

## Test Results (zidane.jpg)

| Backend | Model | Detections | Key Objects |
|---------|-------|-----------|-------------|
| OpenVINO (C++) | FP32 | 3 | sports ball 0.893, person 0.864, person 0.832 |
| OpenVINO (C++) | INT8 | 3 | sports ball 0.911, person 0.857, person 0.839 |
