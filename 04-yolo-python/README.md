# YOLOv5su Python Inference

YOLOv5su 目标检测单张图片推理，基于 **OpenVINO**，支持 FP32 / INT8 ONNX 模型。

## Project Structure

```
├── outputs/              # Generated outputs
│   └── {model_arch}/     # Model architecture (e.g., yolov5su)
│       ├── fp32/         # FP32 precision outputs
│       ├── fp16/         # FP16 precision outputs (⚠️ conversion issue)
│       └── int8/         # INT8 precision outputs
├── src/                  # Source code modules
├── README.md
└── requirements.txt
```

## Installation

### Prerequisites
- Python 3.8+
- Conda (recommended)

### Setup
```bash
conda create -n yolo_env python=3.12
conda activate yolo_env
pip install -r requirements.txt
```

## Usage

### Quick Start
```bash
python src/inference.py --model ../01-models/yolov5su_fp32.onnx --image ../00-data/zidane.jpg
```

### Detailed Usage
```bash
# FP32 model
python src/inference.py --model ../01-models/yolov5su_fp32.onnx --image ../00-data/zidane.jpg

# INT8 model
python src/inference.py --model ../01-models/yolov5su_int8.onnx --image ../00-data/zidane.jpg

# Custom thresholds
python src/inference.py --model ../01-models/yolov5su_fp32.onnx --image ../00-data/zidane.jpg \
  --conf-threshold 0.5 --iou-threshold 0.4

# Benchmark mode
python src/inference.py --model ../01-models/yolov5su_fp32.onnx --image ../00-data/zidane.jpg --benchmark
```

### Model Quantization (INT8)
```bash
python ../02-tools/quantize_int8_dynamic.py
```

## Models (shared via ../01-models/)

| Model Type | File Name                | Size    | Status  |
|------------|--------------------------|---------|---------|
| FP32       | ../01-models/yolov5su_fp32.onnx | 35.12MB | ✅ Verified |
| INT8 (dynamic) | ../01-models/yolov5su_int8.onnx | 9.09MB  | ✅ Verified |

## Test Results (zidane.jpg)

| Model | Detections | Objects |
|-------|-----------|---------|
| FP32  | 3 | sports ball (0.893), person (0.865), person (0.829) |
| INT8  | 3 | sports ball (0.911), person (0.859), person (0.839) |

## Output Structure
```
outputs/yolov5su/
├── fp32/
│   ├── zidane_result.jpg    # Visualized detection
│   ├── zidane_result.json   # JSON detection results
│   └── zidane_raw.npy       # Raw model output
└── int8/
    ├── zidane_result.jpg
    ├── zidane_result.json
    └── zidane_raw.npy
```

## Environment
Use conda env `yolo_env` (see `Agent.md` for setup).