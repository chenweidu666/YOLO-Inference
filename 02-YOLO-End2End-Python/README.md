# YOLOv5su Python Inference

YOLOv5su目标检测单张图片推理，基于 ONNX Runtime，支持多种精度模型（FP32/INT8）。

## Project Structure

```
├── data/                 # Test data
├── models/               # Trained models
├── outputs/              # Generated outputs
│   └── {model_arch}/     # Model architecture (e.g., yolov5su)
│       ├── fp32/         # FP32 precision outputs
│       ├── fp16/         # FP16 precision outputs (⚠️ conversion issue)
│       └── int8/         # INT8 precision outputs
├── scripts/              # Scripts for running inference
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
python scripts/inference.py --model models/yolov5su_fp32.onnx --image data/zidane.jpg
```

### Detailed Usage
```bash
# FP32 model
python scripts/inference.py --model models/yolov5su_fp32.onnx --image data/zidane.jpg

# INT8 model
python scripts/inference.py --model models/yolov5su_int8.onnx --image data/zidane.jpg

# Custom output paths
python scripts/inference.py --model models/yolov5su_fp32.onnx --image data/zidane.jpg \
  --output outputs/result.jpg --json-output outputs/result.json --raw-npy-output outputs/raw.npy

# Custom thresholds
python scripts/inference.py --model models/yolov5su_fp32.onnx --image data/zidane.jpg \
  --conf-threshold 0.5 --iou-threshold 0.4
```

### Model Quantization (INT8)
```bash
python src/quantize_models.py
```

## Models

| Model Type | File Name         | Size    | Status  |
|------------|-------------------|---------|---------|
| FP32       | yolov5su_fp32.onnx| 35.12MB | ✅ Verified |
| FP16       | yolov5su_fp16.onnx| 17.55MB | ⚠️ Conversion issue (onnxruntime type mismatch) |
| INT8       | yolov5su_int8.onnx| 9.09MB  | ✅ Verified |

> **FP16 Note**: The generated FP16 model cannot be loaded by onnxruntime due to type mismatches in Resize/Cast operators within the model graph. Only FP32 and INT8 models are currently usable.

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