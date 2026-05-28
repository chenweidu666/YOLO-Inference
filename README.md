# 05-Projects

计算机视觉与机器学习项目集合。

## 项目列表

| 项目 | 描述 | 后端 | 模型支持 |
|------|------|------|---------|
| [01-YOLO-End2End-CPP](./01-YOLO-End2End-CPP) | YOLOv5su C++ 推理 | ONNX Runtime + OpenVINO | FP32 / INT8 |
| [02-YOLO-End2End-Python](./02-YOLO-End2End-Python) | YOLOv5su Python 推理 | ONNX Runtime | FP32 / INT8 |

## 推理结果 (zidane.jpg)

| 实现 | FP32 | INT8 |
|------|------|------|
| Python ONNX Runtime | ✅ 3 detections | ✅ 3 detections |
| C++ ONNX Runtime | ✅ 3 detections | ❌ CPU EP 不支持 ConvInteger |
| C++ OpenVINO | ✅ 3 detections | ✅ 3 detections |

## 共享资源

| 路径 | 内容 |
|------|------|
| [01-models/](./00-Models) | yolov5su FP32 + INT8 模型 (45M) |
| [00-data/](./00-Data) | COCO128 校准集 + zidane.jpg |
| [02-tools/](./00-Tools) | 量化、对比、验证脚本 |

## 环境

```bash
conda activate yolo_env    # /home/chenwei/miniconda3/envs/yolo_env
```
