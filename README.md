# 05-Projects

计算机视觉与机器学习项目集合。

## 项目列表

| 项目 | 描述 | 后端 | 模型支持 |
|------|------|------|---------|
| [03-yolo-cpp](./03-yolo-cpp) | YOLOv5su C++ 推理 | ONNX Runtime + OpenVINO | FP32 / INT8 |
| [04-yolo-python](./04-yolo-python) | YOLOv5su Python 推理 | ONNX Runtime | FP32 / INT8 |

## zidane.jpg 测试结果

| 实现 | 模型 | 检测数 | 物体 |
|------|------|--------|------|
| Python ONNX | FP32 | 3 | sports ball 0.893, person 0.865, person 0.829 |
| Python ONNX | INT8 | 3 | sports ball 0.911, person 0.859, person 0.840 |
| C++ ONNX Runtime | FP32 | 3 | sports ball 0.893, person 0.865, person 0.829 |
| C++ OpenVINO | FP32 | 3 | sports ball 0.893, person 0.864, person 0.832 |
| C++ OpenVINO | INT8 | 3 | sports ball 0.911, person 0.857, person 0.839 |

所有方案均检测到相同的 3 个物体，置信度差异在可接受范围内。

## 共享资源

| 路径 | 内容 |
|------|------|
| [01-models](./01-models) | yolov5su FP32 + INT8 模型 (45M) |
| [00-data](./00-data) | COCO128 校准集 + zidane.jpg |
| [02-tools](./02-tools) | 量化、对比、验证脚本 |

## 环境

- Python: `yolo_env` (conda, `/home/chenwei/miniconda3/envs/yolo_env`)
- OpenVINO: 2026.2.0
- ONNX Runtime: 1.26.0
