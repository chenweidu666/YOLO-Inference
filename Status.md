# 项目状态记录

## 当前进展

| 项目 | 状态 |
|------|------|
| 模型共享 | ✅ 01-models/ (FP32/INT8) |
| 数据共享 | ✅ 00-data/ (COCO128校准集) |
| 工具脚本 | ✅ 02-tools/ (量化/对比等) |
| Python推理 | ✅ FP32/INT8 均正常 (ONNX Runtime) |
| CPP推理 (ONNX) | ✅ FP32 正常 (3个检测，与Python一致) |
| CPP推理 (OpenVINO) | ✅ FP32 + INT8 均正常 (3个检测) |

## C++ 推理框架

| 后端 | 模型支持 | 状态 |
|------|---------|------|
| ONNX Runtime | FP32 | ✅ 3 detections |
| OpenVINO | FP32 | ✅ 3 detections (精度与ONNX一致) |
| OpenVINO | INT8 (动态量化) | ✅ 3 detections (精度与FP32一致) |

## zidane.jpg 测试结果对比

| 实现 | 模型 | 检测数 | 物体 |
|------|------|--------|------|
| Python ONNX | FP32 | 3 | sports ball 0.893, person 0.865, person 0.829 |
| Python ONNX | INT8 | 3 | sports ball 0.911, person 0.859, person 0.840 |
| C++ ONNX Runtime | FP32 | 3 | sports ball 0.893, person 0.865, person 0.829 |
| C++ OpenVINO | FP32 | 3 | sports ball 0.893, person 0.864, person 0.832 |
| C++ OpenVINO | INT8 | 3 | sports ball 0.911, person 0.857, person 0.839 |

所有方案在 zidane.jpg 上均检测到相同的 3 个物体（类别ID一致），置信度差异在可接受范围内。

## 环境

- Python: `yolo_env` (conda, /home/chenwei/miniconda3/envs/yolo_env)
- OpenVINO: 2026.2.0 (via pip install openvino)
- ONNX Runtime: 1.26.0
