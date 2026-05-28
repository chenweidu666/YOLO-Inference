# 项目环境配置

## YOLOv5su推理项目专用环境

### 环境名称
`yolo_env`

### 需要安装的包
```bash
# 创建环境
conda create -n yolo_env python=3.12

# 激活环境
conda activate yolo_env

# 安装依赖
pip install opencv-python>=4.5.0
pip install onnxruntime>=1.0.0
pip install numpy>=1.18.0
pip install PyYAML>=5.4.0
pip install onnx>=1.10.0
pip install onnxsim
pip install onnxruntime-tools
pip install onnxconverter-common
pip install sympy>=1.6
```

### 使用方法
```bash
# 在使用项目前激活环境
conda activate yolo_env

# 运行项目
cd /home/chenwei/Workspace/05-Projects/02-YOLO-End2End-Python
python run_inference.py --model models/yolov5su_fp32.onnx --image data/zidane.jpg
```

### 注意事项
- 不要再使用 `env` 目录下的虚拟环境
- 所有项目相关的Python包都应在 `yolo_env` 环境中安装
- 确保每次运行项目前都激活了正确的环境