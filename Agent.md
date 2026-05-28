# 项目环境配置

## Conda 环境

```bash
# 创建环境
conda create -n yolo_env python=3.12
conda activate yolo_env

# 安装依赖
pip install opencv-python onnxruntime onnx onnxsim PyYAML sympy openvino
```

## 推送代码到 GitHub（国内需代理）

通过 furalike 建立 SSH 隧道：

```bash
# 后台建立 SOCKS5 隧道
ssh -f -N -D 1080 furalike

# 通过隧道推送（--force 因历史已用 filter-branch 重写）
ALL_PROXY=socks5://127.0.0.1:1080 git push --force origin master

# 关闭隧道
pkill -f "ssh -f -N -D 1080 furalike"
```

## 项目结构

```
05-Projects/
├── 00-data/          校准 + 测试数据
├── 01-models/        FP32 + INT8 模型
├── 02-tools/         量化、对比、验证脚本
├── 03-yolo-cpp/      C++ OpenVINO 推理
└── 04-yolo-python/   Python ONNX Runtime 推理
```
