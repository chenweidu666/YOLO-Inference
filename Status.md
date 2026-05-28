# 项目状态记录

## 当前进展

| 项目 | 状态 |
|------|------|
| 模型共享 | ✅ 00-Models/ (FP32/INT8) |
| 数据共享 | ✅ 00-Data/ |
| 工具脚本 | ✅ 00-Tools/ (量化/对比等) |
| Python推理 | ✅ FP32/INT8 均正常 |
| CPP推理 | ✅ FP32 正常 (3个检测，与Python一致) |
| CPP INT8 | ❌ 待解决 |

## INT8 CPP推理问题

### 问题描述
ONNX Runtime CPU Execution Provider不支持 `ConvInteger`/`DynamicQuantizeLinear` 算子。动态量化生成的INT8模型无法在CPU上运行。

错误信息：
```
Could not find an implementation for ConvInteger(10) node
```

### 尝试的方案

#### 方案1: 静态量化 QOperator 格式 ✅ 模型可加载，但精度为0
- `QuantFormat.QOperator` + `per_channel=False`
- 模型9.02MB，CPU EP加载成功
- ❌ 检测到0个物体（校准数据不足）

#### 方案2: 静态量化 QDQ + 升级opset ❌ DequantizeLinear axis错误
- `QuantFormat.QDQ` + `per_channel=True`
- 模型原opset=12，不支持DequantizeLinear的axis属性（opset13+）
- `quant_pre_process` 未能正确升级opset

### 根因
1. 原始模型 opset=12，算子集过旧
2. YOLOv5su输出格式特殊（84维，无obj_conf），量化校准需匹配

### 下一步方向
- 手动升级模型opset后再静态量化
- 或使用 OpenVINO EP 加载动态量化模型
- 或探索 QOperator + 充分校准数据
