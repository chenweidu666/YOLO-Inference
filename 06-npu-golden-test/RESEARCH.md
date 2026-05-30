# NPU 输出研究笔记

> 2026-05-29

## 1. 输出内存布局（已确认）

`nbg_meta.json` 定义：

```json
"shape": [1, 84, 8400],
"format": "nchw",
"scale": 2.4993700981140137
```

vpm_run 日志显示 `dim[8400 84]`，但**内存布局是 NCHW**：

```
flat_index = channel * 8400 + anchor
```

| 布局 | 索引 | vs OVXLIB | vs reference |
|------|------|-----------|--------------|
| NCHW `[84,8400]` | `c*8400+a` | **cosine=1.0** | cosine=0.989 |
| 错误 `[8400,84]` | `a*84+c` | cosine=0.014 | cosine=0.014 |

**C 代码修复**：`load_npu_output()` 使用 `raw[c*8400+a]`，不再 transpose。

## 2. INT8 量化 class 通道归零（根因）

| 通道 | OVXLIB/NPU INT8 | reference (ONNXRuntime) |
|------|-----------------|-------------------------|
| 0-3 (bbox) | max≈637 ✅ | max≈637 ✅ |
| 4-83 (class) | **全 0** ❌ | max≈0.91 ✅ |

- `reference_output.0.txt` 来自 **ONNXRuntime FP32**，不是 OVXLIB INT8 仿真输出
- OVXLIB `output0_8400_84_1.txt` 与 NPU `output0_8400_84_1.dat` **逐字节一致**（MD5 相同）
- 高 cosine（0.989）由 bbox 通道主导，**class 通道 cosine=0**

因此：即使布局正确，**纯 asymu8 模型无法做检测**（class score 全灭）。

## 3. 解决方案

### 方案 A：混合量化（推荐）

```bash
cd 01-SmallModel
bash scripts/run_yolo.sh yolov5su "" asymu8 --hybrid float16 --force
```

在 `cust_qnt_layers.txt` 中保留 output 相关层为 FP16：
- `model.24/Sigmoid_output_0`
- `output0` / `attach_output0/out0`

### 方案 B：FP16 全模型导出

若板端 NPU 支持 FP16 推理。

### 方案 C：CPU 后处理补 class

不可行——NPU 输出 class 通道无信息。

## 4. 验证命令

```bash
# 布局验证
python3 -c "
import numpy as np
flat = np.fromfile('input_0.dat path...', dtype=np.uint8)
ovx = np.loadtxt('output0_8400_84_1.txt')
npu = np.stack([flat[c*8400:(c+1)*8400]*2.49937 for c in range(84)])
print('match', np.allclose(npu, ovx.reshape(84,8400)))
"

# PC 后处理（用 Acuity float ref，仅验证 C 代码）
make native -C 3-yolo_inference
./deploy/yolo_detect_host zidane.jpg model.nb --skip-inference --golden-ref reference_output.0.txt
```
