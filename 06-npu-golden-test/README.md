# NPU Golden I/O 测试

用板端固定的 `input_0.dat` / `output0_8400_84_1.dat` 做离线验证，**跳过前处理**，专注排查推理输出与后处理。

## 数据路径

默认指向 `01-SmallModel` 部署目录（可用环境变量覆盖）：

| 文件 | 默认路径 |
|------|----------|
| `input_0.dat` | `INPUT_DAT` → `01-SmallModel/deploy_intranet/3-yolo_inference/deploy/models/yolov5su_ASYMU8/input_0.dat` |
| `output0_8400_84_1.dat` | `OUTPUT_GOLDEN` → 同目录 |
| Acuity ONNX 参考输出 | `ACUITY_REF` → `01-SmallModel/output/yolov5su/acuity_ovxlib/reference_output.0.txt` |

## 输入格式

`input_0.dat` 为 **CHW uint8**（`640×640×3`），与 Acuity `model_input.npy * 255` 完全一致：

```
shape: (3, 640, 640) uint8  →  float32 [1,3,640,640] / 255.0
```

## 运行

```bash
conda activate yolo_env
cd 06-npu-golden-test
python run_golden_test.py
```

## 测试项

1. **输入一致性**：`input_0.dat` vs `model_input.npy`
2. **ONNX 推理**：FP32 / INT8 模型 + `input_0.dat`
3. **输出比对**：ONNX vs Acuity ref vs NPU golden（反量化）
4. **后处理**：检测框输出（letterbox: ratio=2.0, pad=(0,140)）

## 已知结论（2026-05-29）

| 对比 | MAE | 检测是否正确 |
|------|-----|-------------|
| Acuity ref 后处理 | — | ✅ 3 类目标 |
| NPU NCHW 布局 + asymu8 | — | ❌ 0 检测（class 通道全 0） |
| 错误 `[8400,84]` 布局 | — | ❌ 64 误检 |

说明：详见 [RESEARCH.md](RESEARCH.md)。`reference_output.0.txt` 是 FP32 参考，不是 INT8 NPU 输出。

**解决方向**：混合量化 `--hybrid float16`，见 `01-SmallModel/models/yolov5su/cust_qnt_layers.txt`
