#!/usr/bin/env python3
"""
NPU Golden I/O 测试：用固定 input_0.dat 跑 ONNX 推理，与 golden 对比。

跳过前处理，直接使用板端/Acuity 生成的 CHW uint8 输入。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent

# 默认数据路径（可用环境变量覆盖）
INPUT_DAT = Path(os.environ.get(
    "INPUT_DAT",
    WORKSPACE / "01-SmallModel/deploy_intranet/3-yolo_inference/deploy/models/yolov5su_ASYMU8/input_0.dat",
))
OUTPUT_GOLDEN = Path(os.environ.get(
    "OUTPUT_GOLDEN",
    WORKSPACE / "01-SmallModel/deploy_intranet/3-yolo_inference/deploy/models/yolov5su_ASYMU8/output0_8400_84_1.dat",
))
ACUITY_REF = Path(os.environ.get(
    "ACUITY_REF",
    WORKSPACE / "01-SmallModel/output/yolov5su/acuity_ovxlib/reference_output.0.txt",
))
MODEL_INPUT_NPY = Path(os.environ.get(
    "MODEL_INPUT_NPY",
    WORKSPACE / "01-SmallModel/output/yolov5su/model_input.npy",
))
MODELS_DIR = ROOT.parent / "01-models"

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# zidane.jpg letterbox @ 640: ratio=0.5, pad=(0, 140)
LETTERBOX_RATIO = 2.0
PAD_X, PAD_Y = 0, 140
CONF_THRESH = 0.25
IOU_THRESH = 0.45
NPU_OUTPUT_SCALE = 2.499370


def load_input_dat(path: Path) -> np.ndarray:
    """CHW uint8 -> float32 [1, 3, 640, 640]."""
    raw = np.fromfile(path, dtype=np.uint8)
    expected = 3 * 640 * 640
    if raw.size != expected:
        raise ValueError(f"{path}: size {raw.size}, expected {expected}")
    chw = raw.reshape(3, 640, 640).astype(np.float32) / 255.0
    return chw[np.newaxis, ...]


def load_acuity_ref(path: Path) -> np.ndarray:
    vals = []
    with open(path) as f:
        for line in f:
            for x in line.split():
                vals.append(float(x))
    arr = np.array(vals, dtype=np.float32)
    return arr.reshape(84, 8400)


def load_npu_golden(path: Path, scale: float = NPU_OUTPUT_SCALE) -> np.ndarray:
    """NBG output [1,84,8400] NCHW uint8 -> float32 [84, 8400]."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size != 8400 * 84:
        raise ValueError(f"{path}: size {raw.size}")
    out = np.zeros((84, 8400), dtype=np.float32)
    for c in range(84):
        out[c] = raw[c * 8400:(c + 1) * 8400].astype(np.float32) * scale
    return out


def compare_tensors(a: np.ndarray, b: np.ndarray, name: str) -> dict:
    diff = np.abs(a - b)
    dot = float(np.dot(a.flatten(), b.flatten()))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return {
        "name": name,
        "mae": float(diff.mean()),
        "max_abs": float(diff.max()),
        "cosine": dot / norm,
    }


def postprocess(pred: np.ndarray, source: str) -> list[dict]:
    """pred: [84, 8400] or [1, 84, 8400]."""
    if pred.ndim == 3:
        pred = pred[0]
    pred_t = pred.T
    boxes = pred_t[:, :4]
    scores = pred_t[:, 4:]
    conf = scores.max(axis=1)
    cls_id = scores.argmax(axis=1)

    detections = []
    for i in np.where(conf > CONF_THRESH)[0]:
        cx, cy, bw, bh = boxes[i]
        x1 = (cx - bw / 2 - PAD_X) / LETTERBOX_RATIO
        y1 = (cy - bh / 2 - PAD_Y) / LETTERBOX_RATIO
        x2 = (cx + bw / 2 - PAD_X) / LETTERBOX_RATIO
        y2 = (cy + bh / 2 - PAD_Y) / LETTERBOX_RATIO
        detections.append({
            "anchor": int(i),
            "class_id": int(cls_id[i]),
            "class_name": COCO_NAMES[cls_id[i]],
            "confidence": float(conf[i]),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "source": source,
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return apply_nms(detections)


def apply_nms(detections: list[dict]) -> list[dict]:
    if not detections:
        return []
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        remaining = []
        for d in detections:
            if d["class_id"] != best["class_id"]:
                remaining.append(d)
                continue
            iou = box_iou(best["bbox"], d["bbox"])
            if iou <= IOU_THRESH:
                remaining.append(d)
        detections = remaining
    return keep


def box_iou(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def run_onnx(model_path: Path, inp: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: inp})[0]
    return out[0]


def check_input_consistency() -> dict:
    raw = np.fromfile(INPUT_DAT, dtype=np.uint8).reshape(3, 640, 640)
    npy = np.load(MODEL_INPUT_NPY)
    npy_u8 = (npy[0] * 255).astype(np.uint8)
    match = bool(np.all(raw == npy_u8))
    diff = np.abs(raw.astype(np.int16) - npy_u8.astype(np.int16))
    return {
        "input_dat": str(INPUT_DAT),
        "model_input_npy": str(MODEL_INPUT_NPY),
        "match": match,
        "max_diff": int(diff.max()),
        "avg_diff": float(diff.mean()),
    }


def main() -> int:
    print("=" * 60)
    print("NPU Golden I/O Test")
    print("=" * 60)

    for p in (INPUT_DAT, OUTPUT_GOLDEN, ACUITY_REF):
        if not p.exists():
            print(f"ERROR: missing {p}")
            return 1

    # 1. Input check
    inp_info = check_input_consistency()
    print("\n[1] Input consistency")
    print(f"    input_0.dat vs model_input.npy*255: match={inp_info['match']}")

    inp = load_input_dat(INPUT_DAT)
    acuity_ref = load_acuity_ref(ACUITY_REF)
    npu_golden = load_npu_golden(OUTPUT_GOLDEN)

    # 2. ONNX inference
    print("\n[2] ONNX inference with input_0.dat")
    results = {}
    for tag, model in [("fp32", MODELS_DIR / "yolov5su_fp32.onnx"),
                       ("int8", MODELS_DIR / "yolov5su_int8.onnx")]:
        if not model.exists():
            print(f"    SKIP {tag}: {model} not found")
            continue
        out = run_onnx(model, inp)
        cmp = compare_tensors(out, acuity_ref, f"onnx_{tag}_vs_acuity")
        results[tag] = {"output_shape": list(out.shape), "compare": cmp}
        print(f"    {tag}: shape={out.shape} MAE={cmp['mae']:.6f} cosine={cmp['cosine']:.6f}")

    # 3. Output compare
    print("\n[3] Output tensor compare")
    cmp_npu = compare_tensors(npu_golden, acuity_ref, "npu_golden_vs_acuity")
    print(f"    NPU golden vs Acuity ref: MAE={cmp_npu['mae']:.4f} cosine={cmp_npu['cosine']:.6f}")

    if "fp32" in results:
        onnx_fp32 = run_onnx(MODELS_DIR / "yolov5su_fp32.onnx", inp)
        cmp_onpu = compare_tensors(onnx_fp32, npu_golden, "onnx_fp32_vs_npu_golden")
        print(f"    ONNX FP32 vs NPU golden: MAE={cmp_onpu['mae']:.4f} cosine={cmp_onpu['cosine']:.6f}")

    # 4. Postprocess
    print("\n[4] Postprocess (conf>0.25, letterbox ratio=2.0 pad_y=140)")
    sources = {}
    if (MODELS_DIR / "yolov5su_int8.onnx").exists():
        sources["onnx_int8"] = run_onnx(MODELS_DIR / "yolov5su_int8.onnx", inp)
    sources["acuity_ref"] = acuity_ref
    sources["npu_golden"] = npu_golden

    all_dets = {}
    for name, tensor in sources.items():
        dets = postprocess(tensor, name)
        all_dets[name] = dets
        print(f"\n    [{name}] {len(dets)} detections after NMS:")
        for d in dets[:5]:
            b = d["bbox"]
            print(f"      {d['class_name']}: {d['confidence']:.3f} "
                  f"({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f})")

    # 5. Save report
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    report = {
        "input": inp_info,
        "tensor_compare": {
            "npu_vs_acuity": cmp_npu,
            **{f"onnx_{k}": v["compare"] for k, v in results.items()},
        },
        "detections": {
            k: v[:10] for k, v in all_dets.items()
        },
    }
    report_path = out_dir / "golden_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[5] Report saved: {report_path}")

    # Expected: acuity_ref should find sports ball + persons
    acuity_names = {d["class_name"] for d in all_dets.get("acuity_ref", [])}
    onnx_names = {d["class_name"] for d in all_dets.get("onnx_int8", [])}
    ok = "sports ball" in acuity_names and "person" in acuity_names
    print("\n" + "=" * 60)
    print(f"Acuity ref OK (sports ball + person): {ok}")
    print(f"ONNX INT8 classes: {sorted(onnx_names)}")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
