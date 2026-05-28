#!/usr/bin/env python3
"""
Convert FP32 model to FP16 with proper type handling
"""
import onnx
from onnx import helper, TensorProto
from onnxconverter_common import float16
from onnxsim import simplify
import numpy as np
import os


def fix_residual_float32_casts(model):
    """
    After float16 conversion, fix any residual Cast nodes that still
    reference float32 (type 1) where they should reference float16 (type 10).
    """
    graph = model.graph
    
    for node in graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == 1:
                    # This casts to float32 - change to float16
                    attr.i = 10
                    print(f"  Fixed Cast node: {node.name}")
    
    return model


def convert_fp32_to_fp16(input_path, output_path):
    print(f"Loading model: {input_path}")
    fp32_model = onnx.load(input_path)
    orig_size = os.path.getsize(input_path)
    
    print("Converting to FP16...")
    fp16_model = float16.convert_float_to_float16(fp32_model)
    
    print("Fixing residual type issues...")
    fp16_model = fix_residual_float32_casts(fp16_model)
    
    print("Simplifying...")
    try:
        fp16_model, check = simplify(fp16_model)
        if not check:
            print("  Warning: simplified model check failed")
    except Exception as e:
        print(f"  Simplification warning: {e}")
    
    print("Fixing residual type issues again after simplification...")
    fp16_model = fix_residual_float32_casts(fp16_model)
    
    print(f"Saving FP16 model to: {output_path}")
    onnx.save(fp16_model, output_path)
    fp16_size = os.path.getsize(output_path)
    
    print(f"  Original FP32 size: {orig_size / 1024 / 1024:.2f} MB")
    print(f"  Converted FP16 size: {fp16_size / 1024 / 1024:.2f} MB")
    print(f"  Reduction: {(1 - fp16_size / orig_size) * 100:.1f}%")
    
    return fp16_model


def test_model(model_path):
    """Try to load model with onnxruntime to verify it works"""
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_name = session.get_outputs()[0].name
        print(f"  ✅ Model loads successfully!")
        print(f"     Input: {input_name} {input_shape}")
        print(f"     Output: {output_name} {session.get_outputs()[0].shape}")
        return True
    except Exception as e:
        print(f"  ❌ Model load failed: {e}")
        return False


if __name__ == "__main__":
    input_path = "models/yolov5su_fp32.onnx"
    output_path = "models/yolov5su_fp16.onnx"
    
    convert_fp32_to_fp16(input_path, output_path)
    print("\nVerifying model...")
    test_model(output_path)
