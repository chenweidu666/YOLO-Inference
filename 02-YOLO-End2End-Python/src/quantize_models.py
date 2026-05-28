#!/usr/bin/env python3
"""
Script to quantize FP32 ONNX model to FP16 and INT8
"""
import onnx
import numpy as np
import cv2
import os
from onnxsim import simplify
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxconverter_common
from onnxconverter_common import float16


def convert_to_fp16(input_model_path, output_model_path):
    """Convert FP32 model to FP16"""
    print(f"Converting {input_model_path} to FP16...")
    
    # Load the model
    model = onnx.load(input_model_path)
    
    # Convert to FP16
    model_fp16 = float16.convert_float_to_float16(model)
    
    # Simplify the model
    try:
        model_fp16, check = simplify(model_fp16)
        assert check, "Simplified model is not equal to the original"
    except Exception as e:
        print(f"Simplifier warning: {e}")
    
    # Save the converted model
    onnx.save(model_fp16, output_model_path)
    print(f"FP16 model saved to {output_model_path}")
    
    # Get model size
    size_mb = os.path.getsize(output_model_path) / (1024 * 1024)
    print(f"FP16 model size: {size_mb:.2f} MB")


def convert_to_int8(input_model_path, output_model_path):
    """Convert FP32 model to INT8 using dynamic quantization"""
    print(f"Converting {input_model_path} to INT8...")
    
    # Dynamic quantization (weights only)
    quantized_model = quantize_dynamic(
        input_model_path,
        output_model_path,
        weight_type=QuantType.QInt8
    )
    
    print(f"INT8 model saved to {output_model_path}")
    
    # Get model size
    size_mb = os.path.getsize(output_model_path) / (1024 * 1024)
    print(f"INT8 model size: {size_mb:.2f} MB")


def create_calibration_dataset(image_path, num_samples=10):
    """Create calibration dataset for INT8 quantization"""
    # Load and preprocess the image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    # Calculate scaling factor to maintain aspect ratio
    img_size = 640
    ratio = min(img_size / w, img_size / h)
    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))
    
    # Resize image while maintaining aspect ratio
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create canvas of target size and center the image
    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    start_x = (img_size - new_w) // 2
    start_y = (img_size - new_h) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_img
    
    # Convert from BGR to RGB, then HWC to CHW
    rgb_canvas = canvas[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    
    # Normalize to [0, 1] range
    normalized_img = rgb_canvas.astype(np.float32) / 255.0
    
    # Add batch dimension
    input_tensor = np.expand_dims(normalized_img, axis=0)
    
    # Create multiple samples with slight variations
    calibration_data = []
    for i in range(num_samples):
        # Add small random noise to create variation
        noisy_input = input_tensor + np.random.normal(0, 0.01, input_tensor.shape).astype(np.float32)
        noisy_input = np.clip(noisy_input, 0, 1)  # Keep in valid range
        calibration_data.append(noisy_input)
    
    return calibration_data


def test_model_accuracy(original_model_path, quantized_model_path, image_path, threshold_diff=0.1):
    """Compare accuracy between original and quantized models"""
    print(f"Testing accuracy difference between models...")
    
    # Create session for original model
    orig_session = ort.InferenceSession(original_model_path, providers=['CPUExecutionProvider'])
    orig_input_name = orig_session.get_inputs()[0].name
    
    # Create session for quantized model
    quant_session = ort.InferenceSession(quantized_model_path, providers=['CPUExecutionProvider'])
    quant_input_name = quant_session.get_inputs()[0].name
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    
    # Preprocessing function (copied from inference.py)
    def preprocess(image, img_size=640):
        h, w = image.shape[:2]
        
        # Calculate scaling factor to maintain aspect ratio
        ratio = min(img_size / w, img_size / h)
        
        # Calculate new dimensions after scaling
        new_w = int(round(w * ratio))
        new_h = int(round(h * ratio))
        
        # Resize image while maintaining aspect ratio
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create canvas of target size and center the image
        canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
        start_x = (img_size - new_w) // 2
        start_y = (img_size - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_img
        
        # Convert from BGR to RGB, then HWC to CHW
        rgb_canvas = canvas[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        
        # Normalize to [0, 1] range
        normalized_img = rgb_canvas.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_tensor = np.expand_dims(normalized_img, axis=0)
        
        return input_tensor
    
    input_tensor = preprocess(image)
    
    # Run inference on both models
    orig_output = orig_session.run(None, {orig_input_name: input_tensor})[0]
    quant_output = quant_session.run(None, {quant_input_name: input_tensor})[0]
    
    # Calculate absolute difference
    abs_diff = np.abs(orig_output - quant_output)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    print(f"Maximum absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    
    if max_diff > threshold_diff:
        print(f"Warning: Accuracy difference is high (>{threshold_diff})")
    else:
        print(f"Accuracy difference is acceptable (<={threshold_diff})")
    
    return max_diff, mean_diff


def main():
    # Define paths
    original_model_path = "models/yolov5su_fp32.onnx"
    fp16_model_path = "models/yolov5su_fp16.onnx"
    int8_model_path = "models/yolov5su_int8.onnx"
    test_image_path = "data/zidane.jpg"
    
    print("Model Quantization Process")
    print("="*50)
    
    # Check if original model exists
    if not os.path.exists(original_model_path):
        print(f"Error: Original model {original_model_path} not found")
        return
    
    # Check if test image exists
    if not os.path.exists(test_image_path):
        print(f"Error: Test image {test_image_path} not found")
        return
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Convert to FP16
    convert_to_fp16(original_model_path, fp16_model_path)
    
    # Convert to INT8
    convert_to_int8(original_model_path, int8_model_path)
    
    print("\nQuantization complete!")
    print(f"Original model: {original_model_path}")
    print(f"FP16 model: {fp16_model_path}")
    print(f"INT8 model: {int8_model_path}")
    
    # Test accuracy differences
    print("\nTesting accuracy differences...")
    print("FP32 vs FP16:")
    fp16_max_diff, fp16_mean_diff = test_model_accuracy(original_model_path, fp16_model_path, test_image_path)
    
    print("\nFP32 vs INT8:")
    int8_max_diff, int8_mean_diff = test_model_accuracy(original_model_path, int8_model_path, test_image_path)
    
    print("\nModel Comparison Summary:")
    print(f"{'Model':<10} {'Size (MB)':<10} {'Max Diff':<12} {'Mean Diff':<12}")
    print("-" * 46)
    
    orig_size = os.path.getsize(original_model_path) / (1024 * 1024)
    fp16_size = os.path.getsize(fp16_model_path) / (1024 * 1024)
    int8_size = os.path.getsize(int8_model_path) / (1024 * 1024)
    
    print(f"{'FP32':<10} {orig_size:<10.2f} {'N/A':<12} {'N/A':<12}")
    print(f"{'FP16':<10} {fp16_size:<10.2f} {fp16_max_diff:<12.6f} {fp16_mean_diff:<12.6f}")
    print(f"{'INT8':<10} {int8_size:<10.2f} {int8_max_diff:<12.6f} {int8_mean_diff:<12.6f}")


if __name__ == "__main__":
    main()