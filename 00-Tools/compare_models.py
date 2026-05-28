#!/usr/bin/env python3
"""
Script to compare different precision models (FP32 vs INT8) with new directory structure
"""
import os
import time
import numpy as np
import cv2
import onnxruntime as ort


def preprocess(image, img_size=640):
    """Preprocess image for YOLOv5-SU"""
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


def benchmark_model(session, input_tensor, num_runs=10, warmup_runs=3):
    """Benchmark model inference time"""
    input_name = session.get_inputs()[0].name
    
    # Warmup
    for _ in range(warmup_runs):
        _ = session.run(None, {input_name: input_tensor})
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        _ = session.run(None, {input_name: input_tensor})
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    fps = 1.0 / avg_time
    
    return avg_time, fps


def count_detections(model_path, image_path, conf_threshold=0.25):
    """Count detections with a specific model"""
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    input_tensor = preprocess(image)
    
    # Run inference
    outputs = session.run(None, {input_name: input_tensor})
    predictions = outputs[0]
    
    # Very basic post-processing to count detections above threshold
    pred_transposed = np.transpose(predictions[0])  # Shape: [8400, 84]
    scores = pred_transposed[:, 4:]  # [8400, 80] - confidence for each class
    object_confidences = np.max(scores, axis=1)  # [8400]
    conf_mask = object_confidences > conf_threshold
    num_detections = np.sum(conf_mask)
    
    return num_detections, predictions


def main():
    # Define model paths
    models_info = [
        ("models/yolov5su_fp32.onnx", "FP32"),
        ("models/yolov5su_int8.onnx", "INT8")
    ]
    
    image_path = "data/zidane.jpg"
    num_runs = 10
    
    print("Model Performance Comparison with Nested Output Structure")
    print("="*70)
    print(f"Test image: {image_path}")
    print(f"Benchmark runs: {num_runs}")
    print()
    
    # Check if test image exists
    if not os.path.exists(image_path):
        print(f"Error: Test image {image_path} not found")
        return
    
    results = []
    for model_path, model_name in models_info:
        if not os.path.exists(model_path):
            print(f"Warning: Model {model_path} not found, skipping...")
            continue
        
        print(f"Testing {model_name} model...")
        
        # Load model
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        input_tensor = preprocess(image)
        
        # Benchmark
        avg_time, fps = benchmark_model(session, input_tensor, num_runs)
        
        # Count detections
        num_detections, _ = count_detections(model_path, image_path)
        
        # Get model size
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        
        result = {
            'model_name': model_name,
            'model_size': model_size,
            'avg_time_ms': avg_time * 1000,
            'fps': fps,
            'num_detections': num_detections
        }
        results.append(result)
        
        print(f"  Size: {model_size:.2f} MB")
        print(f"  Avg time: {avg_time*1000:.2f} ms")
        print(f"  FPS: {fps:.2f}")
        print(f"  Detections: {num_detections}")
        print()
    
    # Print comparison summary
    print("="*70)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<10} {'Size (MB)':<10} {'Time (ms)':<12} {'FPS':<10} {'Detections':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['model_name']:<10} {result['model_size']:<10.2f} {result['avg_time_ms']:<12.2f} {result['fps']:<10.2f} {result['num_detections']:<12}")
    
    if len(results) >= 2:
        fp32_result = results[0]
        int8_result = results[1]
        
        print("\nImprovement Analysis:")
        size_reduction = ((fp32_result['model_size'] - int8_result['model_size']) / fp32_result['model_size']) * 100
        
        print(f"  Model size reduction: {size_reduction:.1f}% (from {fp32_result['model_size']:.2f}MB to {int8_result['model_size']:.2f}MB)")
        print(f"  INT8 model is {int8_result['avg_time_ms']/fp32_result['avg_time_ms']:.1f}x slower in this environment")
        print(f"  However, INT8 typically performs better on optimized hardware")


if __name__ == "__main__":
    main()