#!/usr/bin/env python3
"""
Script to compare different precision models (FP32, FP16, INT8)
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
    
    return input_tensor, (ratio, start_x, start_y, w, h)


def postprocess(predictions, info, conf_threshold=0.25, iou_threshold=0.45):
    """Post-process model predictions"""
    ratio, pad_x, pad_y, orig_w, orig_h = info
    
    # Transpose to get [8400, 84] format
    pred_transposed = np.transpose(predictions[0])  # Shape: [8400, 84]
    
    # Extract boxes [x_center, y_center, width, height] and scores
    boxes = pred_transposed[:, :4]  # [8400, 4]
    scores = pred_transposed[:, 4:]  # [8400, 80] - confidence for each class
    
    # Get the max score (highest confidence class) for each detection
    object_confidences = np.max(scores, axis=1)  # [8400]
    class_ids = np.argmax(scores, axis=1)  # [8400]
    
    # Apply confidence threshold
    conf_mask = object_confidences > conf_threshold
    filtered_boxes = boxes[conf_mask]
    filtered_confidences = object_confidences[conf_mask]
    filtered_class_ids = class_ids[conf_mask]
    
    # Convert box format from [x_center, y_center, width, height] to [x1, y1, x2, y2]
    x_centers = filtered_boxes[:, 0]
    y_centers = filtered_boxes[:, 1]
    widths = filtered_boxes[:, 2]
    heights = filtered_boxes[:, 3]
    
    x1 = x_centers - widths / 2
    y1 = y_centers - heights / 2
    x2 = x_centers + widths / 2
    y2 = y_centers + heights / 2
    
    # Adjust for letterbox padding
    x1 = (x1 - pad_x) / ratio
    y1 = (y1 - pad_y) / ratio
    x2 = (x2 - pad_x) / ratio
    y2 = (y2 - pad_y) / ratio
    
    # Clamp values to original image boundaries
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)
    
    # Stack into bounding boxes
    bboxes = np.stack([x1, y1, x2, y2], axis=1)
    
    # Prepare detections
    detections = []
    for bbox, conf, class_id in zip(bboxes, filtered_confidences, filtered_class_ids):
        detections.append({
            'bbox': [float(coord) for coord in bbox],
            'confidence': float(conf),
            'class_id': int(class_id)
        })
    
    # Apply Non-Maximum Suppression (NMS)
    return apply_nms(detections, iou_threshold)


def apply_nms(detections, iou_threshold=0.45):
    """Apply Non-Maximum Suppression to remove duplicate detections"""
    if not detections:
        return []
    
    # Extract bounding boxes and scores
    boxes = np.array([det['bbox'] for det in detections])
    scores = np.array([det['confidence'] for det in detections])
    
    # Compute areas for IoU calculation
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by confidence scores (descending)
    indices = np.argsort(scores)[::-1]
    
    keep_indices = []
    while indices.size > 0:
        # Keep the highest confidence detection
        current_idx = indices[0]
        keep_indices.append(current_idx)
        
        if indices.size == 1:
            break
        
        # Compute IoU with remaining boxes
        current_box = boxes[current_idx]
        remaining_boxes = boxes[indices[1:]]
        
        # Calculate intersection coordinates
        xx1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        yy1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        xx2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        yy2 = np.minimum(current_box[3], remaining_boxes[:, 3])
        
        # Compute intersection area
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        
        # Compute IoU
        iou = inter / (areas[current_idx] + areas[indices[1:]] - inter)
        
        # Find indices to keep (below IoU threshold)
        mask = iou < iou_threshold
        indices = indices[1:][mask]
    
    # Return filtered detections
    return [detections[i] for i in keep_indices]


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


def test_model(model_path, image_path, model_name, conf_threshold=0.25, iou_threshold=0.45):
    """Test a single model and return results"""
    print(f"\nTesting {model_name}...")
    
    # Load model
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    input_tensor, info = preprocess(image)
    
    # Benchmark
    avg_time, fps = benchmark_model(session, input_tensor)
    print(f"  Average inference time: {avg_time*1000:.2f} ms ({fps:.2f} FPS)")
    
    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    predictions = outputs[0]
    
    # Postprocess
    detections = postprocess(predictions, info, conf_threshold, iou_threshold)
    
    print(f"  Number of detections: {len(detections)}")
    
    # Print top detections
    for i, det in enumerate(detections[:3]):  # Show first 3 detections
        print(f"    Detection {i+1}: Class {det['class_id']}, Conf {det['confidence']:.3f}, Box {det['bbox']}")
    
    # Get model size
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    
    return {
        'model_name': model_name,
        'model_size': model_size,
        'avg_time_ms': avg_time * 1000,
        'fps': fps,
        'num_detections': len(detections),
        'predictions': predictions
    }


def main():
    # Define model paths
    model_configs = [
        ("models/yolov5su_fp32.onnx", "FP32"),
        ("models/yolov5su_fp16.onnx", "FP16"),
        ("models/yolov5su_int8.onnx", "INT8")
    ]
    
    image_path = "data/zidane.jpg"
    conf_threshold = 0.25
    iou_threshold = 0.45
    
    print("Precision Comparison Test")
    print("="*80)
    print(f"Test image: {image_path}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    print()
    
    # Check if test image exists
    if not os.path.exists(image_path):
        print(f"Error: Test image {image_path} not found")
        return
    
    # Test each model
    results = []
    for model_path, model_name in model_configs:
        if os.path.exists(model_path):
            result = test_model(model_path, image_path, model_name, conf_threshold, iou_threshold)
            results.append(result)
        else:
            print(f"Warning: Model {model_path} not found, skipping...")
    
    # Print comparison summary
    print("\n" + "="*80)
    print("PRECISION COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<10} {'Size (MB)':<10} {'Time (ms)':<12} {'FPS':<10} {'Detections':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['model_name']:<10} {result['model_size']:<10.2f} {result['avg_time_ms']:<12.2f} {result['fps']:<10.2f} {result['num_detections']:<12}")
    
    # Compare prediction similarity if we have multiple models
    if len(results) >= 2:
        print(f"\nPrediction Similarity Analysis (between {results[0]['model_name']} and others):")
        print("-" * 60)
        
        base_pred = results[0]['predictions']
        for i in range(1, len(results)):
            comp_pred = results[i]['predictions']
            abs_diff = np.abs(base_pred - comp_pred)
            max_diff = np.max(abs_diff)
            mean_diff = np.mean(abs_diff)
            print(f"{results[0]['model_name']} vs {results[i]['model_name']}: Max diff = {max_diff:.6f}, Mean diff = {mean_diff:.6f}")


if __name__ == "__main__":
    main()