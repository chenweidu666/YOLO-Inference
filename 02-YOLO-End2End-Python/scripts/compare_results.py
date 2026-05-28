#!/usr/bin/env python3
"""
Script to compare FP32 and INT8 model results
"""
import json
import numpy as np


def compare_detection_results(fp32_json_path, int8_json_path):
    """Compare detection results between FP32 and INT8 models"""
    
    # Load JSON results
    with open(fp32_json_path, 'r') as f:
        fp32_data = json.load(f)
    
    with open(int8_json_path, 'r') as f:
        int8_data = json.load(f)
    
    print("Detection Results Comparison")
    print("="*50)
    print(f"FP32 detections: {fp32_data['count']}")
    print(f"INT8 detections: {int8_data['count']}")
    
    fp32_dets = fp32_data['detections']
    int8_dets = int8_data['detections']
    
    print("\nDetailed Comparison:")
    print("-" * 80)
    print(f"{'Class':<15} {'FP32 Conf':<10} {'INT8 Conf':<10} {'Conf Diff':<10} {'IOU':<10}")
    print("-" * 80)
    
    # Match detections by class and approximate location
    matched_pairs = []
    used_int8 = set()
    
    for i, fp32_det in enumerate(fp32_dets):
        best_match = None
        best_iou = 0
        
        for j, int8_det in enumerate(int8_dets):
            if j in used_int8:
                continue
                
            # Calculate IoU between bounding boxes
            iou = calculate_bbox_iou(fp32_det['bbox'], int8_det['bbox'])
            
            # Match if same class and IoU > 0.3
            if fp32_det['class_name'] == int8_det['class_name'] and iou > 0.3:
                if iou > best_iou:
                    best_iou = iou
                    best_match = j
        
        if best_match is not None:
            matched_pairs.append((fp32_det, int8_dets[best_match]))
            used_int8.add(best_match)
        else:
            # Unmatched FP32 detection
            matched_pairs.append((fp32_det, None))
    
    # Process matched pairs
    for fp32_det, int8_det in matched_pairs:
        class_name = fp32_det['class_name']
        fp32_conf = fp32_det['confidence']
        
        if int8_det is not None:
            int8_conf = int8_det['confidence']
            conf_diff = abs(fp32_conf - int8_conf)
            iou = calculate_bbox_iou(fp32_det['bbox'], int8_det['bbox'])
            print(f"{class_name:<15} {fp32_conf:<10.3f} {int8_conf:<10.3f} {conf_diff:<10.3f} {iou:<10.3f}")
        else:
            print(f"{class_name:<15} {fp32_conf:<10.3f} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    # Check for unmatched INT8 detections
    for j, int8_det in enumerate(int8_dets):
        if j not in used_int8:
            print(f"{int8_det['class_name']:<15} {'N/A':<10} {int8_det['confidence']:<10.3f} {'N/A':<10} {'N/A':<10}")
    
    print("\nSummary:")
    print(f"- Total FP32 detections: {len(fp32_dets)}")
    print(f"- Total INT8 detections: {len(int8_dets)}")
    print(f"- Matched detections: {len([p for p in matched_pairs if p[1] is not None])}")
    print(f"- Unmatched FP32: {len([p for p in matched_pairs if p[1] is None])}")
    print(f"- Unmatched INT8: {len([j for j, _ in enumerate(int8_dets) if j not in used_int8])}")


def calculate_bbox_iou(bbox1, bbox2):
    """Calculate Intersection over Union of two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # Calculate intersection area
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    # Calculate areas of both boxes
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Calculate union
    union_area = area1 + area2 - inter_area
    
    # Calculate IoU
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compare_raw_outputs(fp32_npy_path, int8_npy_path):
    """Compare raw model outputs"""
    fp32_output = np.load(fp32_npy_path)
    int8_output = np.load(int8_npy_path)
    
    print(f"\nRaw Output Comparison:")
    print(f"FP32 output shape: {fp32_output.shape}")
    print(f"INT8 output shape: {int8_output.shape}")
    
    # Calculate differences
    abs_diff = np.abs(fp32_output - int8_output)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    std_diff = np.std(abs_diff)
    
    print(f"\nOutput Differences:")
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Std deviation of difference: {std_diff:.6f}")


def main():
    # Paths to the results
    fp32_json_path = "outputs/yolov5su/fp32/zidane_result.json"
    int8_json_path = "outputs/yolov5su/int8/zidane_result.json"
    fp32_npy_path = "outputs/yolov5su/fp32/zidane_raw.npy"
    int8_npy_path = "outputs/yolov5su/int8/zidane_raw.npy"
    
    # Compare detection results
    compare_detection_results(fp32_json_path, int8_json_path)
    
    # Compare raw outputs
    compare_raw_outputs(fp32_npy_path, int8_npy_path)


if __name__ == "__main__":
    main()