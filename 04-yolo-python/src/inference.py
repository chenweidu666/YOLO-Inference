#!/usr/bin/env python3
"""
Simple and Clean YOLOv5-SU Single Image Inference
"""

import os
import cv2
import numpy as np
import argparse
from typing import List, Tuple, Dict

from openvino import Core


class YOLOv5SU:
    """
    Clean YOLOv5-SU Inference Pipeline for single image
    """
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 device: str = "CPU"):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        core = Core()
        self.compiled = core.compile_model(model_path, device)
        self.infer_request = self.compiled.create_infer_request()
        
        # Model properties
        self.img_size = 640
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        print(f"Model loaded (OpenVINO/{device}): {model_path}")
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float, int, int]]:
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        ratio = min(self.img_size / w, self.img_size / h)
        new_w = int(round(w * ratio))
        new_h = int(round(h * ratio))
        
        # Resize and pad
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        start_x = (self.img_size - new_w) // 2
        start_y = (self.img_size - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_img
        
        # BGR to RGB, HWC to CHW, normalize
        rgb_canvas = canvas[:, :, ::-1].transpose(2, 0, 1)
        normalized_img = rgb_canvas.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(normalized_img, axis=0)
        
        return input_tensor, (ratio, start_x, start_y, w, h)
    
    def postprocess(self, predictions: np.ndarray, info: Tuple[float, float, float, int, int]) -> List[Dict]:
        ratio, pad_x, pad_y, orig_w, orig_h = info
        
        # Process predictions
        pred_transposed = np.transpose(predictions[0])  # [8400, 84]
        boxes = pred_transposed[:, :4]  # [8400, 4]
        scores = pred_transposed[:, 4:]  # [8400, 80]
        
        # Get max scores and class IDs
        object_confidences = np.max(scores, axis=1)  # [8400]
        class_ids = np.argmax(scores, axis=1)  # [8400]
        
        # Apply confidence threshold
        conf_mask = object_confidences > self.conf_threshold
        filtered_boxes = boxes[conf_mask]
        filtered_confidences = object_confidences[conf_mask]
        filtered_class_ids = class_ids[conf_mask]
        
        # Convert boxes to [x1, y1, x2, y2] format
        x_centers = filtered_boxes[:, 0]
        y_centers = filtered_boxes[:, 1]
        widths = filtered_boxes[:, 2]
        heights = filtered_boxes[:, 3]
        
        x1 = x_centers - widths / 2
        y1 = y_centers - heights / 2
        x2 = x_centers + widths / 2
        y2 = y_centers + heights / 2
        
        # Adjust for padding and scale back to original size
        x1 = (x1 - pad_x) / ratio
        y1 = (y1 - pad_y) / ratio
        x2 = (x2 - pad_x) / ratio
        y2 = (y2 - pad_y) / ratio
        
        # Clamp to image boundaries
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        
        bboxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # Prepare detections
        detections = []
        for bbox, conf, class_id in zip(bboxes, filtered_confidences, filtered_class_ids):
            detections.append({
                'bbox': [float(coord) for coord in bbox],
                'confidence': float(conf),
                'class_id': int(class_id),
                'class_name': self.class_names[class_id]
            })
        
        # Apply NMS
        return self._apply_nms(detections)
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            return []
        
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])
        
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        indices = np.argsort(scores)[::-1]
        
        keep_indices = []
        while indices.size > 0:
            current_idx = indices[0]
            keep_indices.append(current_idx)
            
            if indices.size == 1:
                break
            
            current_box = boxes[current_idx]
            remaining_boxes = boxes[indices[1:]]
            
            # Calculate IoU
            xx1 = np.maximum(current_box[0], remaining_boxes[:, 0])
            yy1 = np.maximum(current_box[1], remaining_boxes[:, 1])
            xx2 = np.minimum(current_box[2], remaining_boxes[:, 2])
            yy2 = np.minimum(current_box[3], remaining_boxes[:, 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[current_idx] + areas[indices[1:]] - inter)
            
            mask = iou < self.iou_threshold
            indices = indices[1:][mask]
        
        return [detections[i] for i in keep_indices]
    
    def detect(self, image_path: str, output_path: str = None, json_path: str = None, raw_npy_path: str = None) -> List[Dict]:
        """
        Perform detection on an image file
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            json_path: Path to save post-processed JSON results (optional)
            raw_npy_path: Path to save raw model output as numpy array (optional)
            
        Returns:
            List of detection dictionaries
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Processing: {image_path}")
        print(f"Shape: {image.shape}")
        
        # Preprocess
        input_tensor, info = self.preprocess(image)
        
        # Run inference (OpenVINO)
        self.infer_request.infer({0: input_tensor})
        predictions = self.infer_request.get_output_tensor(0).data

        # Save raw model output if requested
        if raw_npy_path:
            np.save(raw_npy_path, predictions)
            print(f"Raw model output saved as numpy array: {raw_npy_path}")
        
        # Postprocess
        detections = self.postprocess(predictions, info)
        
        print(f"Found {len(detections)} objects")
        
        # Draw results if output path provided
        if output_path:
            result_img = self.draw_detections(image, detections)
            cv2.imwrite(output_path, result_img)
            print(f"Image result saved: {output_path}")
        
        # Save post-processed JSON results if json_path provided
        if json_path:
            import json
            results_dict = {
                "image_path": image_path,
                "detections": detections,
                "count": len(detections)
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
            print(f"Post-processed JSON result saved: {json_path}")
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict], thickness: int = 2) -> np.ndarray:
        result_img = image.copy()
        
        for det in detections:
            bbox = [int(coord) for coord in det['bbox']]
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(result_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), thickness)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            top_left = (bbox[0], bbox[1] - label_size[1] - 10)
            bottom_right = (bbox[0] + label_size[0], bbox[1])
            cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), -1)
            cv2.putText(result_img, label, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return result_img


def main():
    parser = argparse.ArgumentParser(description='Simple YOLOv5-SU Single Image Inference')
    parser.add_argument('--model', type=str, default='../01-models/yolov5su_fp32.onnx', help='Path to ONNX model')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, help='Output image path')
    parser.add_argument('--json-output', type=str, help='Post-processed JSON output path')
    parser.add_argument('--raw-npy-output', type=str, help='Raw model output as numpy array (.npy)')
    parser.add_argument('--device', type=str, default='CPU', help='OpenVINO device (e.g. CPU)')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark and measure inference time')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.45, help='IoU threshold for NMS')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    
    # Parse model name to extract architecture and precision
    model_filename = os.path.basename(args.model)
    # Expected format: yolov5su_fp32.onnx, yolov5su_int8.onnx, etc.
    parts = model_filename.replace('.onnx', '').split('_')
    if len(parts) >= 2:
        model_arch = '_'.join(parts[:-1])  # yolov5su
        precision = parts[-1]              # fp32, fp16, int8
    else:
        model_arch = "unknown"
        precision = "unknown"
    
    # Create output paths with nested structure if not provided
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    
    # Create nested directory structure: outputs/{model_arch}/{precision}/
    output_subdir = f"outputs/{model_arch}/{precision}"
    os.makedirs(output_subdir, exist_ok=True)
    
    if not args.output:
        args.output = f"{output_subdir}/{base_name}_result.jpg"
    if not args.json_output:
        args.json_output = f"{output_subdir}/{base_name}_result.json"
    if not args.raw_npy_output:
        args.raw_npy_output = f"{output_subdir}/{base_name}_raw.npy"
    
    # Ensure output directory exists
    os.makedirs(output_subdir, exist_ok=True)
    
    # Run detection
    detector = YOLOv5SU(args.model, args.conf_threshold, args.iou_threshold, args.device)
    
    if args.benchmark:
        import time
        # Warm up
        for _ in range(3):
            _ = detector.detect(args.image)
        
        # Benchmark
        num_runs = 10
        start_time = time.time()
        for _ in range(num_runs):
            detections = detector.detect(args.image, args.output, args.json_output, args.raw_npy_output)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        fps = 1.0 / avg_time
        
        print(f"Benchmark completed over {num_runs} runs:")
        print(f"Average inference time: {avg_time:.4f}s ({avg_time*1000:.2f}ms)")
        print(f"Average FPS: {fps:.2f}")
    else:
        detections = detector.detect(args.image, args.output, args.json_output, args.raw_npy_output)
    
    # Print results
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class_name']}: {det['confidence']:.3f} at {det['bbox']}")


if __name__ == "__main__":
    main()