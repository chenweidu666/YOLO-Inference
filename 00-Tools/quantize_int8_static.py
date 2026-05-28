"""
Static INT8 quantization for CPU-compatible inference
Uses QLinearConv instead of ConvInteger (which needs special EP)
"""
import onnx
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
from onnxruntime.quantization import CalibrationMethod
import numpy as np
import cv2
import os


class DataReader:
    """Calibration data reader for static quantization"""
    def __init__(self, image_path, batch_size=1, img_size=640):
        self.image_path = image_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.count = 0
        
        # Generate diverse calibration samples
        self.samples = []
        for _ in range(50):  # 50 calibration samples
            img = self._preprocess()
            self.samples.append(img)
    
    def _preprocess(self):
        """Preprocess with variety: random crops, flips, brightness"""
        image = cv2.imread(self.image_path)
        h, w = image.shape[:2]
        
        # Random crop
        crop_ratio = np.random.uniform(0.7, 1.0)
        crop_w = int(w * crop_ratio)
        crop_h = int(h * crop_ratio)
        x_offset = np.random.randint(0, w - crop_w + 1) if w > crop_w else 0
        y_offset = np.random.randint(0, h - crop_h + 1) if h > crop_h else 0
        image = image[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w]
        
        # Random flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness
        brightness = np.random.uniform(0.8, 1.2)
        image = np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
        
        # Resize + letterbox (same as YOLOv5 preprocessing)
        h, w = image.shape[:2]
        ratio = min(self.img_size / w, self.img_size / h)
        new_w = int(round(w * ratio))
        new_h = int(round(h * ratio))
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        sx = (self.img_size - new_w) // 2
        sy = (self.img_size - new_h) // 2
        canvas[sy:sy+new_h, sx:sx+new_w] = resized
        
        rgb = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(rgb, axis=0)
    
    def get_next(self):
        """Get next calibration batch"""
        if self.count >= len(self.samples):
            return None
        batch = self.samples[self.count]
        self.count += 1
        return {"images": batch}


def main():
    input_path = "00-Models/yolov5su_fp32.onnx"
    output_path = "00-Models/yolov5su_int8_cpu.onnx"
    calib_image = "00-Data/zidane.jpg"
    
    if not os.path.exists(input_path):
        print(f"Model not found: {input_path}")
        return
    
    print(f"Static INT8 quantization...")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    
    data_reader = DataReader(calib_image)
    
    from onnxruntime.quantization.shape_inference import quant_pre_process
    # Upgrade opset for better quantization support
    preproc_path = input_path.replace('.onnx', '_preproc.onnx')
    quant_pre_process(input_path, preproc_path)
    
    quantize_static(
        model_input=preproc_path,
        model_output=output_path,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,  # Q/DQ pattern with upgraded opset
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.Entropy,
    )
    
    # Clean up temp file
    if os.path.exists(preproc_path):
        os.remove(preproc_path)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Size: {size_mb:.2f} MB")
    
    # Verify with onnxruntime
    import onnxruntime as ort
    try:
        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        print("  ✅ CPU EP loads successfully!")
    except Exception as e:
        print(f"  ❌ CPU EP load failed: {e}")
        return
    
    # Quick inference test
    img = cv2.imread(calib_image)
    h, w = img.shape[:2]
    ratio = min(640 / w, 640 / h)
    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    sx = (640 - new_w) // 2
    sy = (640 - new_h) // 2
    canvas[sy:sy+new_h, sx:sx+new_w] = resized
    rgb = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(rgb, axis=0)
    
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    print(f"  Output shape: {outputs[0].shape}")
    print(f"  ✅ Inference works!")


if __name__ == "__main__":
    main()
